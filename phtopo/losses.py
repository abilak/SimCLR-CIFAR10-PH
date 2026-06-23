"""
phtopo.losses
=============

Differentiable training objectives built on the PH primitives in
`phtopo.diffph`. These REPLACE the old `ph_rank_loss`, whose persistence
diagrams were detached to numpy (no gradient through topology) and which only
used PH to *select* a hard negative for a cosine ranking hinge.

Design of the differentiable persistent-separation objective
------------------------------------------------------------
A SimCLR batch contains 2B augmented views, interleaved as
[img0_v1, img0_v2, img1_v1, img1_v2, ...], so the positive partner of view i is
its sibling (i^1). For each view we form a local point cloud from its reduced
backbone feature map (H*W points in R^c), and summarize it with its H0
persistence diagram PD_i = PD(Z_i). The persistent-separation functional from
the paper is Gamma(i, j) = SW(PD_i, PD_j).

We optimize a topological triplet:

    L = mean_i  relu( margin + Gamma(i, i^1) - min_{j in neg(i)} Gamma(i, j) )

Minimizing L pulls each anchor's persistence diagram TOWARD its positive
partner's (Gamma(i, i^1) small) and PUSHES it away from its hardest negative
(the negative with the *smallest* current separation). Crucially the gradient
flows X -> H0 deaths -> SW -> L, so the encoder is shaped to produce
topologically separated neighborhoods. This is a faithful, differentiable
instantiation of "control Gamma" and aligns the implementation with the theory.

Controls (for Tier-1 #2: topology-specificity)
----------------------------------------------
`raw_sw_separation_loss` uses the IDENTICAL triplet structure and the SAME
sliced-Wasserstein machinery, but computes SW directly between the *raw point
clouds* (no persistence / no MST). It is matched in form and magnitude and has a
comparable Lipschitz profile, differing only in that the persistent-homology
step is removed. If PHSim's robustness gain survives swapping the topological
loss for this control, the effect is "some Wasserstein penalty between
neighborhoods," not topology.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from .diffph import (
    vr_h0_persistence_batch,
    sliced_wasserstein_h0_batch,
)


# ---------------------------------------------------------------------------
# Point-cloud construction & standardization
# ---------------------------------------------------------------------------
def feature_map_to_pointcloud(h_map: torch.Tensor, num_points: int = None) -> torch.Tensor:
    """
    (B, C, H, W) -> (B, N, C) point cloud. Optionally subsample N = num_points
    spatial locations on a regular grid (deterministic, differentiable).
    """
    B, C, H, W = h_map.shape
    pts = h_map.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
    N = H * W
    if num_points is not None and num_points < N:
        idx = torch.linspace(0, N - 1, steps=num_points, device=h_map.device).long()
        pts = pts[:, idx, :]
    return pts


def standardize_pointcloud_batch(pts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-cloud standardization (zero mean, unit std over the point axis), matching
    the eval-time `_standardize_np`. Differentiable. pts: (B, N, C).
    """
    mu = pts.mean(dim=1, keepdim=True)
    sd = pts.std(dim=1, keepdim=True)
    return (pts - mu) / (sd + eps)


# ---------------------------------------------------------------------------
# Negative sampling (vectorized)
# ---------------------------------------------------------------------------
def _sample_negatives(n_views: int, neg_k: int, device, generator=None) -> torch.Tensor:
    """
    For each anchor i in [0, n_views), sample neg_k negatives excluding i and its
    positive partner i^1. Returns (n_views, neg_k) long tensor of indices.

    Implementation: draw uniform scores, mask out self+partner, take top-k. Fully
    on-device, no Python loop over anchors.
    """
    scores = torch.rand(n_views, n_views, device=device, generator=generator)
    ar = torch.arange(n_views, device=device)
    partner = ar ^ 1
    scores[ar, ar] = -1.0
    scores[ar, partner] = -1.0
    neg_k = min(neg_k, n_views - 2)
    _, idx = torch.topk(scores, k=neg_k, dim=1)
    return idx  # (n_views, neg_k)


# ---------------------------------------------------------------------------
# Differentiable persistent-separation triplet loss
# ---------------------------------------------------------------------------
def topo_separation_loss(
    h_map: torch.Tensor,
    num_points: int = 25,
    neg_k: int = 4,
    margin: float = 1.0,
    n_directions: int = 32,
    standardize: bool = True,
    generator: torch.Generator = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Differentiable PH separation triplet loss.

    Args:
        h_map: (2B, C, H, W) reduced feature maps for the 2B interleaved views.
        num_points, neg_k, margin, n_directions: see module docstring.
    Returns:
        (loss, stats) where stats holds detached scalars for logging:
        gamma_pos, gamma_neg (mean hardest-negative separation), and gamma_sep
        (= gamma_neg - gamma_pos, the realized topological separation).
    """
    n_views = h_map.shape[0]
    pts = feature_map_to_pointcloud(h_map, num_points=num_points)
    if standardize:
        pts = standardize_pointcloud_batch(pts)

    deaths = vr_h0_persistence_batch(pts)          # (2B, P)
    partner = (torch.arange(n_views, device=h_map.device) ^ 1)
    deaths_partner = deaths[partner]               # (2B, P)

    gamma_pos = sliced_wasserstein_h0_batch(deaths, deaths_partner, n_directions)  # (2B,)

    neg_idx = _sample_negatives(n_views, neg_k, h_map.device, generator)  # (2B, k)
    k = neg_idx.shape[1]
    anchor_rep = deaths.repeat_interleave(k, dim=0)                        # (2B*k, P), contiguous
    neg_rep = deaths[neg_idx.reshape(-1)]                                  # (2B*k, P)
    gamma_neg_all = sliced_wasserstein_h0_batch(anchor_rep, neg_rep, n_directions)
    gamma_neg_all = gamma_neg_all.reshape(n_views, k)
    gamma_neg_hard, _ = gamma_neg_all.min(dim=1)                          # (2B,) hardest neg

    loss = F.relu(margin + gamma_pos - gamma_neg_hard).mean()

    stats = {
        "gamma_pos": float(gamma_pos.mean().detach()),
        "gamma_neg": float(gamma_neg_hard.mean().detach()),
        "gamma_sep": float((gamma_neg_hard - gamma_pos).mean().detach()),
    }
    return loss, stats


# ---------------------------------------------------------------------------
# Non-topological control: sliced-Wasserstein on RAW point clouds (no PH)
# ---------------------------------------------------------------------------
def _sliced_wasserstein_pointcloud_batch(
    A: torch.Tensor, Bc: torch.Tensor, n_directions: int = 32, generator=None
) -> torch.Tensor:
    """
    SW distance between aligned batches of raw point clouds (NO persistence).
    A, Bc: (M, N, C) -- M clouds of N points in R^C. Returns (M,).
    Projects points onto random unit directions in R^C, sorts, L1. Differentiable.
    This is the matched non-topological functional for the Tier-1 #2 control.
    """
    M, N, C = A.shape
    dirs = torch.randn(C, n_directions, device=A.device, dtype=A.dtype, generator=generator)
    dirs = dirs / (dirs.norm(dim=0, keepdim=True) + 1e-8)
    pa = A @ dirs   # (M, N, K)
    pb = Bc @ dirs
    pa, _ = torch.sort(pa, dim=1)
    pb, _ = torch.sort(pb, dim=1)
    return (pa - pb).abs().mean(dim=(1, 2))  # (M,)


def raw_sw_separation_loss(
    h_map: torch.Tensor,
    num_points: int = 25,
    neg_k: int = 4,
    margin: float = 1.0,
    n_directions: int = 32,
    standardize: bool = True,
    generator: torch.Generator = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Non-topological control loss: identical triplet structure to
    `topo_separation_loss`, but separation is sliced-Wasserstein between the RAW
    point clouds (no MST / no persistence diagram). Use as `method=swcontrol`.
    """
    n_views = h_map.shape[0]
    pts = feature_map_to_pointcloud(h_map, num_points=num_points)
    if standardize:
        pts = standardize_pointcloud_batch(pts)

    partner = (torch.arange(n_views, device=h_map.device) ^ 1)
    gamma_pos = _sliced_wasserstein_pointcloud_batch(pts, pts[partner], n_directions, generator)

    neg_idx = _sample_negatives(n_views, neg_k, h_map.device, generator)
    k = neg_idx.shape[1]
    anchor_rep = pts.repeat_interleave(k, dim=0)
    neg_rep = pts[neg_idx.reshape(-1)]
    gamma_neg_all = _sliced_wasserstein_pointcloud_batch(anchor_rep, neg_rep, n_directions, generator)
    gamma_neg_hard, _ = gamma_neg_all.reshape(n_views, k).min(dim=1)

    loss = F.relu(margin + gamma_pos - gamma_neg_hard).mean()
    stats = {
        "gamma_pos": float(gamma_pos.mean().detach()),
        "gamma_neg": float(gamma_neg_hard.mean().detach()),
        "gamma_sep": float((gamma_neg_hard - gamma_pos).mean().detach()),
    }
    return loss, stats
