"""
phtopo.diffph
=============

Differentiable, batched, GPU-friendly persistent homology primitives for the
PH-ACL / PHSim project.

Why this module exists
----------------------
The original implementation computed persistence with `ripser` (CPU) per sample
inside a Python loop and Wasserstein distances with `persim` (scipy Hungarian
assignment, CPU). That path is (a) extremely slow and (b) *non-differentiable* —
diagrams were detached to numpy, so no gradient ever flowed through the topology.
The paper's theory, however, treats Gamma (a sliced-Wasserstein distance between
persistence diagrams) as a Lipschitz, differentiable quantity that training
controls. This module closes that gap.

What it provides
----------------
* `vr_h0_persistence(X)` and `vr_h0_persistence_batch(X)`:
    0-dimensional Vietoris-Rips persistence of a Euclidean point cloud. H0 deaths
    are exactly the edge weights of the Euclidean minimum spanning tree (the
    single-linkage merge distances). We compute the MST *structure* with a
    vectorized batched Prim's algorithm, then read the death values straight out
    of the (differentiable) pairwise-distance tensor, so gradients flow to the
    input coordinates. This is exact (matches GUDHI/ripser H0) and runs entirely
    on-device with no CPU sync.

* `sliced_wasserstein_pd(D1, D2)`:
    Differentiable sliced-Wasserstein distance between two persistence diagrams,
    following Carriere, Cuturi & Oudot (2017). Handles the diagonal correctly by
    augmenting each diagram with the diagonal projections of the other. Uses
    `torch.sort` (sub-differentiable) so it backpropagates.

The H0 path is the workhorse: it is exact, fast, fully differentiable, and is
precisely the regime the paper's Sections 5-7 analyze (0-dim persistence mass).
For H1 (loops), see `phtopo.h1` which wraps an external engine for the
evaluation-only diagnostics where backprop is not required.

All functions are dtype/device agnostic and assume the *last* dimension is the
coordinate dimension.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Pairwise distances
# ---------------------------------------------------------------------------
def _pairwise_dist(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Euclidean pairwise distances, computed WITHOUT torch.cdist.

    We use the expansion ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b and take
    sqrt(. + eps) so the gradient is finite at zero distance (the diagonal),
    avoiding the classic cdist/pdist NaN-at-zero gradient. This also sidesteps
    `aten::_cdist_backward`, which is unimplemented on Apple MPS, so the
    differentiable PH path runs natively on CPU, CUDA, and MPS.

    Args:
        X: (..., N, D) point cloud(s).
    Returns:
        (..., N, N) distance matrix, differentiable w.r.t. X.
    """
    x2 = (X * X).sum(-1)                                   # (..., N)
    gram = torch.matmul(X, X.transpose(-1, -2))            # (..., N, N)
    sq = x2.unsqueeze(-1) + x2.unsqueeze(-2) - 2.0 * gram  # (..., N, N)
    sq = sq.clamp_min(0.0)
    return torch.sqrt(sq + eps)


# ---------------------------------------------------------------------------
# 0-dimensional VR persistence (differentiable, batched)
# ---------------------------------------------------------------------------
def vr_h0_persistence_batch(X: torch.Tensor) -> torch.Tensor:
    """
    Batched 0-dim Vietoris-Rips persistence via a vectorized Prim's MST.

    For a point cloud, every point is born at filtration value 0 and the (N-1)
    finite H0 deaths are exactly the MST edge weights. We grow the MST with
    Prim's algorithm, which at each of the N-1 steps adds the cheapest edge
    crossing the cut; that edge weight is the next death time. The death value is
    *gathered from the differentiable distance matrix*, so gradients flow to X.

    Args:
        X: (B, N, D) batch of point clouds (N >= 2).
    Returns:
        deaths: (B, N-1) tensor of H0 death times, sorted ascending per cloud.
                Births are all 0 (the corresponding H0 diagram is
                {(0, d) : d in deaths}). Differentiable w.r.t. X.
    """
    if X.dim() != 3:
        raise ValueError(f"expected (B, N, D), got shape {tuple(X.shape)}")
    B, N, _ = X.shape
    if N < 2:
        return X.new_zeros((B, 0))

    dist = _pairwise_dist(X)  # (B, N, N), differentiable
    device = X.device
    INF = torch.finfo(dist.dtype).max

    visited = torch.zeros(B, N, dtype=torch.bool, device=device)
    visited[:, 0] = True
    # mindist[b, j] = current cheapest distance from the growing tree to node j
    mindist = dist[:, 0, :].clone()  # (B, N), differentiable
    mindist[:, 0] = INF  # node 0 already in tree

    deaths = []
    barange = torch.arange(B, device=device)
    for _ in range(N - 1):
        # cheapest node to attach next (masked argmin over unvisited)
        masked = mindist.masked_fill(visited, INF)
        new_node = torch.argmin(masked, dim=1)  # (B,)
        # death time = the differentiable distance that attaches new_node
        death = mindist[barange, new_node]  # (B,), carries gradient
        deaths.append(death)
        visited[barange, new_node] = True
        # relax: distances from the newly added node to all others
        new_dist = dist[barange, new_node, :]  # (B, N)
        mindist = torch.minimum(mindist, new_dist)
        mindist[barange, new_node] = INF

    deaths = torch.stack(deaths, dim=1)  # (B, N-1)
    deaths, _ = torch.sort(deaths, dim=1)
    return deaths


def vr_h0_persistence(X: torch.Tensor) -> torch.Tensor:
    """Single-cloud convenience wrapper. X: (N, D) -> deaths: (N-1,)."""
    return vr_h0_persistence_batch(X.unsqueeze(0)).squeeze(0)


def h0_diagram_from_deaths(deaths: torch.Tensor) -> torch.Tensor:
    """
    Convert a (..., M) tensor of H0 deaths into a (..., M, 2) diagram with
    births = 0. Keeps gradients.
    """
    births = torch.zeros_like(deaths)
    return torch.stack([births, deaths], dim=-1)


# ---------------------------------------------------------------------------
# Sliced-Wasserstein distance between persistence diagrams (differentiable)
# ---------------------------------------------------------------------------
def _diagonal_projection(D: torch.Tensor) -> torch.Tensor:
    """
    Orthogonal projection of each (birth, death) point onto the diagonal y = x:
    (b, d) -> ((b+d)/2, (b+d)/2). Keeps gradients. D: (..., M, 2).
    """
    m = 0.5 * (D[..., 0] + D[..., 1])
    return torch.stack([m, m], dim=-1)


def sliced_wasserstein_pd(
    D1: torch.Tensor,
    D2: torch.Tensor,
    n_directions: int = 50,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Differentiable sliced-Wasserstein distance between two persistence diagrams.

    Implements the SW distance of Carriere, Cuturi & Oudot (2017): each diagram
    is augmented with the diagonal projections of the *other* diagram so the two
    augmented sets have equal cardinality, then for a set of directions theta the
    1D Wasserstein-1 distance between the projected, sorted coordinates is
    averaged. We average over uniformly spaced directions in [-pi/2, pi/2]
    (a deterministic quadrature of the SW integral) and normalize by pi so the
    value matches the standard SW1 definition up to the quadrature error.

    Args:
        D1: (P, 2) persistence diagram (birth, death). May be empty (P == 0).
        D2: (Q, 2) persistence diagram. May be empty.
        n_directions: number of projection directions (quadrature points).
    Returns:
        scalar tensor, differentiable w.r.t. D1 and D2.

    Edge cases:
        * If both diagrams are empty, returns 0.
        * If exactly one is empty, the distance reduces to the total persistence
          mass of the non-empty diagram against the diagonal (handled by the
          augmentation: the empty side contributes only diagonal projections).
    """
    P = D1.shape[0]
    Q = D2.shape[0]
    # establish a reference tensor for device/dtype even when one side is empty
    ref = D1 if P > 0 else D2
    if P == 0 and Q == 0:
        return ref.new_zeros(()) if ref.numel() else torch.zeros((), device=D1.device)

    # Augment: A = D1 + diag-proj(D2);  B = D2 + diag-proj(D1)
    A = torch.cat([D1, _diagonal_projection(D2)], dim=0)  # (P+Q, 2)
    Bset = torch.cat([D2, _diagonal_projection(D1)], dim=0)  # (P+Q, 2)

    # Deterministic quadrature of directions over the half-circle.
    thetas = torch.linspace(
        -math.pi / 2, math.pi / 2, steps=n_directions + 1, device=ref.device, dtype=ref.dtype
    )[:-1]
    dirs = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)  # (K, 2)

    projA = A @ dirs.t()  # (P+Q, K)
    projB = Bset @ dirs.t()  # (P+Q, K)

    projA, _ = torch.sort(projA, dim=0)
    projB, _ = torch.sort(projB, dim=0)

    # W1 in 1D between equal-size sets = mean abs diff of sorted coords.
    w1_per_dir = (projA - projB).abs().mean(dim=0)  # (K,)
    sw = w1_per_dir.mean() * math.pi / math.pi  # average over directions ~ SW1/pi * pi
    return sw


def sliced_wasserstein_h0(
    deaths1: torch.Tensor,
    deaths2: torch.Tensor,
    n_directions: int = 50,
) -> torch.Tensor:
    """
    Convenience: SW distance between two H0 diagrams given only their death
    vectors (births = 0). Differentiable.
    """
    D1 = h0_diagram_from_deaths(deaths1)
    D2 = h0_diagram_from_deaths(deaths2)
    return sliced_wasserstein_pd(D1, D2, n_directions=n_directions)


def sliced_wasserstein_h0_batch(
    deaths_a: torch.Tensor,
    deaths_b: torch.Tensor,
    n_directions: int = 50,
) -> torch.Tensor:
    """
    Fully vectorized SW between *aligned batches* of H0 diagrams. No Python loop,
    so this is the GPU hot-path primitive for the training loss.

    Both inputs have shape (B, P): B diagrams, each with P death values (births
    are 0). The two batches must share P (true when all clouds have the same
    number of points N, since P = N - 1). Returns (B,) SW distances,
    differentiable w.r.t. both inputs.

    Diagonal handling: H0 points are (0, d), whose diagonal projection is
    (d/2, d/2). The augmented set for each side has 2P points.
    """
    if deaths_a.dim() != 2 or deaths_b.dim() != 2:
        raise ValueError("expected (B, P) death tensors")
    B, P = deaths_a.shape
    if deaths_b.shape != deaths_a.shape:
        raise ValueError(f"shape mismatch {tuple(deaths_a.shape)} vs {tuple(deaths_b.shape)}")
    ref = deaths_a
    # Diagrams: A points (0, da); diag-proj of B points (db/2, db/2)
    # Augmented A_aug = [ (0,da_i) ] ++ [ (db_j/2, db_j/2) ]   -> (B, 2P, 2)
    zeros = torch.zeros_like(deaths_a)
    A_pts = torch.stack([zeros, deaths_a], dim=-1)                 # (B,P,2)
    A_diag = torch.stack([deaths_b / 2, deaths_b / 2], dim=-1)     # (B,P,2)
    A_aug = torch.cat([A_pts, A_diag], dim=1)                      # (B,2P,2)
    B_pts = torch.stack([zeros, deaths_b], dim=-1)
    B_diag = torch.stack([deaths_a / 2, deaths_a / 2], dim=-1)
    B_aug = torch.cat([B_pts, B_diag], dim=1)                      # (B,2P,2)

    thetas = torch.linspace(
        -math.pi / 2, math.pi / 2, steps=n_directions + 1, device=ref.device, dtype=ref.dtype
    )[:-1]
    dirs = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)  # (K,2)

    projA = A_aug @ dirs.t()   # (B, 2P, K)
    projB = B_aug @ dirs.t()
    projA, _ = torch.sort(projA, dim=1)
    projB, _ = torch.sort(projB, dim=1)
    w1 = (projA - projB).abs().mean(dim=1)  # (B, K)
    return w1.mean(dim=1)                    # (B,)


# ---------------------------------------------------------------------------
# Diagram summaries (differentiable where it matters)
# ---------------------------------------------------------------------------
def total_persistence(D: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """Sum of (death - birth)^p over off-diagonal points. D: (M, 2)."""
    if D.shape[0] == 0:
        return D.new_zeros(())
    pers = (D[:, 1] - D[:, 0]).clamp_min(0.0)
    return (pers ** p).sum()


def persistence_entropy(D: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Persistence entropy: -sum_i (l_i / L) log(l_i / L), l_i = persistence,
    L = total persistence. Standard TDA descriptor. D: (M, 2).
    """
    if D.shape[0] == 0:
        return D.new_zeros(())
    pers = (D[:, 1] - D[:, 0]).clamp_min(0.0)
    L = pers.sum()
    if float(L) <= eps:
        return D.new_zeros(())
    pn = pers / L
    return -(pn * (pn + eps).log()).sum()
