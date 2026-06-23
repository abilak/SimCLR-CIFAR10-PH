"""
phtopo.descriptors
==================

Evaluation-only topological descriptors (no backprop needed here). Uses GUDHI's
fast C++ Vietoris-Rips for H0 *and* H1 on the small class-conditioned point
clouds, then sliced-Wasserstein (from phtopo.diffph) for diagram comparison.

Used by:
  * the fast class-separation Gamma eval during training,
  * the mechanism test (Tier-1 #3): Betti numbers, persistence entropy, total
    persistence of class-conditioned embeddings, clean vs adversarial.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .diffph import sliced_wasserstein_pd


# ---------------------------------------------------------------------------
# Diagrams via GUDHI
# ---------------------------------------------------------------------------
def _standardize(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    X = X - X.mean(0, keepdims=True)
    X = X / (X.std(0, keepdims=True) + eps)
    return X.astype(np.float64)


def rips_diagrams(X: np.ndarray, maxdim: int = 1, standardize: bool = True) -> Dict[int, np.ndarray]:
    """
    Vietoris-Rips persistence diagrams of a point cloud via GUDHI.

    Returns {dim: (n_dim, 2) array of finite (birth, death) points}. The single
    infinite H0 bar (the whole-space component) is dropped.
    """
    import gudhi

    if standardize:
        X = _standardize(X)
    rc = gudhi.RipsComplex(points=X)
    st = rc.create_simplex_tree(max_dimension=maxdim + 1)
    st.compute_persistence()
    out: Dict[int, np.ndarray] = {}
    for d in range(maxdim + 1):
        pairs = st.persistence_intervals_in_dimension(d)
        if len(pairs) == 0:
            out[d] = np.zeros((0, 2), dtype=np.float64)
            continue
        pairs = np.asarray(pairs, dtype=np.float64)
        finite = pairs[np.isfinite(pairs).all(axis=1)]
        finite = finite[(finite[:, 1] - finite[:, 0]) > 1e-9]
        out[d] = finite if finite.size else np.zeros((0, 2), dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Scalar descriptors (numpy)
# ---------------------------------------------------------------------------
def n_features(dgm: np.ndarray, min_persistence: float = 0.0) -> int:
    """Number of off-diagonal points with persistence > min_persistence."""
    if dgm.shape[0] == 0:
        return 0
    pers = dgm[:, 1] - dgm[:, 0]
    return int((pers > min_persistence).sum())


def total_persistence_np(dgm: np.ndarray, p: float = 1.0) -> float:
    if dgm.shape[0] == 0:
        return 0.0
    pers = np.clip(dgm[:, 1] - dgm[:, 0], 0, None)
    return float((pers ** p).sum())


def persistence_entropy_np(dgm: np.ndarray, eps: float = 1e-12) -> float:
    if dgm.shape[0] == 0:
        return 0.0
    pers = np.clip(dgm[:, 1] - dgm[:, 0], 0, None)
    L = pers.sum()
    if L <= eps:
        return 0.0
    pn = pers / L
    return float(-(pn * np.log(pn + eps)).sum())


def topology_descriptors(X: np.ndarray, maxdim: int = 1, min_persistence: float = 0.0) -> Dict[str, float]:
    """
    Full descriptor bundle for one point cloud. This is the per-class measurement
    used by the mechanism test (clean vs adversarial).
    """
    dgms = rips_diagrams(X, maxdim=maxdim)
    d0, d1 = dgms[0], dgms.get(1, np.zeros((0, 2)))
    return {
        "n_h0": n_features(d0, min_persistence),
        "n_h1": n_features(d1, min_persistence),
        "total_pers_h0": total_persistence_np(d0),
        "total_pers_h1": total_persistence_np(d1),
        "pers_entropy_h0": persistence_entropy_np(d0),
        "pers_entropy_h1": persistence_entropy_np(d1),
    }


# ---------------------------------------------------------------------------
# Class-conditioned separation Gamma (paper-faithful eval)
# ---------------------------------------------------------------------------
def class_separation_gamma(
    feats_by_class: Dict[int, np.ndarray],
    w_h0: float = 0.2,
    w_h1: float = 1.0,
    maxdim: int = 1,
    n_directions: int = 50,
) -> float:
    """
    Mean weighted sliced-Wasserstein distance between class-conditioned
    persistence diagrams over all class pairs. This is the eval-time Gamma(f):
    larger = more topological separation between classes.

    Replaces the old ripser + persim.wasserstein implementation; uses GUDHI for
    diagrams and the differentiable SW (run under no_grad) for diagram distance,
    so eval and training measure the *same* SW functional.
    """
    classes = sorted(feats_by_class.keys())
    dgms0, dgms1 = {}, {}
    for c in classes:
        d = rips_diagrams(feats_by_class[c], maxdim=maxdim)
        dgms0[c] = torch.tensor(d[0], dtype=torch.float64)
        dgms1[c] = torch.tensor(d.get(1, np.zeros((0, 2))), dtype=torch.float64)

    dists: List[float] = []
    with torch.no_grad():
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                a, b = classes[i], classes[j]
                d0 = float(sliced_wasserstein_pd(dgms0[a], dgms0[b], n_directions))
                d1 = float(sliced_wasserstein_pd(dgms1[a], dgms1[b], n_directions))
                dists.append(w_h0 * d0 + w_h1 * d1)
    return float(np.mean(dists)) if dists else 0.0
