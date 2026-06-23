"""
Correctness tests for phtopo.diffph.

Validates:
  1. H0 deaths from the differentiable Prim's MST match GUDHI's Rips H0.
  2. Gradients flow through H0 deaths (gradcheck, double precision).
  3. Sliced-Wasserstein basic metric properties + gradient flow.
  4. SW between identical diagrams is ~0; SW grows with separation.
"""
import math
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phtopo.diffph import (
    vr_h0_persistence,
    vr_h0_persistence_batch,
    sliced_wasserstein_pd,
    sliced_wasserstein_h0,
    h0_diagram_from_deaths,
)


def gudhi_h0_deaths(X: np.ndarray) -> np.ndarray:
    import gudhi
    rc = gudhi.RipsComplex(points=X.astype(np.float64))
    st = rc.create_simplex_tree(max_dimension=1)
    st.compute_persistence()
    dgm = st.persistence_intervals_in_dimension(0)
    deaths = np.sort([d for (b, d) in dgm if np.isfinite(d)])
    return deaths


def test_h0_matches_gudhi():
    torch.manual_seed(0)
    np.random.seed(0)
    for _ in range(5):
        N, D = np.random.randint(5, 30), np.random.randint(2, 8)
        X = np.random.randn(N, D)
        ours = vr_h0_persistence(torch.tensor(X, dtype=torch.float64)).numpy()
        ref = gudhi_h0_deaths(X)
        ours = np.sort(ours)
        assert len(ours) == len(ref), f"count mismatch {len(ours)} vs {len(ref)}"
        assert np.allclose(ours, ref, atol=1e-6), f"\nours={ours}\nref ={ref}"
    print("[OK] H0 deaths match GUDHI (5 random clouds)")


def test_h0_batch_consistency():
    torch.manual_seed(1)
    X = torch.randn(4, 20, 5, dtype=torch.float64)
    batched = vr_h0_persistence_batch(X)
    for b in range(4):
        single = vr_h0_persistence(X[b])
        assert torch.allclose(batched[b], single, atol=1e-9)
    print("[OK] batched == per-sample H0")


def test_h0_gradcheck():
    torch.manual_seed(2)
    X = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    # sum of deaths is a smooth function of X away from distance ties
    ok = torch.autograd.gradcheck(
        lambda x: vr_h0_persistence(x).sum(), (X,), atol=1e-4, rtol=1e-3
    )
    assert ok
    print("[OK] H0 deaths gradcheck passed")


def test_sw_properties():
    torch.manual_seed(3)
    D1 = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 0.5]], dtype=torch.float64)
    D2 = D1.clone()
    assert float(sliced_wasserstein_pd(D1, D2)) < 1e-9, "SW(D,D) != 0"
    # symmetry
    Da = torch.tensor([[0.0, 1.0], [0.0, 3.0]], dtype=torch.float64)
    Db = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    s1 = float(sliced_wasserstein_pd(Da, Db))
    s2 = float(sliced_wasserstein_pd(Db, Da))
    assert abs(s1 - s2) < 1e-9, f"asymmetric {s1} {s2}"
    # monotonicity: pulling a diagram further from another increases SW
    base = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    near = torch.tensor([[0.0, 1.2]], dtype=torch.float64)
    far = torch.tensor([[0.0, 5.0]], dtype=torch.float64)
    assert float(sliced_wasserstein_pd(base, near)) < float(sliced_wasserstein_pd(base, far))
    # empty-diagram handling
    empty = torch.zeros((0, 2), dtype=torch.float64)
    assert float(sliced_wasserstein_pd(base, empty)) > 0
    assert float(sliced_wasserstein_pd(empty, empty)) == 0.0
    print("[OK] SW: identity=0, symmetric, monotone, empty-safe")


def test_sw_gradcheck():
    torch.manual_seed(4)
    D1 = torch.rand(4, 2, dtype=torch.float64, requires_grad=True)
    D2 = torch.rand(3, 2, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda a, b: sliced_wasserstein_pd(a, b, n_directions=16), (D1, D2),
        atol=1e-4, rtol=1e-3,
    )
    assert ok
    print("[OK] SW gradcheck passed")


def test_end_to_end_grad():
    # gradient should flow X -> H0 deaths -> SW -> scalar loss
    torch.manual_seed(5)
    Xpos = torch.randn(12, 4, dtype=torch.float64, requires_grad=True)
    Xneg = torch.randn(12, 4, dtype=torch.float64, requires_grad=True)
    dp = vr_h0_persistence(Xpos)
    dn = vr_h0_persistence(Xneg)
    gamma = sliced_wasserstein_h0(dp, dn, n_directions=16)
    gamma.backward()
    assert Xpos.grad is not None and Xpos.grad.abs().sum() > 0
    assert Xneg.grad is not None and Xneg.grad.abs().sum() > 0
    print(f"[OK] end-to-end grad through X->H0->SW (gamma={float(gamma):.4f})")


if __name__ == "__main__":
    test_h0_matches_gudhi()
    test_h0_batch_consistency()
    test_h0_gradcheck()
    test_sw_properties()
    test_sw_gradcheck()
    test_end_to_end_grad()
    print("\nAll diffph tests passed.")
