"""
phtopo.robustness
=================

Robustness evaluation suite for Tier-1 #1. Goes well beyond a single PGD-10
number, which post-2018 reviewers reject because TDA-style regularizers can
induce gradient masking that inflates PGD. The suite computes, at eps=8/255 (and
a sweep), everything needed to *detect* masking and to report a credible robust
accuracy:

  1. Clean accuracy.
  2. PGD step-count curve {10,20,50,100}: genuine robustness PLATEAUS; if accuracy
     keeps collapsing as steps grow, PGD was simply under-optimized.
  3. PGD random-restart curve: robustness should be ~flat across restarts.
  4. Epsilon sweep up to large eps: robust accuracy MUST approach 0 as eps grows;
     if it does not, gradients are masked.
  5. Square attack (gradient-free) via AutoAttack: bypasses masking entirely.
  6. AutoAttack standard ensemble (APGD-CE, APGD-T, FAB-T, Square): the field
     standard; reported as the headline robust accuracy.
  7. Black-box transfer from a surrogate (e.g., a baseline-trained model).

The suite then emits explicit masking flags and a verdict. AutoAttack/Square use
the official `autoattack` package when installed; if absent, those entries are
marked "unavailable" (install `autoattack` on the run machine) and the verdict
relies on the PGD step/eps/restart curves + transfer.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

from .attacks import pgd_linf, accuracy_under, transfer_attack


def _iter_batches(loader, max_batches: int):
    for bidx, (x, y) in enumerate(loader):
        if max_batches > 0 and bidx >= max_batches:
            break
        yield x, y


@torch.no_grad()
def _clean_accuracy(model, loader, device, max_batches: int) -> float:
    correct, n = 0, 0
    for x, y in _iter_batches(loader, max_batches):
        x, y = x.to(device), y.to(device)
        correct += int((model(x).argmax(1) == y).sum())
        n += x.size(0)
    return correct / max(1, n)


def _pgd_accuracy(model, loader, device, eps, steps, restarts, max_batches) -> float:
    correct, n = 0, 0
    for x, y in _iter_batches(loader, max_batches):
        x, y = x.to(device), y.to(device)
        x_adv = pgd_linf(model, x, y, eps=eps, steps=steps, restarts=restarts)
        correct += int((model(x_adv).argmax(1) == y).sum())
        n += x.size(0)
    return correct / max(1, n)


def _autoattack_accuracy(model, loader, device, eps, max_batches, version="standard") -> Dict[str, float]:
    """
    Run AutoAttack (and report Square separately). Returns dict possibly empty if
    the package is unavailable.
    """
    try:
        from autoattack import AutoAttack
    except Exception:
        return {"autoattack": float("nan"), "square": float("nan"), "available": 0.0}

    # gather a fixed subset of test points
    xs, ys = [], []
    for x, y in _iter_batches(loader, max_batches):
        xs.append(x); ys.append(y)
    X = torch.cat(xs).to(device); Y = torch.cat(ys).to(device)

    out: Dict[str, float] = {"available": 1.0}
    adversary = AutoAttack(model, norm="Linf", eps=eps, version=version, verbose=False)
    x_adv = adversary.run_standard_evaluation(X, Y, bs=min(256, X.size(0)))
    with torch.no_grad():
        out["autoattack"] = float((model(x_adv).argmax(1) == Y).float().mean())

    # Square alone (gradient-free) — strongest single masking probe
    adv_sq = AutoAttack(model, norm="Linf", eps=eps, version="standard", verbose=False)
    adv_sq.attacks_to_run = ["square"]
    x_sq = adv_sq.run_standard_evaluation(X, Y, bs=min(256, X.size(0)))
    with torch.no_grad():
        out["square"] = float((model(x_sq).argmax(1) == Y).float().mean())
    return out


def _transfer_accuracy(target, surrogate, loader, device, eps, steps, max_batches) -> float:
    correct, n = 0, 0
    for x, y in _iter_batches(loader, max_batches):
        x, y = x.to(device), y.to(device)
        x_adv = transfer_attack(target, surrogate, x, y, eps=eps, steps=steps)
        with torch.no_grad():
            correct += int((target(x_adv).argmax(1) == y).sum())
        n += x.size(0)
    return correct / max(1, n)


def run_robustness_suite(
    model: Callable,
    loader,
    device: str,
    eps: float = 8 / 255,
    pgd_steps: List[int] = (10, 20, 50, 100),
    pgd_restarts: List[int] = (1, 5),
    eps_sweep_px: List[float] = (2, 4, 8, 16, 32, 64, 128),
    surrogate: Optional[Callable] = None,
    run_autoattack: bool = True,
    max_batches: int = -1,
) -> Dict:
    """
    Run the full suite and return a structured result with masking diagnostics
    and a verdict. `model` and `surrogate` are callables mapping images->logits
    on `device`; images are in [0,1].
    """
    model_eval = model
    if hasattr(model, "eval"):
        model.eval()

    res: Dict = {"eps": float(eps), "eps_px": round(eps * 255, 3)}
    res["clean_acc"] = _clean_accuracy(model_eval, loader, device, max_batches)

    # 2. PGD step-count curve
    res["pgd_by_steps"] = {
        int(s): _pgd_accuracy(model_eval, loader, device, eps, s, 1, max_batches)
        for s in pgd_steps
    }
    # 3. PGD restart curve at the largest step budget
    smax = max(pgd_steps)
    res["pgd_by_restarts"] = {
        int(r): _pgd_accuracy(model_eval, loader, device, eps, smax, r, max_batches)
        for r in pgd_restarts
    }
    # 4. Epsilon sweep (PGD-50), incl. large eps sanity
    res["pgd_by_eps_px"] = {
        float(e): _pgd_accuracy(model_eval, loader, device, e / 255.0, 50, 1, max_batches)
        for e in eps_sweep_px
    }
    # 5/6. AutoAttack + Square
    if run_autoattack:
        res["autoattack"] = _autoattack_accuracy(model_eval, loader, device, eps, max_batches)
    # 7. Transfer
    if surrogate is not None:
        res["transfer_acc"] = _transfer_accuracy(model_eval, surrogate, loader, device, eps, smax, max_batches)

    res["masking"] = _masking_verdict(res)
    return res


def _masking_verdict(res: Dict, plateau_tol: float = 0.03, large_eps_tol: float = 0.05) -> Dict:
    """
    Heuristic gradient-masking flags. Each flag True == evidence of masking.
    """
    flags: Dict[str, object] = {}
    pgd = res.get("pgd_by_steps", {})

    # (a) PGD does not plateau: large gap between fewest and most steps means PGD
    #     was under-optimized (a mild masking signal).
    if pgd:
        ks = sorted(pgd.keys())
        flags["pgd_not_plateaued"] = bool((pgd[ks[0]] - pgd[ks[-1]]) > plateau_tol * 3)

    # (b) Large-eps robustness not vanishing -> strong masking signal.
    eps_curve = res.get("pgd_by_eps_px", {})
    if eps_curve:
        big = max(eps_curve.keys())
        flags["large_eps_not_zero"] = bool(eps_curve[big] > large_eps_tol)

    # (c) Gradient-free / ensemble << white-box PGD -> masking.
    pgd_best = min(pgd.values()) if pgd else float("nan")
    aa = res.get("autoattack", {})
    if aa.get("available", 0.0) == 1.0:
        flags["square_gap"] = bool((pgd_best - aa.get("square", pgd_best)) > 0.05)
        flags["autoattack_gap"] = bool((pgd_best - aa.get("autoattack", pgd_best)) > 0.05)

    # (d) Transfer breaks it far more than white-box -> masking.
    if "transfer_acc" in res and pgd:
        flags["transfer_gap"] = bool((pgd_best - res["transfer_acc"]) > 0.05)

    any_flag = any(bool(v) for v in flags.values())
    flags["verdict"] = "POSSIBLE GRADIENT MASKING" if any_flag else "no masking signal detected"
    # The credible headline number: AutoAttack if available, else worst-case PGD.
    if aa.get("available", 0.0) == 1.0:
        flags["robust_acc_headline"] = aa.get("autoattack")
    else:
        flags["robust_acc_headline"] = pgd_best
    return flags
