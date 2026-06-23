"""
phtopo.stats
============

Honest statistics for the Gamma <-> robustness claim (Tier-1 #4) and the
matched-clean-accuracy control (Tier-1 #2).

Everything here uses only numpy/scipy/pandas and runs anywhere (no GPU). The
prototype these functions productionize already showed, on the committed
seed-1 data, that the raw Spearman is indistinguishable from zero and vanishes
under epoch control. These helpers re-run that analysis *pooled across all seeds
and checkpoints* once the full sweep exists.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


def spearman_with_inference(x: np.ndarray, y: np.ndarray, n_boot: int = 10000,
                            seed: int = 0) -> Dict[str, float]:
    """Spearman rho with p-value, n, and a bootstrap 95% CI."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": n,
                "ci_low": float("nan"), "ci_high": float("nan")}
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    idx = np.arange(n)
    for _ in range(n_boot):
        s = rng.choice(idx, n, replace=True)
        if np.std(x[s]) == 0 or np.std(y[s]) == 0:
            continue
        boots.append(stats.spearmanr(x[s], y[s])[0])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))
    return {"rho": float(rho), "p": float(p), "n": int(n),
            "ci_low": float(lo), "ci_high": float(hi)}


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    """
    Partial Spearman correlation of x,y controlling for z: Pearson correlation of
    the rank-residuals of x and y after linearly regressing each on rank(z).
    Controls the epoch confound (both Gamma and robustness drift with epoch).
    """
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 4:
        return {"rho_partial": float("nan"), "p": float("nan"), "n": n}
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    rz = pd.Series(z).rank().values

    def resid(a, b):
        B = np.c_[np.ones_like(b), b]
        beta, *_ = np.linalg.lstsq(B, a, rcond=None)
        return a - B @ beta

    ex, ey = resid(rx, rz), resid(ry, rz)
    r, _ = stats.pearsonr(ex, ey)
    dof = n - 3
    t = r * np.sqrt(dof / max(1e-12, 1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), dof)
    return {"rho_partial": float(r), "p": float(p), "n": int(n)}


def correlation_report(df: pd.DataFrame, gamma_col: str, pgd_col: str,
                       epoch_col: str = "epoch", method_col: str = "method") -> Dict:
    """
    Full Gamma<->PGD correlation analysis pooled across seeds/checkpoints, overall
    and per method, with bootstrap CI and epoch-controlled partial correlation.
    """
    out: Dict = {"gamma_col": gamma_col, "pgd_col": pgd_col}

    def block(sub: pd.DataFrame) -> Dict:
        g, p = sub[gamma_col].values, sub[pgd_col].values
        d = {"spearman": spearman_with_inference(g, p)}
        if epoch_col in sub.columns:
            d["partial_spearman_epoch"] = partial_spearman(g, p, sub[epoch_col].values)
            d["epoch_confound"] = {
                "spearman_gamma_epoch": float(stats.spearmanr(g, sub[epoch_col].values)[0]),
                "spearman_pgd_epoch": float(stats.spearmanr(p, sub[epoch_col].values)[0]),
            }
        return d

    out["overall"] = block(df)
    out["by_method"] = {m: block(s) for m, s in df.groupby(method_col)} if method_col in df.columns else {}
    return out


def matched_clean_control(df: pd.DataFrame, clean_col: str, pgd_col: str,
                          method_col: str = "method",
                          ph_method: str = "phsim", baseline_method: str = "baseline",
                          tol: float = 0.02) -> Dict:
    """
    Matched-clean-accuracy control (Tier-1 #2). For each PHSim checkpoint, find
    baseline checkpoints whose clean accuracy is within `tol`, and compare PGD
    robustness at matched clean accuracy. If the matched baseline is equally
    robust, the "topology" story reduces to the accuracy-robustness tradeoff.
    """
    ph = df[df[method_col] == ph_method]
    base = df[df[method_col] == baseline_method]
    pairs = []
    for _, r in ph.iterrows():
        cand = base[(base[clean_col] - r[clean_col]).abs() <= tol]
        if len(cand) == 0:
            continue
        pairs.append({
            "clean": float(r[clean_col]),
            "phsim_pgd": float(r[pgd_col]),
            "baseline_pgd_matched_mean": float(cand[pgd_col].mean()),
            "baseline_pgd_matched_max": float(cand[pgd_col].max()),
            "n_matched_baseline": int(len(cand)),
        })
    if not pairs:
        return {"n_pairs": 0, "note": "no clean-accuracy matches within tol; widen tol or sweep more epochs"}
    pdf = pd.DataFrame(pairs)
    adv = pdf["phsim_pgd"] - pdf["baseline_pgd_matched_mean"]
    return {
        "n_pairs": len(pdf),
        "tol": tol,
        "mean_phsim_pgd": float(pdf["phsim_pgd"].mean()),
        "mean_baseline_pgd_at_matched_clean": float(pdf["baseline_pgd_matched_mean"].mean()),
        "mean_robustness_advantage_of_phsim": float(adv.mean()),
        "advantage_holds": bool(adv.mean() > 0.01),
        "interpretation": (
            "PHSim more robust THAN baseline at matched clean accuracy -> not just the tradeoff"
            if adv.mean() > 0.01 else
            "PHSim NOT more robust than baseline at matched clean -> consistent with generic tradeoff"
        ),
    }
