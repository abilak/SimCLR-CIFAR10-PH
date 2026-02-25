#!/usr/bin/env python3
"""
analyze_gamma_vs_robustness.py

Reads merged_all_methods_all_seeds.csv and produces:
1) Correlation between gamma (Γ) and PGD robust accuracy (Pearson + Spearman), overall and per method
2) Linear regression slope gamma -> PGD (overall and per method), with R^2
3) Mean/std of gamma and PGD (and clean) per method
4) Updated scatter plots:
   - Γ vs clean_at_best_clean
   - Γ vs pgd_at_best_clean
   Each plot includes per-method best-fit line, and (optional) overall fit.

Usage:
  python analyze_gamma_vs_robustness.py \
      --csv merged_all_methods_all_seeds.csv \
      --outdir outputs_gamma_analysis \
      --xcol gamma \
      --clean_col clean_at_best_clean \
      --pgd_col pgd_at_best_clean \
      --methods baseline phsim

Notes:
- Uses only standard scientific Python: pandas, numpy, matplotlib.
- Saves a summary CSV and a JSON with stats for Overleaf copy/paste.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearmanr_safe(x: np.ndarray, y: np.ndarray) -> float:
    # Spearman = Pearson correlation of ranks
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return pearsonr_safe(xr, yr)


def linreg_1d(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Ordinary least squares y = a + b x.
    Returns: intercept a, slope b, r2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0:
        return {"intercept": float("nan"), "slope": float("nan"), "r2": float("nan")}
    b, a = np.polyfit(x, y, deg=1)  # y ~ b*x + a
    yhat = b * x + a
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"intercept": float(a), "slope": float(b), "r2": float(r2)}


def summarize_method(df: pd.DataFrame, xcol: str, ycols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["n"] = int(len(df))
    out["mean"] = {c: float(df[c].mean()) for c in [xcol] + ycols}
    out["std"] = {c: float(df[c].std(ddof=1)) for c in [xcol] + ycols}
    for y in ycols:
        out[f"pearson_{xcol}_vs_{y}"] = pearsonr_safe(df[xcol].to_numpy(), df[y].to_numpy())
        out[f"spearman_{xcol}_vs_{y}"] = spearmanr_safe(df[xcol].to_numpy(), df[y].to_numpy())
        out[f"linreg_{xcol}_to_{y}"] = linreg_1d(df[xcol].to_numpy(), df[y].to_numpy())
    return out


def plot_scatter_with_fits(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    out_png: Path,
    title: str,
    methods: List[str],
    add_overall_fit: bool = True,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    plt.figure()

    # Scatter per method
    for m in methods:
        sub = df[df["method"] == m]
        if len(sub) == 0:
            continue
        plt.scatter(sub[xcol], sub[ycol], s=60, label=m)

        # Per-method fit line
        reg = linreg_1d(sub[xcol].to_numpy(), sub[ycol].to_numpy())
        if np.isfinite(reg["slope"]) and np.isfinite(reg["intercept"]):
            xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 100)
            ys = reg["slope"] * xs + reg["intercept"]
            plt.plot(xs, ys, linewidth=2, linestyle="--")

    # Optional overall fit line
    if add_overall_fit:
        reg_all = linreg_1d(df[xcol].to_numpy(), df[ycol].to_numpy())
        if np.isfinite(reg_all["slope"]) and np.isfinite(reg_all["intercept"]):
            xs = np.linspace(df[xcol].min(), df[xcol].max(), 150)
            ys = reg_all["slope"] * xs + reg_all["intercept"]
            plt.plot(xs, ys, linewidth=2, linestyle="-", label="overall fit")

    plt.title(title)
    plt.xlabel(xlabel or xcol)
    plt.ylabel(ylabel or ycol)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str, help="Path to merged_all_methods_all_seeds.csv")
    ap.add_argument("--outdir", default="outputs_gamma_analysis", type=str)
    ap.add_argument("--xcol", default="gamma", type=str)
    ap.add_argument("--clean_col", default="clean_at_best_clean", type=str)
    ap.add_argument("--pgd_col", default="pgd_at_best_clean", type=str)
    ap.add_argument("--methods", nargs="*", default=None, help="Subset of methods to keep (e.g., baseline phsim)")
    ap.add_argument("--add_overall_fit", action="store_true", help="Add an overall regression line on plots")
    ap.add_argument(
        "--use_best_pgd_over_lin",
        action="store_true",
        help="If set, uses pgd_best_over_lin as y instead of pgd_col for PGD analysis/plot",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Basic cleaning / type coercion
    needed = ["method", args.xcol, args.clean_col, args.pgd_col, "seed", "up_epoch"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {csv_path}")

    # Coerce numeric cols
    for c in [args.xcol, args.clean_col, args.pgd_col, "seed", "up_epoch"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional PGD metric switch
    pgd_metric = "pgd_best_over_lin" if args.use_best_pgd_over_lin else args.pgd_col
    if pgd_metric not in df.columns:
        raise ValueError(f"Missing column '{pgd_metric}' in {csv_path}")

    df[pgd_metric] = pd.to_numeric(df[pgd_metric], errors="coerce")

    # Drop NA
    df = df.dropna(subset=["method", args.xcol, args.clean_col, pgd_metric])

    # Filter methods
    if args.methods:
        df = df[df["method"].isin(args.methods)]
        methods = list(args.methods)
    else:
        methods = sorted(df["method"].unique().tolist())

    if len(df) == 0:
        raise RuntimeError("No rows left after filtering/NA drops.")

    # === Stats summary ===
    ycols = [args.clean_col, pgd_metric]
    summary: Dict[str, Any] = {}

    # Overall
    summary["overall"] = summarize_method(df, args.xcol, ycols)

    # Per method
    summary["by_method"] = {}
    for m in methods:
        sub = df[df["method"] == m]
        summary["by_method"][m] = summarize_method(sub, args.xcol, ycols)

    # Write JSON (easy to read in Overleaf)
    json_path = outdir / "gamma_analysis_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Also write a tidy CSV table for quick inspection
    rows = []
    for key in ["overall"] + [f"method:{m}" for m in methods]:
        if key == "overall":
            block = summary["overall"]
            label = "overall"
        else:
            m = key.split("method:", 1)[1]
            block = summary["by_method"][m]
            label = m

        rows.append({
            "group": label,
            "n": block["n"],
            f"mean_{args.xcol}": block["mean"][args.xcol],
            f"std_{args.xcol}": block["std"][args.xcol],
            f"mean_{args.clean_col}": block["mean"][args.clean_col],
            f"std_{args.clean_col}": block["std"][args.clean_col],
            f"mean_{pgd_metric}": block["mean"][pgd_metric],
            f"std_{pgd_metric}": block["std"][pgd_metric],
            f"pearson_{args.xcol}_vs_{args.clean_col}": block[f"pearson_{args.xcol}_vs_{args.clean_col}"],
            f"pearson_{args.xcol}_vs_{pgd_metric}": block[f"pearson_{args.xcol}_vs_{pgd_metric}"],
            f"spearman_{args.xcol}_vs_{args.clean_col}": block[f"spearman_{args.xcol}_vs_{args.clean_col}"],
            f"spearman_{args.xcol}_vs_{pgd_metric}": block[f"spearman_{args.xcol}_vs_{pgd_metric}"],
            f"slope_{args.xcol}_to_{args.clean_col}": block[f"linreg_{args.xcol}_to_{args.clean_col}"]["slope"],
            f"r2_{args.xcol}_to_{args.clean_col}": block[f"linreg_{args.xcol}_to_{args.clean_col}"]["r2"],
            f"slope_{args.xcol}_to_{pgd_metric}": block[f"linreg_{args.xcol}_to_{pgd_metric}"]["slope"],
            f"r2_{args.xcol}_to_{pgd_metric}": block[f"linreg_{args.xcol}_to_{pgd_metric}"]["r2"],
        })

    summary_df = pd.DataFrame(rows)
    summary_csv_path = outdir / "gamma_analysis_summary_table.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # === Plots ===
    plot_scatter_with_fits(
        df=df,
        xcol=args.xcol,
        ycol=args.clean_col,
        out_png=outdir / "gamma_vs_clean_best_clean_epoch.png",
        title="Γ vs Clean Accuracy (Best-Clean Epoch)",
        methods=methods,
        add_overall_fit=args.add_overall_fit,
        xlabel="gamma (Γ)",
        ylabel=args.clean_col,
    )

    plot_scatter_with_fits(
        df=df,
        xcol=args.xcol,
        ycol=pgd_metric,
        out_png=outdir / "gamma_vs_pgd_best_clean_epoch.png",
        title=f"Γ vs PGD Robust Accuracy ({'Best PGD over linear eval' if args.use_best_pgd_over_lin else 'Best-Clean Epoch'})",
        methods=methods,
        add_overall_fit=args.add_overall_fit,
        xlabel="gamma (Γ)",
        ylabel=pgd_metric,
    )

    # Print quick console summary (copyable)
    print("\n=== Saved outputs ===")
    print("JSON:", json_path)
    print("Table CSV:", summary_csv_path)
    print("Plot 1:", outdir / "gamma_vs_clean_best_clean_epoch.png")
    print("Plot 2:", outdir / "gamma_vs_pgd_best_clean_epoch.png")

    print("\n=== Quick stats (overall) ===")
    overall = summary["overall"]
    print("n:", overall["n"])
    print(f"Pearson(Γ, {pgd_metric}):", overall[f"pearson_{args.xcol}_vs_{pgd_metric}"])
    print(f"Spearman(Γ, {pgd_metric}):", overall[f"spearman_{args.xcol}_vs_{pgd_metric}"])
    reg = overall[f"linreg_{args.xcol}_to_{pgd_metric}"]
    print(f"LinReg {pgd_metric} = a + b*Γ  |  b:", reg["slope"], "R^2:", reg["r2"])

    print("\n=== Per-method PGD correlation ===")
    for m in methods:
        block = summary["by_method"][m]
        print(
            f"{m:>10s}  n={block['n']:3d}  "
            f"Pearson={block[f'pearson_{args.xcol}_vs_{pgd_metric}']:.4f}  "
            f"Slope={block[f'linreg_{args.xcol}_to_{pgd_metric}']['slope']:.6f}  "
            f"R2={block[f'linreg_{args.xcol}_to_{pgd_metric}']['r2']:.4f}"
        )


if __name__ == "__main__":
    main()