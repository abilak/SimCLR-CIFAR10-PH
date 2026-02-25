#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_eps_key(k: str) -> float:
    # keys are saved as strings of floats like "0.031372549..."
    return float(k)


def trapezoid_auc(xs, ys):
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    # assumes xs sorted
    return float(np.trapz(ys, xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/downstream", help="where your hydra run dirs live")
    ap.add_argument("--glob", default="**/logs/eps_curve_*.json", help="pattern under --root")
    ap.add_argument("--out_dir", default="agg_eps", help="output folder")
    ap.add_argument("--eps_px_ref", default="0,2,4,6,8,10",
                    help="reference eps grid in px; used for ordering + missing-value checks")
    ap.add_argument("--plot_by_upE", action="store_true",
                    help="also produce one plot per upstream epoch")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps_ref = [int(x.strip()) for x in args.eps_px_ref.split(",") if x.strip() != ""]
    eps_ref = [e / 255.0 for e in eps_ref]

    paths = sorted(root.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No JSONs found under {root} with glob={args.glob}")

    rows = []
    for p in paths:
        with open(p, "r") as f:
            d = json.load(f)

        method = d.get("method")
        seed = int(d.get("seed"))
        upE = int(d.get("load_epoch"))
        backbone = d.get("backbone", "unknown")
        clean = float(d.get("adv_clean_acc", float("nan")))

        pgd = d.get("pgd_acc_by_eps", {})
        # keys may be strings; normalize to float
        pgd_map = {parse_eps_key(k): float(v) for k, v in pgd.items()}

        # Keep only eps values present; later we align/intersect
        for eps, acc in pgd_map.items():
            rows.append({
                "method": method,
                "backbone": backbone,
                "seed": seed,
                "up_epoch": upE,
                "eps": float(eps),
                "rob_acc": float(acc),
                "clean_acc": clean,
                "json_path": str(p),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "eps_sweep_long.csv", index=False)

    # --- determine eps grid to report ---
    # safest: use intersection of eps values present across all runs (avoids NaNs)
    eps_vals_all = sorted(df["eps"].unique().tolist())
    # prefer the reference eps grid if present
    eps_grid = [e for e in eps_ref if any(abs(e - x) < 1e-9 for x in eps_vals_all)]
    if not eps_grid:
        eps_grid = eps_vals_all

    # --- aggregate: mean/std by method and up_epoch and eps ---
    agg = (
        df[df["eps"].isin(eps_grid)]
        .groupby(["method", "up_epoch", "eps"])
        .agg(
            n=("rob_acc", "count"),
            mean=("rob_acc", "mean"),
            std=("rob_acc", "std"),
        )
        .reset_index()
        .sort_values(["method", "up_epoch", "eps"])
    )
    agg.to_csv(out_dir / "eps_sweep_agg.csv", index=False)

    # --- compute summary metrics per (method, up_epoch, seed): AUC + point at 8/255 (if exists) ---
    eps_target = 8 / 255.0
    have_eps_target = any(abs(e - eps_target) < 1e-9 for e in eps_grid)

    summaries = []
    for (method, upE, seed), sub in df[df["eps"].isin(eps_grid)].groupby(["method", "up_epoch", "seed"]):
        sub = sub.sort_values("eps")
        xs = sub["eps"].to_numpy()
        ys = sub["rob_acc"].to_numpy()
        auc = trapezoid_auc(xs, ys)

        # robust acc at target eps (if present)
        if have_eps_target:
            y_t = float(sub.loc[np.isclose(sub["eps"], eps_target), "rob_acc"].iloc[0])
        else:
            y_t = float("nan")

        clean = float(sub["clean_acc"].iloc[0])

        summaries.append({
            "method": method,
            "up_epoch": int(upE),
            "seed": int(seed),
            "clean_acc": clean,
            "rob_acc_at_8_255": y_t,
            "auc_eps": auc,
        })

    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(out_dir / "eps_sweep_summary_by_seed.csv", index=False)

    sum_agg = (
        sum_df.groupby(["method", "up_epoch"])
        .agg(
            n=("seed", "count"),
            clean_mean=("clean_acc", "mean"),
            clean_std=("clean_acc", "std"),
            rob8_mean=("rob_acc_at_8_255", "mean"),
            rob8_std=("rob_acc_at_8_255", "std"),
            auc_mean=("auc_eps", "mean"),
            auc_std=("auc_eps", "std"),
        )
        .reset_index()
        .sort_values(["up_epoch", "method"])
    )
    sum_agg.to_csv(out_dir / "eps_sweep_summary_agg.csv", index=False)

    # --- plot: overall curves per up_epoch (mean ± std), baseline vs phsim ---
    # One combined plot per up_epoch by default (recommended for paper)
    up_epochs = sorted(agg["up_epoch"].unique().tolist())
    methods = sorted(agg["method"].unique().tolist())

    for upE in up_epochs:
        plt.figure()
        for method in methods:
            sub = agg[(agg["up_epoch"] == upE) & (agg["method"] == method)].sort_values("eps")
            if sub.empty:
                continue
            x = sub["eps"].to_numpy()
            y = sub["mean"].to_numpy()
            s = sub["std"].fillna(0.0).to_numpy()
            plt.plot(x, y, marker="o", label=f"{method} (n={int(sub['n'].max())})")
            plt.fill_between(x, y - s, y + s, alpha=0.2)

        plt.xlabel(r"$\epsilon$ ($\ell_\infty$)")
        plt.ylabel("PGD robust accuracy")
        plt.title(f"Robustness vs epsilon (upstream epoch {upE})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"eps_curve_upE{upE}.png", dpi=200)
        plt.close()

        if not args.plot_by_upE:
            # if not requested, still keep these plots (they're useful), but no extra work needed
            pass

    print(f"[OK] Wrote:\n"
          f"  {out_dir / 'eps_sweep_long.csv'}\n"
          f"  {out_dir / 'eps_sweep_agg.csv'}\n"
          f"  {out_dir / 'eps_sweep_summary_by_seed.csv'}\n"
          f"  {out_dir / 'eps_sweep_summary_agg.csv'}\n"
          f"  plus plots: {out_dir / 'eps_curve_upE*.png'}")


if __name__ == "__main__":
    main()