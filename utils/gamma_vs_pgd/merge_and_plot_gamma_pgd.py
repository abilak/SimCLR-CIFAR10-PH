import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def glob_all(pattern: str):
    return sorted(glob.glob(pattern, recursive=True))


def glob_first(pattern: str):
    m = glob_all(pattern)
    return m[0] if m else None


def parse_seed_from_path(p: str) -> int | None:
    # expects ".../seed0/..." or ".../seed12/..."
    parts = Path(p).parts
    for part in parts:
        if part.startswith("seed"):
            s = part.replace("seed", "")
            if s.isdigit():
                return int(s)
    return None


def load_upstream_gamma(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns or "gamma" not in df.columns:
        raise ValueError(f"Missing epoch/gamma in {csv_path}. cols={list(df.columns)}")
    df = df[["epoch", "gamma"]].copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
    df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)
    return df


def find_downstream_lin_csv(root: Path, method: str, seed: int, up_epoch: int) -> str | None:
    # robust search: allow nested hydra folders
    pats = [
        str(root / "logs" / "downstream" / method / f"seed{seed}" / f"upE{up_epoch}" / "**" / "lin_history_*.csv"),
        str(root / "logs" / "downstream" / method / f"seed{seed}" / f"upE{up_epoch}" / "**" / "*lin_history*.csv"),
    ]
    for p in pats:
        hit = glob_first(p)
        if hit:
            return hit
    return None


def extract_metrics_from_lin_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)

    if "epoch" not in df.columns or "test_acc" not in df.columns:
        raise ValueError(f"Missing epoch/test_acc in {csv_path}. cols={list(df.columns)}")

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["test_acc"] = pd.to_numeric(df["test_acc"], errors="coerce")
    if "pgd_acc" in df.columns:
        df["pgd_acc"] = pd.to_numeric(df["pgd_acc"], errors="coerce")
    else:
        df["pgd_acc"] = np.nan

    df = df.dropna(subset=["epoch", "test_acc"]).copy()
    df["epoch"] = df["epoch"].astype(int)

    # best clean
    idx_best = df["test_acc"].idxmax()
    best_clean_epoch = int(df.loc[idx_best, "epoch"])
    clean_at_best = float(df.loc[idx_best, "test_acc"])
    pgd_at_best = float(df.loc[idx_best, "pgd_acc"])

    # last
    idx_last = df["epoch"].idxmax()
    last_epoch = int(df.loc[idx_last, "epoch"])
    clean_at_last = float(df.loc[idx_last, "test_acc"])
    pgd_at_last = float(df.loc[idx_last, "pgd_acc"])

    # debug: best pgd
    pgd_best = float(df["pgd_acc"].max()) if df["pgd_acc"].notna().any() else float("nan")

    return dict(
        best_clean_lin_epoch=best_clean_epoch,
        clean_at_best_clean=clean_at_best,
        pgd_at_best_clean=pgd_at_best,
        last_lin_epoch=last_epoch,
        clean_at_last=clean_at_last,
        pgd_at_last=pgd_at_last,
        pgd_best_over_lin=pgd_best,
    )


def plot_scatter(df: pd.DataFrame, x: str, y: str, out_png: Path, title: str):
    plt.figure()
    for method, sub in df.groupby("method"):
        plt.scatter(sub[x], sub[y], s=55, label=method)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)  # no seed/epoch in title
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--methods", nargs="+", default=["baseline", "phsim"])
    ap.add_argument("--min_up_epoch", type=int, default=1)
    ap.add_argument("--max_up_epoch", type=int, default=10**9)
    ap.add_argument("--require_downstream", action="store_true",
                    help="if set, only keep rows that have downstream metrics")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    rows = []

    for method in args.methods:
        # find all upstream train histories for any seed
        upstream_pats = [
            str(root / "logs" / "upstream" / method / "seed*" / "*train_history*.csv"),
            str(root / "logs" / "upstream" / method / "seed*" / "**" / "*train_history*.csv"),
        ]
        upstream_csvs = []
        for p in upstream_pats:
            upstream_csvs.extend(glob_all(p))
        upstream_csvs = sorted(set(upstream_csvs))

        if not upstream_csvs:
            print(f"[warn] no upstream train_history found for method={method}")
            continue

        for up_csv in upstream_csvs:
            seed = parse_seed_from_path(up_csv)
            if seed is None:
                continue

            up_df = load_upstream_gamma(up_csv)
            up_df = up_df[(up_df["epoch"] >= args.min_up_epoch) & (up_df["epoch"] <= args.max_up_epoch)].copy()

            for _, r in up_df.iterrows():
                up_epoch = int(r["epoch"])
                gamma = float(r["gamma"])
                if not np.isfinite(gamma):
                    continue

                lin_csv = find_downstream_lin_csv(root, method, seed, up_epoch)
                if lin_csv is None:
                    if args.require_downstream:
                        continue
                    # keep upstream-only row if desired
                    rows.append({
                        "method": method,
                        "seed": seed,
                        "up_epoch": up_epoch,
                        "gamma": gamma,
                        "downstream_csv": None,
                    })
                    continue

                met = extract_metrics_from_lin_csv(lin_csv)
                rows.append({
                    "method": method,
                    "seed": seed,
                    "up_epoch": up_epoch,
                    "gamma": gamma,
                    "downstream_csv": lin_csv,
                    **met,
                })

    if not rows:
        print("[FAIL] no rows merged. likely missing logs paths.")
        return

    df = pd.DataFrame(rows)

    # If downstream exists for some rows, keep only those for plotting
    has_down = df["downstream_csv"].notna()
    df_plot = df[has_down].copy()

    out_csv = root / "merged_all_methods_all_seeds.csv"
    df.to_csv(out_csv, index=False)
    print(f"[ok] wrote merged CSV: {out_csv}")
    print(f"[info] total rows={len(df)} | with downstream={len(df_plot)}")

    if df_plot.empty:
        print("[FAIL] no downstream rows found; cannot plot.")
        return

    # Plots (best-clean aligned + last-epoch aligned)
    plot_scatter(df_plot, "gamma", "pgd_at_best_clean",
                 root / "scatter_gamma_vs_pgd_bestclean.png",
                 r"$\Gamma$ vs PGD Robust Accuracy (Best-Clean Epoch)")

    plot_scatter(df_plot, "gamma", "clean_at_best_clean",
                 root / "scatter_gamma_vs_clean_bestclean.png",
                 r"$\Gamma$ vs Clean Accuracy (Best-Clean Epoch)")

    plot_scatter(df_plot, "gamma", "pgd_at_last",
                 root / "scatter_gamma_vs_pgd_last.png",
                 r"$\Gamma$ vs PGD Robust Accuracy (Last Epoch)")

    plot_scatter(df_plot, "gamma", "clean_at_last",
                 root / "scatter_gamma_vs_clean_last.png",
                 r"$\Gamma$ vs Clean Accuracy (Last Epoch)")

    print("[ok] wrote 4 scatter plots.")


if __name__ == "__main__":
    main()