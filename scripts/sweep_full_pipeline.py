"""
sweep_full_pipeline.py

Discovers upstream checkpoints produced by sweep_upstream_train.py, runs linear evaluation
(simclr_lin.py) for each, collects Γ (topological separation) and PGD robust accuracy,
then writes a merged CSV and scatter plot.

Expected upstream run layout (produced by sweep_upstream_train.py):
  runs/upstream/{method}_{backbone}_e{E}_seed{seed}/
    checkpoints/upstream/{method}/simclr_{method}_{backbone}_epoch{E}_seed{seed}.pt
    visuals/upstream/{method}/{method}_epoch{E}_seed{seed}_train_history.csv

Downstream outputs written to:
  runs/downstream/{method}_{backbone}_e{E}_seed{seed}/
    checkpoints/downstream/
    visuals/downstream/{method}/lin_history_{method}_{backbone}.csv

Summary written to:
  runs/summary/gamma_vs_pgd_merged.csv
  runs/summary/gamma_vs_pgd_scatter.png

Usage examples:
  # Eval all found checkpoints (baseline + phsim, all seeds)
  python sweep_full_pipeline.py --methods baseline,phsim

  # Only epoch 50 checkpoints, seeds 0 and 1
  python sweep_full_pipeline.py --methods baseline,phsim,hybrid --seeds 0,1 --only_epochs 50

  # Dry-run to preview commands
  python sweep_full_pipeline.py --methods baseline --dry_run
"""

import argparse
import re
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants / regex
# ---------------------------------------------------------------------------

CKPT_RE = re.compile(
    r"^simclr_(?P<method>baseline|phsim|hybrid)"
    r"_(?P<backbone>resnet18|resnet34)"
    r"_epoch(?P<epoch>\d+)"
    r"_seed(?P<seed>\d+)\.pt$"
)

UPSTREAM_ROOT = Path("runs/upstream")
DOWNSTREAM_ROOT = Path("runs/downstream")
SUMMARY_ROOT = Path("runs/summary")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], dry_run: bool = False, cwd: Optional[Path] = None):
    print("\n[CMD]", " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_ckpt_name(p: Path) -> Optional[dict]:
    m = CKPT_RE.match(p.name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "method":   d["method"],
        "backbone": d["backbone"],
        "epoch":    int(d["epoch"]),
        "seed":     int(d["seed"]),
        "path":     p,
    }


def discover_ckpts(
    methods: list[str],
    backbone: str,
    seeds: Optional[list[int]],
    only_epochs: Optional[list[int]],
) -> list[dict]:
    """
    Walk runs/upstream/{method}_{backbone}_e*_seed*/ and look for checkpoints inside
    each run dir (Hydra chdirs to hydra.run.dir so simclr.py writes relative to it).
    Also checks top-level checkpoints/ as fallback for manually-run jobs.
    """
    found = []
    for method in methods:
        search_dirs = []

        # primary: inside each hydra run dir, with seed subfolder
        for run_dir in sorted(UPSTREAM_ROOT.glob(f"{method}_{backbone}_e*_seed*")):
            for seed_dir in [
                run_dir / "checkpoints" / "upstream" / method / f"seed{run_dir.name.split('_seed')[-1]}",
                run_dir / "checkpoints" / "upstream" / method,  # flat fallback
            ]:
                if seed_dir.exists():
                    search_dirs.append(seed_dir)
                    break

        # fallback: top-level checkpoints/ (manually-run or old jobs), with seed subfolders
        for seed_subdir in (Path("checkpoints") / "upstream" / method).glob("seed*"):
            if seed_subdir.is_dir():
                search_dirs.append(seed_subdir)
        toplevel = Path("checkpoints") / "upstream" / method
        if toplevel.exists():
            search_dirs.append(toplevel)

        seen = set()
        for ckpt_dir in search_dirs:
            # new layout: seed{S}/epoch{E}/*.pt — recurse two levels
            for p in ckpt_dir.glob(f"epoch*/simclr_{method}_{backbone}_epoch*_seed*.pt"):
                if p.name in seen:
                    continue
                seen.add(p.name)
                info = parse_ckpt_name(p)
                if info is None:
                    continue
                if seeds is not None and info["seed"] not in seeds:
                    continue
                if only_epochs is not None and info["epoch"] not in only_epochs:
                    continue
                found.append(info)
            # flat fallback: seed{S}/*.pt (old layout)
            for p in ckpt_dir.glob(f"simclr_{method}_{backbone}_epoch*_seed*.pt"):
                if p.name in seen:
                    continue
                seen.add(p.name)
                info = parse_ckpt_name(p)
                if info is None:
                    continue
                if seeds is not None and info["seed"] not in seeds:
                    continue
                if only_epochs is not None and info["epoch"] not in only_epochs:
                    continue
                found.append(info)

    found.sort(key=lambda x: (x["method"], x["seed"], x["epoch"]))
    return found


def find_upstream_history_csv(method: str, seed: int) -> Optional[Path]:
    """
    Find the train_history CSV for a given method+seed.
    New layout: logs/upstream/{method}/seed{S}/{method}_seed{S}_train_history.csv
    Searches inside hydra run dirs first, then top-level fallback.
    """
    search_dirs = []

    # primary: logs/ inside each matching hydra run dir
    for run_dir in sorted(UPSTREAM_ROOT.glob(f"{method}_*_seed{seed}")):
        for subdir in [
            run_dir / "logs" / "upstream" / method / f"seed{seed}",   # new
            run_dir / "visuals" / "upstream" / method / f"seed{seed}", # old fallback
            run_dir / "visuals" / "upstream" / method,                 # older flat
        ]:
            if subdir.exists():
                search_dirs.append(subdir)

    # top-level fallbacks
    for subdir in [
        Path("logs")    / "upstream" / method / f"seed{seed}",
        Path("visuals") / "upstream" / method / f"seed{seed}",
        Path("visuals") / "upstream" / method,
    ]:
        if subdir.exists():
            search_dirs.append(subdir)

    for log_dir in search_dirs:
        cands = sorted(log_dir.glob(f"*seed{seed}*train_history*.csv"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]
        cands = sorted(log_dir.glob("*train_history*.csv"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]

    return None


def read_gamma_at_epoch(csv_path: Path, epoch: int) -> float:
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns or "gamma" not in df.columns:
        return float("nan")
    row = df[df["epoch"] == epoch]
    if not row.empty:
        return float(row["gamma"].iloc[0])
    # fallback: last recorded epoch ≤ requested
    row = df[df["epoch"] <= epoch].tail(1)
    return float(row["gamma"].iloc[0]) if not row.empty else float("nan")


def find_downstream_history_csv(run_dir: Path, method: str, backbone: str, seed: int) -> Path:
    """
    simclr_lin.py writes CSV to logs/downstream/{method}/seed{S}/ (new),
    with fallback to visuals/ (old layout).
    """
    for log_dir in [
        run_dir / "logs"    / "downstream" / method / f"seed{seed}",  # new
        run_dir / "visuals" / "downstream" / method / f"seed{seed}",  # old fallback
        run_dir / "visuals" / "downstream" / method,                  # older flat
        Path("logs")    / "downstream" / method / f"seed{seed}",      # top-level new
        Path("visuals") / "downstream" / method / f"seed{seed}",      # top-level old
        Path("visuals") / "downstream" / method,                      # top-level flat
    ]:
        if not log_dir.exists():
            continue
        canonical = log_dir / f"lin_history_{method}_{backbone}.csv"
        if canonical.exists():
            return canonical
        cands = sorted(log_dir.glob("lin_history_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]

    raise FileNotFoundError(
        f"Cannot find lin_history CSV for {method}/{backbone}/seed{seed} "
        f"under {run_dir}/logs/downstream/ or {run_dir}/visuals/downstream/"
    )


def extract_pgd_acc(csv_path: Path) -> tuple[float, float]:
    """Returns (final_pgd_acc, best_pgd_acc)."""
    df = pd.read_csv(csv_path)
    if "pgd_acc" not in df.columns:
        return float("nan"), float("nan")
    return float(df["pgd_acc"].iloc[-1]), float(df["pgd_acc"].max())


def extract_best_test_acc(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    if "test_acc" not in df.columns:
        return float("nan")
    return float(df["test_acc"].max())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

MARKER_BY_SEED = ["o", "s", "^", "D", "v", "P", "*"]
COLOR_BY_METHOD = {"baseline": "#1f77b4", "phsim": "#d62728", "hybrid": "#2ca02c"}


def plot_scatter(df: pd.DataFrame, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 5))

    for (method, seed), sub in df.groupby(["method", "seed"]):
        color  = COLOR_BY_METHOD.get(method, "grey")
        marker = MARKER_BY_SEED[int(seed) % len(MARKER_BY_SEED)]
        ax.scatter(
            sub["gamma"], sub["pgd_acc_best"],
            label=f"{method} seed{seed}",
            color=color, marker=marker, s=70, edgecolors="k", linewidths=0.5,
        )

    ax.set_xlabel("Topological Separation Γ (inter-class Wasserstein)", fontsize=11)
    ax.set_ylabel("PGD-10 Robust Accuracy (best over lin epochs)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved scatter → {out_png}")


def plot_clean_acc(df: pd.DataFrame, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 5))

    for (method, seed), sub in df.groupby(["method", "seed"]):
        color  = COLOR_BY_METHOD.get(method, "grey")
        marker = MARKER_BY_SEED[int(seed) % len(MARKER_BY_SEED)]
        ax.scatter(
            sub["gamma"], sub["best_test_acc"],
            label=f"{method} seed{seed}",
            color=color, marker=marker, s=70, edgecolors="k", linewidths=0.5,
        )

    ax.set_xlabel("Topological Separation Γ", fontsize=11)
    ax.set_ylabel("Best Clean Test Accuracy", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved clean-acc scatter → {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--backbone",    default="resnet18",      choices=["resnet18", "resnet34"])
    ap.add_argument("--methods",     default="baseline,phsim", help="Comma list: baseline,phsim,hybrid")
    ap.add_argument("--seeds",       default="",              help="Comma list of seeds (empty = all found)")
    ap.add_argument("--only_epochs", default="",              help="Comma list of upstream epochs (empty = all found)")

    # Downstream training
    ap.add_argument("--batch_size",      type=int, default=128)
    ap.add_argument("--finetune_epochs", type=int, default=10)

    # Attack params
    ap.add_argument("--eps",             type=float, default=8/255)
    ap.add_argument("--pgd_steps",       type=int,   default=10)
    ap.add_argument("--pgd_alpha",       type=float, default=2/255)
    ap.add_argument("--max_test_batches",type=int,   default=-1,
                    help="Limit test batches for speed (-1 = all).")

    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run downstream eval even if output dir already exists.")
    ap.add_argument("--dry_run",   action="store_true",
                    help="Print commands without executing.")

    args = ap.parse_args()

    methods     = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds       = [int(x) for x in args.seeds.split(",") if x.strip()] or None
    only_epochs = [int(x) for x in args.only_epochs.split(",") if x.strip()] or None
    backbone    = args.backbone

    # ── Discover checkpoints ────────────────────────────────────────────────
    ckpts = discover_ckpts(methods, backbone, seeds, only_epochs)
    if not ckpts:
        raise RuntimeError(
            "No checkpoints found.\n"
            f"Searched under: {UPSTREAM_ROOT.resolve()}\n"
            f"methods={methods}, backbone={backbone}, seeds={seeds}, only_epochs={only_epochs}"
        )

    print(f"[Info] Found {len(ckpts)} checkpoint(s):")
    for c in ckpts:
        print(f"  {c['method']:10s} seed={c['seed']} epoch={c['epoch']:4d}  {c['path']}")

    ensure_dir(SUMMARY_ROOT)
    merged_rows = []

    for info in ckpts:
        method   = info["method"]
        backbone = info["backbone"]
        up_epoch = info["epoch"]
        seed     = info["seed"]
        ckpt_src = info["path"]

        run_name = f"{method}_{backbone}_e{up_epoch}_seed{seed}"
        run_dir  = DOWNSTREAM_ROOT / run_name

        # ── Skip if already done (unless --overwrite) ───────────────────────
        lin_done = False
        try:
            csv_path = find_downstream_history_csv(run_dir, method, backbone, seed)
            lin_done = True
        except FileNotFoundError:
            pass

        if lin_done and not args.overwrite:
            print(f"[Skip] Already evaluated: {run_name}  (use --overwrite to redo)")
        else:
            ensure_dir(run_dir)

            # simclr_lin.py loads from checkpoints/upstream/{method}/simclr_...pt
            # relative to its cwd. Since we run it from the repo root (no cwd override),
            # it will find the checkpoint where simclr.py wrote it: top-level checkpoints/.

            # Pass absolute checkpoint path so simclr_lin.py can find it
            # regardless of what Hydra sets as cwd.
            ckpt_abs = ckpt_src.resolve()

            cmd = [
                "python", "simclr_lin.py",
                f"method={method}",
                f"backbone={backbone}",
                f"batch_size={args.batch_size}",
                f"load_epoch={up_epoch}",
                f"seed={seed}",
                f"finetune_epochs={args.finetune_epochs}",
                f"+ckpt_path={ckpt_abs}",  # + prefix appends new key to Hydra config
                "attack.enabled=true",
                "attack.fgsm=false",
                "attack.pgd=true",
                f"attack.eps={args.eps}",
                f"attack.pgd_steps={args.pgd_steps}",
                f"attack.pgd_alpha={args.pgd_alpha}",
                "attack.pgd_random_start=true",
                f"attack.max_test_batches={args.max_test_batches}",
                f"hydra.run.dir={run_dir.as_posix()}",
                "hydra.output_subdir=.hydra",
            ]
            run(cmd, dry_run=args.dry_run)

        # ── Read downstream results ─────────────────────────────────────────
        if args.dry_run:
            pgd_final = pgd_best = best_test_acc = float("nan")
        else:
            try:
                lin_csv = find_downstream_history_csv(run_dir, method, backbone, seed)
                pgd_final, pgd_best = extract_pgd_acc(lin_csv)
                best_test_acc       = extract_best_test_acc(lin_csv)
            except FileNotFoundError as e:
                print(f"[Warn] {e}")
                pgd_final = pgd_best = best_test_acc = float("nan")
                lin_csv = Path("N/A")

        # ── Read Γ from upstream history ────────────────────────────────────
        hist_csv = find_upstream_history_csv(method, seed)
        if hist_csv is not None and not args.dry_run:
            gamma_val = read_gamma_at_epoch(hist_csv, up_epoch)
        else:
            gamma_val = float("nan")

        row = {
            "method":          method,
            "backbone":        backbone,
            "seed":            seed,
            "upstream_epoch":  up_epoch,
            "gamma":           gamma_val,
            "pgd_acc_final":   pgd_final,
            "pgd_acc_best":    pgd_best,
            "best_test_acc":   best_test_acc,
            "ckpt_path":       str(ckpt_src),
            "lin_csv":         str(lin_csv) if not args.dry_run else "",
            "up_hist_csv":     str(hist_csv) if hist_csv else "",
        }
        merged_rows.append(row)

        print(
            f"[Result] {method:8s} seed={seed} upE={up_epoch:3d} | "
            f"Γ={gamma_val:.4f} | pgd_best={pgd_best:.4f} | clean_best={best_test_acc:.4f}"
        )

    # ── Summary CSV + plots ─────────────────────────────────────────────────
    merged = (
        pd.DataFrame(merged_rows)
        .sort_values(["method", "seed", "upstream_epoch"])
        .reset_index(drop=True)
    )

    merged_csv = SUMMARY_ROOT / "gamma_vs_pgd_merged.csv"
    merged.to_csv(merged_csv, index=False)
    print(f"\n[Summary] CSV → {merged_csv}")

    if not args.dry_run and not merged.empty:
        plot_scatter(
            merged,
            SUMMARY_ROOT / "gamma_vs_pgd_scatter.png",
            title=f"Topological Separation Γ vs PGD Robust Acc ({backbone})",
        )
        plot_clean_acc(
            merged,
            SUMMARY_ROOT / "gamma_vs_clean_acc_scatter.png",
            title=f"Topological Separation Γ vs Clean Accuracy ({backbone})",
        )

    print("\n[Done]")
    print(f"  Upstream checkpoints : {UPSTREAM_ROOT.resolve()}")
    print(f"  Downstream runs      : {DOWNSTREAM_ROOT.resolve()}")
    print(f"  Summary              : {SUMMARY_ROOT.resolve()}")


if __name__ == "__main__":
    main()