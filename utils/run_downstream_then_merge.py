"""
run_downstream_then_merge.py

For each method and each upstream checkpoint epoch in {10,20,30,40,50}:
  - runs simclr_lin.py downstream evaluation
  - reads downstream lin_history CSV to extract clean/pgd metrics
  - reads upstream train_history CSV to extract gamma at that upstream epoch
  - merges into one CSV and produces plots

Assumes:
  - upstream ckpts exist at:
      checkpoints/upstream/{method}/seed{seed}/epoch{E}/simclr_{method}_{backbone}_epoch{E}_seed{seed}.pt
    OR you can pass --ckpt_root to point elsewhere.

  - upstream gamma history exists at:
      logs/upstream/{method}/seed{seed}/*train_history*.csv
    (it will pick the newest one)

Outputs:
  {summary_dir}/gamma_vs_acc_merged.csv
  {summary_dir}/gamma_vs_clean.png
  {summary_dir}/gamma_vs_pgd.png
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


COLOR_BY_METHOD = {"baseline": "#1f77b4", "phsim": "#d62728", "hybrid": "#2ca02c"}
MARKER_BY_SEED = ["o", "s", "^", "D", "v", "P", "*"]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], dry_run: bool = False):
    print("\n[CMD]", " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def newest_csv(dir_: Path, pattern: str) -> Path:
    cands = sorted(dir_.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No files {pattern} under {dir_}")
    return cands[0]


def upstream_history_csv(method: str, seed: int) -> Path:
    d = Path("logs") / "upstream" / method / f"seed{seed}"
    if not d.exists():
        raise FileNotFoundError(f"Missing upstream log dir: {d}")
    return newest_csv(d, "*train_history*.csv")


def gamma_at_epoch(hist_csv: Path, epoch: int) -> float:
    df = pd.read_csv(hist_csv)
    if "epoch" not in df.columns or "gamma" not in df.columns:
        return float("nan")
    row = df[df["epoch"] == epoch]
    if not row.empty:
        return float(row["gamma"].iloc[0])
    row = df[df["epoch"] <= epoch].tail(1)
    return float(row["gamma"].iloc[0]) if not row.empty else float("nan")


def ckpt_path(ckpt_root: Path, method: str, backbone: str, seed: int, up_epoch: int) -> Path:
    p = (
        ckpt_root
        / method
        / f"seed{seed}"
        / f"epoch{up_epoch}"
        / f"simclr_{method}_{backbone}_epoch{up_epoch}_seed{seed}.pt"
    )
    if not p.exists():
        raise FileNotFoundError(f"Missing checkpoint: {p}")
    return p


def downstream_run_dir(method: str, seed: int, up_epoch: int) -> Path:
    # You said "all in logs nothing in runs" -> keep everything in logs/
    return Path("logs") / "downstream" / method / f"seed{seed}" / f"upE{up_epoch}"


def downstream_history_csv(run_dir: Path, method: str, backbone: str) -> Path:
    # simclr_lin.py writes: logs/downstream/{method}/seed{seed}/lin_history_{method}_{backbone}.csv
    # BUT since we set hydra.run.dir=run_dir, it will instead write to run_dir/logs/downstream/...
    # In your current simclr_lin.py, log_dir = out_dir/logs/downstream/{method}/seed{seed}
    # where out_dir == hydra.run.dir.
    p = run_dir / "logs" / "downstream" / method / f"seed{run_dir.parent.name.replace('seed','')}" / f"lin_history_{method}_{backbone}.csv"

    # The above is a bit hacky; better: just glob inside run_dir for lin_history.
    if p.exists():
        return p

    cands = sorted(run_dir.rglob(f"lin_history_{method}_{backbone}.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]

    raise FileNotFoundError(f"Missing downstream CSV under {run_dir} (searched recursively)")


def extract_downstream_metrics(lin_csv: Path) -> dict:
    df = pd.read_csv(lin_csv)
    out = {}
    out["clean_best"] = float(df["test_acc"].max()) if "test_acc" in df.columns else float("nan")
    out["clean_final"] = float(df["test_acc"].iloc[-1]) if "test_acc" in df.columns else float("nan")
    out["pgd_best"] = float(df["pgd_acc"].max()) if "pgd_acc" in df.columns else float("nan")
    out["pgd_final"] = float(df["pgd_acc"].iloc[-1]) if "pgd_acc" in df.columns else float("nan")
    return out


def scatter(df: pd.DataFrame, ycol: str, out_png: Path, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for (method, seed), sub in df.groupby(["method", "seed"]):
        ax.scatter(
            sub["gamma"], sub[ycol],
            label=f"{method} seed{seed}",
            color=COLOR_BY_METHOD.get(method, "grey"),
            marker=MARKER_BY_SEED[int(seed) % len(MARKER_BY_SEED)],
            s=80, edgecolors="k", linewidths=0.5,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                str(int(r["upstream_epoch"])),
                (r["gamma"], r[ycol]),
                fontsize=8,
                xytext=(4, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("Topological Separation Γ", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[Saved] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"])
    ap.add_argument("--methods", default="baseline,phsim", help="comma list")
    ap.add_argument("--seeds", default="0", help="comma list, e.g. 0 or 0,1,2")
    ap.add_argument("--up_epochs", default="10,20,30,40,50", help="comma list upstream epochs")
    ap.add_argument("--finetune_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)

    # attack knobs
    ap.add_argument("--eps", type=float, default=2/255)
    ap.add_argument("--pgd_steps", type=int, default=5)
    ap.add_argument("--pgd_alpha", type=float, default=1/255)
    ap.add_argument("--max_test_batches", type=int, default=-1)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--summary_dir", default="runs/summary")
    ap.add_argument("--ckpt_root", default="checkpoints/upstream")

    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    up_epochs = [int(x) for x in args.up_epochs.split(",") if x.strip()]
    backbone = args.backbone
    ckpt_root = Path(args.ckpt_root)

    rows = []

    for method in methods:
        for seed in seeds:
            hist_csv = upstream_history_csv(method, seed)

            for up_epoch in up_epochs:
                g = gamma_at_epoch(hist_csv, up_epoch)
                ckpt = ckpt_path(ckpt_root, method, backbone, seed, up_epoch).resolve()

                run_dir = downstream_run_dir(method, seed, up_epoch)

                # if downstream csv already exists and not overwrite, skip running
                lin_csv_exists = False
                if run_dir.exists():
                    try:
                        _ = newest_csv(run_dir, f"**/lin_history_{method}_{backbone}.csv")
                        lin_csv_exists = True
                    except FileNotFoundError:
                        lin_csv_exists = False

                if lin_csv_exists and not args.overwrite:
                    print(f"[Skip] downstream already exists: {run_dir}")
                else:
                    ensure_dir(run_dir)

                    cmd = [
                        "python", "simclr_lin.py",
                        f"method={method}",
                        f"backbone={backbone}",
                        f"batch_size={args.batch_size}",
                        f"seed={seed}",
                        f"load_epoch={up_epoch}",
                        f"finetune_epochs={args.finetune_epochs}",
                        f"+ckpt_path={ckpt.as_posix()}",
                        "attack.enabled=true",
                        "attack.fgsm=false",
                        "attack.pgd=true",
                        f"attack.eps={args.eps}",
                        f"attack.pgd_steps={args.pgd_steps}",
                        f"attack.pgd_alpha={args.pgd_alpha}",
                        "attack.pgd_random_start=true",
                        f"attack.max_test_batches={args.max_test_batches}",
                        f"hydra.run.dir={run_dir.as_posix()}",
                        "hydra.job.chdir=true",  
                        "hydra.output_subdir=.hydra",
                    ]
                    run(cmd, dry_run=args.dry_run)

                if args.dry_run:
                    metrics = {"clean_best": float("nan"), "clean_final": float("nan"),
                               "pgd_best": float("nan"), "pgd_final": float("nan")}
                    lin_csv = Path("")
                else:
                    lin_csv = downstream_history_csv(run_dir, method, backbone)
                    metrics = extract_downstream_metrics(lin_csv)

                rows.append({
                    "method": method,
                    "backbone": backbone,
                    "seed": seed,
                    "upstream_epoch": up_epoch,
                    "gamma": g,
                    **metrics,
                    "up_hist_csv": str(hist_csv),
                    "ckpt_path": str(ckpt),
                    "lin_csv": str(lin_csv),
                    "down_run_dir": str(run_dir),
                })

                print(
                    f"[Row] {method:8s} seed={seed} upE={up_epoch:3d} "
                    f"| Γ={g:.4f} | clean_best={metrics['clean_best']:.4f} | pgd_best={metrics['pgd_best']:.4f}"
                )

    df = pd.DataFrame(rows).sort_values(["method", "seed", "upstream_epoch"]).reset_index(drop=True)

    summary_dir = Path(args.summary_dir)
    ensure_dir(summary_dir)

    merged_csv = summary_dir / "gamma_vs_acc_merged.csv"
    df.to_csv(merged_csv, index=False)
    print(f"\n[Saved] {merged_csv}")

    if not args.dry_run and not df.empty:
        scatter(
            df, ycol="clean_best",
            out_png=summary_dir / "gamma_vs_clean.png",
            title=f"Γ vs Clean Accuracy ({backbone})",
            ylabel="Best Clean Test Acc (over lin epochs)",
        )
        scatter(
            df, ycol="pgd_best",
            out_png=summary_dir / "gamma_vs_pgd.png",
            title=f"Γ vs PGD Robust Accuracy ({backbone})",
            ylabel="Best PGD Acc (over lin epochs)",
        )

    print("\n[Done]")


if __name__ == "__main__":
    main()