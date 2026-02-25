# sweep_upstream_train.py
"""
Runs SimCLR upstream (pretraining) sweeps over methods × seeds.

Each run is fully self-contained under:
  runs/upstream/{method}_{backbone}_e{E}_seed{seed}/
    checkpoints/upstream/{method}/seed{seed}/epoch{K}/simclr_...pt
    logs/upstream/{method}/seed{seed}/{method}_seed{seed}_train_history.csv
    visuals/upstream/{method}/seed{seed}/epoch{K}/(loss/lr/gamma PNGs)
    .hydra/
"""

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str], dry_run: bool = False):
    print("\n[CMD]", " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def dir_nonempty(p: Path) -> bool:
    return p.exists() and any(p.iterdir())


PRESETS = {
    "dev":        (2,   500,   50),
    "paper_fast": (50,  -1,    -1),
    "paper_full": (200, -1,    -1),
}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--backbone",   default="resnet18", choices=["resnet18", "resnet34"])
    ap.add_argument("--methods",    default="baseline,phsim", help="Comma list: baseline,phsim,hybrid")
    ap.add_argument("--seeds",      default="0", help="Comma list of integer seeds")
    ap.add_argument("--preset",     default="paper_fast", choices=list(PRESETS))
    ap.add_argument("--save_every", type=int, default=5, help="Checkpoint/log interval in epochs")
    ap.add_argument("--overwrite",  action="store_true")
    ap.add_argument("--dry_run",    action="store_true")

    ap.add_argument("--epochs",      type=int, default=None)
    ap.add_argument("--subset_size", type=int, default=None)
    ap.add_argument("--max_steps",   type=int, default=None)

    args = ap.parse_args()

    methods  = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds    = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    backbone = args.backbone

    preset_epochs, preset_subset, preset_steps = PRESETS[args.preset]
    epochs      = args.epochs      if args.epochs      is not None else preset_epochs
    subset_size = args.subset_size if args.subset_size is not None else preset_subset
    max_steps   = args.max_steps   if args.max_steps   is not None else preset_steps
    log_interval = min(args.save_every, epochs)

    base_out = Path("runs/upstream")
    base_out.mkdir(parents=True, exist_ok=True)

    planned = [(seed, method) for seed in seeds for method in methods]
    print(f"[Info] Planning {len(planned)} upstream runs")

    for seed, method in planned:
        run_name = f"{method}_{backbone}_e{epochs}_seed{seed}"
        run_dir  = base_out / run_name

        if dir_nonempty(run_dir) and not args.overwrite:
            raise RuntimeError(f"Run dir exists and non-empty: {run_dir} (use --overwrite)")

        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "simclr.py",
            f"method={method}",
            f"backbone={backbone}",
            f"epochs={epochs}",
            f"log_interval={log_interval}",
            f"seed={seed}",
            f"data.subset_size={subset_size}",
            f"train.max_steps={max_steps}",
            f"hydra.run.dir={run_dir.as_posix()}",
            "hydra.output_subdir=.hydra",
        ]
        run(cmd, dry_run=args.dry_run)

    print(f"\n[Done] Upstream sweeps completed → {base_out.resolve()}")


if __name__ == "__main__":
    main()