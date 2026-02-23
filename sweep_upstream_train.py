"""
sweep_upstream_train.py

Runs SimCLR upstream (pretraining) sweeps over methods × seeds with clean output structure.

File structure produced:
  runs/upstream/{method}_{backbone}_e{E}_seed{seed}/
    checkpoints/upstream/{method}/   <- .pt files from simclr.py
    visuals/upstream/{method}/       <- loss/lr/gamma pngs + CSV
    .hydra/                          <- hydra config snapshot

Usage examples:
  # Quick dev smoke-test (2 epochs, 500 samples, 100 steps/epoch)
  python sweep_upstream_train.py --preset dev --methods baseline,phsim --seeds 0

  # Medium run
  python sweep_upstream_train.py --preset paper_fast --methods baseline,phsim --seeds 0,1

  # Full paper run
  python sweep_upstream_train.py --preset paper_full --methods baseline,phsim,hybrid --seeds 0,1,2 --save_every 10
"""

import argparse
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], dry_run: bool = False):
    print("\n[CMD]", " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def dir_nonempty(p: Path) -> bool:
    return p.exists() and any(p.iterdir())


PRESETS = {
    # (epochs, subset_size, max_steps_per_epoch)
    "dev":        (2,   500,   50),
    "paper_fast": (50,  -1,    -1),
    "paper_full": (200, -1,    -1),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--backbone",   default="resnet18",      choices=["resnet18", "resnet34"])
    ap.add_argument("--methods",    default="baseline,phsim", help="Comma list: baseline,phsim,hybrid")
    ap.add_argument("--seeds",      default="0",              help="Comma list of integer seeds")
    ap.add_argument("--preset",     default="paper_fast",     choices=list(PRESETS),
                    help="Epoch/data preset. dev=smoke-test, paper_fast=50ep, paper_full=200ep")

    ap.add_argument("--save_every", type=int, default=5,
                    help="Checkpoint + CSV log interval (epochs).")
    ap.add_argument("--overwrite",  action="store_true",
                    help="Allow re-running into an existing non-empty run dir.")
    ap.add_argument("--dry_run",    action="store_true",
                    help="Print commands without executing them.")

    # optional overrides (override preset values)
    ap.add_argument("--epochs",      type=int, default=None, help="Override preset epoch count.")
    ap.add_argument("--subset_size", type=int, default=None, help="Override preset subset_size.")
    ap.add_argument("--max_steps",   type=int, default=None, help="Override preset max_steps.")

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
    print(f"[Info] Planning {len(planned)} runs: methods={methods}, seeds={seeds}, "
          f"backbone={backbone}, preset={args.preset} "
          f"(epochs={epochs}, subset={subset_size}, max_steps={max_steps})")

    for seed, method in planned:
        run_name = f"{method}_{backbone}_e{epochs}_seed{seed}"
        run_dir  = base_out / run_name

        if dir_nonempty(run_dir) and not args.overwrite:
            raise RuntimeError(
                f"\nRun dir already exists and is non-empty:\n  {run_dir}\n"
                f"Use --overwrite to allow reuse, or change --preset / --seeds / --methods."
            )

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

    print(f"\n[Done] Upstream sweeps completed. Outputs under: {base_out.resolve()}")


if __name__ == "__main__":
    main()