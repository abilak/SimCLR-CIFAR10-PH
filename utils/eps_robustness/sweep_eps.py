#!/usr/bin/env python3
"""
sweep_eps.py

Launches simclr_lin.py across methods/seeds/upstream epochs,
writing outputs into:
  logs/downstream/{method}/seed{seed}/upE{load_epoch}/

Example:
  python sweep_eps.py --backbone resnet18 --methods baseline phsim --seeds 0 1 --up_epochs 10 20 30
"""

import argparse
import subprocess
import os


def upstream_ckpt_exists(method, seed, upE, backbone):
    """
    Checks whether upstream checkpoint exists.
    """
    ckpt_path = os.path.join(
        "checkpoints", "upstream", method,
        f"seed{seed}",
        f"epoch{upE}",
        f"simclr_{method}_{backbone}_epoch{upE}_seed{seed}.pt"
    )
    return os.path.exists(ckpt_path)


def run(cmd):
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--methods", nargs="+", default=["baseline", "phsim"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--up_epochs", nargs="+", type=int, default=[10, 20, 30])
    ap.add_argument("--eps_px", default="0,2,4,6,8,10",
                    help="comma-separated integers, interpreted as /255")
    ap.add_argument("--pgd_steps", type=int, default=10)
    ap.add_argument("--fast", action="store_true",
                    help="debug mode: fewer test batches")
    args = ap.parse_args()

    eps_list_str = "[" + ",".join(
        [s.strip() for s in args.eps_px.split(",") if s.strip() != ""]
    ) + "]"

    for method in args.methods:
        for seed in args.seeds:
            for upE in args.up_epochs:

                if not upstream_ckpt_exists(method, seed, upE, args.backbone):
                    print(f"[SKIP] Missing ckpt: method={method} seed={seed} upE={upE}")
                    continue

                print(f"[RUN] method={method} seed={seed} upE={upE}")

                run_dir = f"logs/downstream/{method}/seed{seed}/upE{upE}"

                cmd = [
                    "python", "simclr_lin.py",
                    f"backbone={args.backbone}",
                    f"method={method}",
                    f"seed={seed}",
                    f"load_epoch={upE}",
                    f"hydra.run.dir={run_dir}",
                    "attack.enabled=true",
                    "attack.sweep=true",
                    f"attack.eps_px={eps_list_str}",
                    f"attack.pgd_steps={args.pgd_steps}",
                    "attack.pgd_alpha=-1.0",
                    "attack.pgd_random_start=true",
                ]

                if args.fast:
                    cmd.append("attack.max_test_batches=20")

                run(cmd)


if __name__ == "__main__":
    main()