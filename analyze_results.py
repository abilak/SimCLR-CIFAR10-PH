#!/usr/bin/env python3
"""
analyze_results.py  (Tier-1 #4 + #2 stats)

Honest correlation analysis (Spearman + p-value + bootstrap CI + epoch-controlled
partial correlation) and the matched-clean-accuracy control, on a merged results
CSV. Runs anywhere (no GPU).

Examples
--------
  # The committed single-seed gamma_adv vs pgd file
  python analyze_results.py --csv output/gamma_adv_vs_pgd/merged_seed1.csv \
      --gamma_col gamma_adv --pgd_col pgd_acc --epoch_col epoch --out runs/stats/seed1.json

  # A full sweep (pool all seeds/epochs); add the matched-clean control
  python analyze_results.py --csv runs/summary/merged_all.csv \
      --gamma_col gamma --pgd_col pgd_at_best_clean --clean_col clean_at_best_clean \
      --epoch_col up_epoch --out runs/stats/full.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from phtopo.stats import correlation_report, matched_clean_control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--gamma_col", default="gamma")
    ap.add_argument("--pgd_col", default="pgd_acc")
    ap.add_argument("--clean_col", default=None, help="enables matched-clean control")
    ap.add_argument("--epoch_col", default="epoch")
    ap.add_argument("--method_col", default="method")
    ap.add_argument("--match_tol", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    report = {"csv": args.csv, "n_rows": len(df)}
    report["correlation"] = correlation_report(
        df, args.gamma_col, args.pgd_col, args.epoch_col, args.method_col
    )
    if args.clean_col and args.clean_col in df.columns:
        report["matched_clean_control"] = matched_clean_control(
            df, args.clean_col, args.pgd_col, args.method_col, tol=args.match_tol
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=float)

    # human-readable summary
    print(f"\n=== Gamma({args.gamma_col}) vs {args.pgd_col} ===")
    def show(name, block):
        sp = block["spearman"]
        line = f"  {name:10s} n={sp['n']:3d}  rho={sp['rho']:+.3f}  p={sp['p']:.3f}  CI=[{sp['ci_low']:+.3f},{sp['ci_high']:+.3f}]"
        if "partial_spearman_epoch" in block:
            ps = block["partial_spearman_epoch"]
            line += f"  | partial(epoch) rho={ps['rho_partial']:+.3f} p={ps['p']:.3f}"
        print(line)
    show("overall", report["correlation"]["overall"])
    for m, b in report["correlation"].get("by_method", {}).items():
        show(m, b)
    if "matched_clean_control" in report:
        mc = report["matched_clean_control"]
        print("\n=== Matched-clean-accuracy control ===")
        print("  ", mc.get("interpretation", mc.get("note")))
    print(f"\n[analyze] wrote {args.out}")


if __name__ == "__main__":
    main()
