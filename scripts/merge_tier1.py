#!/usr/bin/env python3
"""
merge_tier1.py

Assembles runs/summary/gamma_vs_pgd_merged.csv from whatever a Tier-1 run
produced, tolerant of missing files:
  * per-epoch Gamma from each upstream train_history CSV,
  * final-epoch robustness (clean / PGD-50 / AutoAttack) from the robustness JSONs,
    joined onto the matching (max) epoch row.

Rows where robustness is absent keep NaN there (analyze_results.py drops them for
the correlation). This is the cross-run, final-epoch view; for the per-epoch
Gamma<->robustness correlation you must evaluate robustness at multiple
checkpoints (see note printed at the end).

Usage:
  python scripts/merge_tier1.py --root runs/tier1_pilot --out runs/summary/gamma_vs_pgd_merged.csv
"""
import argparse, glob, json, os, re
from pathlib import Path
import pandas as pd


def load_gamma_rows(root):
    rows = []
    for csv in glob.glob(f"{root}/upstream/*/logs/**/*train_history*.csv", recursive=True):
        m = re.search(r"upstream/([a-z]+)_seed(\d+)/", csv)
        if not m:
            continue
        method, seed = m.group(1), int(m.group(2))
        df = pd.read_csv(csv)
        if "gamma" not in df or "epoch" not in df:
            continue
        for _, r in df.iterrows():
            rows.append({"method": method, "seed": seed, "epoch": int(r["epoch"]),
                         "gamma": float(r["gamma"]), "loss": float(r.get("loss", float("nan")))})
    return pd.DataFrame(rows)


def load_robustness(root):
    rows = []
    for jf in glob.glob(f"{root}/robustness/*_seed*.json"):
        name = os.path.basename(jf)
        m = re.match(r"([a-z]+)_seed(\d+)(?:_e(\d+))?\.json", name)
        if not m:
            continue
        method, seed = m.group(1), int(m.group(2))
        epoch = int(m.group(3)) if m.group(3) else None  # None => final-epoch run
        d = json.load(open(jf))
        pgd = d.get("pgd_by_steps", {})
        pgd50 = pgd.get("50", pgd.get(50, float("nan")))
        aa = d.get("autoattack", {}).get("autoattack", float("nan"))
        rows.append({"method": method, "seed": seed, "epoch": epoch,
                     "clean_acc": d.get("clean_acc", float("nan")),
                     "pgd_acc": float(pgd50), "autoattack": float(aa),
                     "robust_headline": d.get("masking", {}).get("robust_acc_headline", float("nan"))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/tier1_pilot")
    ap.add_argument("--out", default="runs/summary/gamma_vs_pgd_merged.csv")
    args = ap.parse_args()

    g = load_gamma_rows(args.root)
    r = load_robustness(args.root)
    if g.empty:
        raise SystemExit(f"No train_history CSVs under {args.root}/upstream/*/logs/")

    # final-epoch robustness rows (epoch is None) -> attach to each run's max epoch
    finals = r[r["epoch"].isna()].copy()
    if not finals.empty:
        maxep = g.groupby(["method", "seed"])["epoch"].max().rename("epoch").reset_index()
        finals = finals.drop(columns=["epoch"]).merge(maxep, on=["method", "seed"], how="left")
    explicit = r[r["epoch"].notna()].copy()
    rob = pd.concat([finals, explicit], ignore_index=True)

    merged = g.merge(rob, on=["method", "seed", "epoch"], how="left")
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    n_rob = merged["pgd_acc"].notna().sum()
    print(f"[merge] wrote {args.out}  ({len(merged)} rows, {n_rob} with robustness)")
    print("[merge] columns:", list(merged.columns))
    if n_rob < 3:
        print("[merge] NOTE: <3 rows have robustness -> correlation not meaningful.")
        print("        Evaluate robustness at multiple checkpoints (epochs) for a real #4 correlation.")
    print("\nNext:")
    print(f"  python analyze_results.py --csv {args.out} \\")
    print( "     --gamma_col gamma --pgd_col pgd_acc --clean_col clean_acc --epoch_col epoch \\")
    print( "     --out runs/tier1_pilot/stats/full_report.json")


if __name__ == "__main__":
    main()
