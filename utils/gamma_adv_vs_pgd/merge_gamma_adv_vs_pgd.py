import os
import pandas as pd
import matplotlib.pyplot as plt

SEED = 1
METHODS = ["baseline", "phsim"]
MAX_EPOCH = 30  # adjust if needed

out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

all_rows = []

for method in METHODS:
    print(f"\nProcessing {method} seed{SEED}")

    # upstream clean gamma
    up_path = f"logs/upstream/{method}/seed{SEED}/{method}_seed{SEED}_train_history.csv"
    adv_path = f"logs/upstream/{method}/seed{SEED}/{method}_seed{SEED}_gamma_adv.csv"

    up = pd.read_csv(up_path)
    adv = pd.read_csv(adv_path)[["epoch","gamma_adv"]]

    for E in range(1, MAX_EPOCH + 1):

        # downstream path for this upstream epoch
        dn_path = f"logs/downstream/{method}/seed{SEED}/upE{E}/logs/downstream/{method}/seed{SEED}/lin_history_{method}_resnet18.csv"

        if not os.path.exists(dn_path):
            print(f"Missing downstream file for epoch {E}")
            continue

        dn = pd.read_csv(dn_path)

        # take FINAL linear eval row
        final = dn.iloc[-1]
        pgd_acc = final["pgd_acc"]
        test_acc = final["test_acc"]

        gamma_clean = float(up.loc[up["epoch"] == E, "gamma"])
        gamma_adv = float(adv.loc[adv["epoch"] == E, "gamma_adv"])

        all_rows.append({
            "method": method,
            "epoch": E,
            "gamma_clean": gamma_clean,
            "gamma_adv": gamma_adv,
            "pgd_acc": pgd_acc,
            "test_acc": test_acc
        })

merged = pd.DataFrame(all_rows)
merged.to_csv(f"{out_dir}/merged_seed{SEED}.csv", index=False)

print("\nMerged data:")
print(merged)

# ----- scatter gamma_adv vs pgd_acc -----
subset = merged[(merged["epoch"] <= 15) | (merged["epoch"] >= 25)]

print("\n=== Spearman correlation (epochs 1–15 & 26–30 combined) ===")
for method, sub in subset.groupby("method"):
    sub = sub.sort_values("epoch")
    r = sub[["gamma_adv", "pgd_acc"]].corr(method="spearman").iloc[0, 1]
    print(f"{method}: Spearman = {r:.4f}  (n={len(sub)})")

plt.figure()
for method, sub in subset.groupby("method"):
    plt.scatter(sub["gamma_adv"], sub["pgd_acc"], label=method)

plt.xlabel("Upstream Gamma_adv")
plt.ylabel("Downstream PGD robust acc")
plt.title(f"Seed{SEED}: Gamma_adv vs PGD acc")
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_dir}/gamma_adv_vs_pgd_subset_seed{SEED}.png", dpi=200)
plt.close()

print(f"\nSaved plot + merged CSV in {out_dir}/")
