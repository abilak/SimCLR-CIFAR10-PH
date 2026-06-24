#!/usr/bin/env bash
# =============================================================================
# run_tier1.sh  --  Tier-1 experiment driver (run on the CUDA box / Colab)
#
# Runs the four reviewer-critical experiments end to end:
#   #1 Robustness + gradient-masking diagnostics (AutoAttack, PGD curves, transfer)
#   #2 Topology-specificity controls (swcontrol method + matched-clean analysis)
#   #3 Topology-under-attack mechanism test (collapse vs preservation)
#   #4 Statistical rehabilitation of the Gamma<->robustness correlation
#
# Prereqs:
#   pip install -r requirements.txt
#   pip install git+https://github.com/fra31/auto-attack    # AutoAttack
#
# Edit the knobs below, then:  bash scripts/run_tier1.sh
# =============================================================================
set -euo pipefail

BACKBONE=resnet18
SEEDS=(0 1 2)                 # Tier-1 #1 requires >= 5 seeds
METHODS=(baseline phsim swcontrol)
EPOCHS=50                        # real pretraining (not the epoch-10 toy checkpoint)
SAVE_EVERY=10                     # checkpoint cadence (gives the epoch sweep for #4)
EPS_PX=8                          # 8/255 Linf
OUT=runs/tier1_pilot
PY=python

mkdir -p "$OUT"/{upstream,robustness,mechanism,stats}

ckpt_path () {  # method seed epoch -> checkpoint path
  echo "$OUT/upstream/${1}_seed${2}/checkpoints/upstream/${1}/seed${2}/epoch${3}/simclr_${1}_${BACKBONE}_epoch${3}_seed${2}.pt"
}

# ----------------------------------------------------------------------------
# Train: full CIFAR-10, all methods x seeds. Differentiable Gamma for phsim;
# matched non-topological control for swcontrol.
# ----------------------------------------------------------------------------
for seed in "${SEEDS[@]}"; do
  for m in "${METHODS[@]}"; do
    rd="$OUT/upstream/${m}_seed${seed}"
    echo "==== TRAIN $m seed=$seed ===="
    $PY simclr.py method="$m" backbone="$BACKBONE" seed="$seed" \
        epochs="$EPOCHS" log_interval="$SAVE_EVERY" \
        data.subset_size=-1 train.max_steps=-1 \
        hydra.run.dir="$rd" hydra.output_subdir=.hydra hydra.job.chdir=true
  done
done

# ----------------------------------------------------------------------------
# #1 Robustness + masking, at the final epoch. Baseline (same seed) is the
#    surrogate for black-box transfer.
# ----------------------------------------------------------------------------
for seed in "${SEEDS[@]}"; do
  surr=$(ckpt_path baseline "$seed" "$EPOCHS")
  for m in phsim swcontrol baseline; do
    ck=$(ckpt_path "$m" "$seed" "$EPOCHS")
    echo "==== ROBUSTNESS $m seed=$seed ===="
    $PY eval_robustness.py --ckpt "$ck" --surrogate_ckpt "$surr" \
        --eps_px "$EPS_PX" --out "$OUT/robustness/${m}_seed${seed}.json" \
        --max_test_batches 8        # raise/remove for full test set
  done
done

# ----------------------------------------------------------------------------
# #3 Mechanism: class-conditioned topology, clean vs attacked, per method,
#    for EVERY seed (so the Γ-stability ordering has n=#seeds, not n=1).
# ----------------------------------------------------------------------------
for seed in "${SEEDS[@]}"; do
  echo "==== MECHANISM seed=$seed ===="
  $PY eval_mechanism.py \
      --ckpt baseline=$(ckpt_path baseline "$seed" "$EPOCHS") \
      --ckpt phsim=$(ckpt_path phsim "$seed" "$EPOCHS") \
      --ckpt swcontrol=$(ckpt_path swcontrol "$seed" "$EPOCHS") \
      --out "$OUT/mechanism/seed${seed}" --eps_px "$EPS_PX" --per_class 80
done

# ----------------------------------------------------------------------------
# #2 + #4 Stats. First merge per-checkpoint Gamma + robustness into one CSV
#    (sweep_full_pipeline.py builds runs/summary/gamma_vs_pgd_merged.csv), then:
# ----------------------------------------------------------------------------
# $PY scripts/sweep_full_pipeline.py --methods baseline,phsim,swcontrol --seeds 0,1,2,3,4
$PY analyze_results.py --csv runs/summary/gamma_vs_pgd_merged.csv \
    --gamma_col gamma --pgd_col pgd_acc_best --clean_col best_test_acc \
    --epoch_col upstream_epoch --out "$OUT/stats/full_report.json" || \
    echo "[note] build the merged CSV first (see sweep_full_pipeline.py), then re-run analyze_results.py"

echo "[run_tier1] done -> $OUT"
