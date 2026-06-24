#!/usr/bin/env bash
# =============================================================================
# run_phacl.sh -- Adversarial PH-ACL experiment matrix (run on the CUDA box)
#
# Trains the method matrix that tests whether topology earns its place in the
# ADVERSARIAL regime (the one the paper's theory is actually about), then runs
# the masking-aware robustness suite, the mechanism test, and the stats.
#
# Methods:
#   baseline       clean SimCLR (reference; high clean, ~0 robust)
#   adv_baseline   adversarial SimCLR = ACL (the proper robust-SSL reference)
#   adv_phsim      adversarial topological separation
#   adv_swcontrol  adversarial raw-SW separation (non-topo control, adv regime)
#   topoacl        FLAGSHIP: NT-Xent + beta * topological adversarial consistency
#   rawacl         topoacl's non-topological control
#
# The decisive comparisons:
#   adv_baseline vs baseline   -> does adversarial training give real robustness?
#   adv_phsim vs adv_swcontrol -> is separation topology-specific under attack?
#   topoacl vs rawacl          -> is consistency topology-specific under attack?
#   topoacl vs adv_baseline    -> does topological consistency beat plain ACL?
#   (clean acc of topoacl)     -> does NT-Xent keep clean high (above tradeoff)?
#
# Cost note: adversarial methods are ~ (adv.steps) x slower per step than clean.
# Start with the pilot knobs; scale SEEDS/EPOCHS once it looks promising.
# =============================================================================
set -euo pipefail

BACKBONE=resnet18
SEEDS=(0)               # 1 seed for a same-day directional read; expand to (0 1 2) after
METHODS=(baseline adv_baseline topoacl rawacl)   # the 4 that answer the 2 decisive questions
EPOCHS=50
SAVE_EVERY=10
WARMUP=5                  # clean-baseline warmup epochs (stabilizes adv methods)
EPS_PX=8
ADV_STEPS=3               # valid inner attack, ~1.5x faster than 5
SOURCE_LAYER=layer3       # 16-pt PH cloud; set layer2 (64 pts) for richer/H1
EXTRA_LAYERS="[]"         # multiscale, e.g. "[layer2]" to add a second depth
NEG_AGG=hard              # or 'soft' (smooth-min over all negatives)
DUAL_BN=true              # the whole point: AdvProp/AdvCL dual-BN (clean/adv split BN) for adv methods
OUT=runs/phacl_dualbn     # fresh dir, doesn't touch your old runs
PY=python

mkdir -p "$OUT"/{upstream,robustness,mechanism,stats}

ckpt_path () {  # method seed epoch -> path
  echo "$OUT/upstream/${1}_seed${2}/checkpoints/upstream/${1}/seed${2}/epoch${3}/simclr_${1}_${BACKBONE}_epoch${3}_seed${2}.pt"
}

# ---- Train (skip if final checkpoint exists; runs auto-resume otherwise) ----
for seed in "${SEEDS[@]}"; do
  for m in "${METHODS[@]}"; do
    if [[ -f "$(ckpt_path "$m" "$seed" "$EPOCHS")" ]]; then
      echo "==== SKIP train $m seed=$seed (final checkpoint exists) ===="; continue
    fi
    rd="$OUT/upstream/${m}_seed${seed}"
    echo "==== TRAIN $m seed=$seed ===="
    $PY simclr.py method="$m" backbone="$BACKBONE" seed="$seed" \
        epochs="$EPOCHS" log_interval="$SAVE_EVERY" train.warmup_epochs="$WARMUP" \
        data.subset_size=-1 train.max_steps=-1 \
        ph.source_layer="$SOURCE_LAYER" "ph.extra_layers=$EXTRA_LAYERS" ph.neg_agg="$NEG_AGG" \
        adv.steps="$ADV_STEPS" adv.eps=$(python -c "print($EPS_PX/255)") adv.dual_bn="$DUAL_BN" \
        hydra.run.dir="$rd" hydra.output_subdir=.hydra hydra.job.chdir=true
  done
done

# ---- Robustness (#1): masking-aware suite, baseline surrogate for transfer --
for seed in "${SEEDS[@]}"; do
  surr=$(ckpt_path baseline "$seed" "$EPOCHS")
  for m in "${METHODS[@]}"; do
    out_json="$OUT/robustness/${m}_seed${seed}.json"
    if [[ -f "$out_json" ]]; then echo "==== SKIP robustness $m seed=$seed (json exists) ===="; continue; fi
    ck=$(ckpt_path "$m" "$seed" "$EPOCHS")
    [[ -f "$ck" ]] || { echo "  (no checkpoint for $m seed=$seed yet)"; continue; }
    echo "==== ROBUSTNESS $m seed=$seed ===="
    surr_arg=(); [[ -f "$surr" && "$m" != "baseline" ]] && surr_arg=(--surrogate_ckpt "$surr")
    $PY eval_robustness.py --ckpt "$ck" "${surr_arg[@]}" \
        --eps_px "$EPS_PX" --out "$out_json" \
        --max_test_batches 8      # remove for full test set (final numbers)
  done
done

# ---- Mechanism (#3): topology under attack, all seeds ----------------------
for seed in "${SEEDS[@]}"; do
  [[ -f "$OUT/mechanism/seed${seed}/mechanism.json" ]] && { echo "==== SKIP mechanism seed=$seed ===="; continue; }
  args=()
  for m in "${METHODS[@]}"; do
    ck=$(ckpt_path "$m" "$seed" "$EPOCHS"); [[ -f "$ck" ]] && args+=(--ckpt "$m=$ck")
  done
  [[ ${#args[@]} -eq 0 ]] && { echo "  (no checkpoints for seed=$seed yet)"; continue; }
  $PY eval_mechanism.py "${args[@]}" --out "$OUT/mechanism/seed${seed}" --eps_px "$EPS_PX" --per_class 80
done

echo "[run_phacl] done -> $OUT  (then: python scripts/merge_tier1.py --root $OUT ... ; python analyze_results.py ...)"
