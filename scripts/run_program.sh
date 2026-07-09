#!/usr/bin/env bash
# =============================================================================
# run_program.sh -- unified, RESUMABLE launcher for the PH-ACL program.
#
# Parameterized over dataset x backbone x methods x seeds, with skip-completed
# guards at every stage (train / robustness / mechanism). Safe to Ctrl-C, crash,
# get preempted (spot), or re-run: finished work is skipped, and individual
# training runs resume from their latest checkpoint (see `resume` in simclr.py).
# This is what makes the cheap-spot and parallel-across-rented-GPUs routes work.
#
# Run the SAME command again after any interruption -- it picks up where it left
# off. To parallelize across N GPUs, launch N copies with disjoint SEEDS (or
# disjoint METHODS) and the same OUT; the skip guards prevent double-work.
#
# Presets (edit the knobs below or override via env):
#   DATASET=cifar10  BACKBONE=resnet18  -> the CIFAR-10 home cell
#   DATASET=cifar100 / stl10            -> reduced generalization datasets
#   BACKBONE=resnet50                   -> architecture check
# =============================================================================
set -euo pipefail

# ---- knobs (override via env, e.g. `DATASET=cifar100 bash scripts/run_program.sh`) ----
DATASET="${DATASET:-cifar10}"           # cifar10 | cifar100 | stl10
BACKBONE="${BACKBONE:-resnet18}"        # resnet18 | resnet34 | resnet50
METHODS="${METHODS:-baseline adv_baseline topoacl rawacl}"   # space-separated
SEEDS="${SEEDS:-0 1 2}"                 # space-separated
EPOCHS="${EPOCHS:-200}"
SAVE_EVERY="${SAVE_EVERY:-20}"
WARMUP="${WARMUP:-10}"                  # clean-baseline warmup (stabilizes adv methods)
EPS_PX="${EPS_PX:-8}"
ADV_STEPS="${ADV_STEPS:-5}"             # inner-PGD steps (3 to iterate faster, 5 for headline)
SOURCE_LAYER="${SOURCE_LAYER:-layer3}"
EXTRA_LAYERS="${EXTRA_LAYERS:-[]}"      # multiscale, e.g. "[layer2]"
BETA="${BETA:-}"                        # consistency weight (adv.beta); empty=config default (1.0)
NEG_AGG="${NEG_AGG:-hard}"
DUAL_BN="${DUAL_BN:-true}"
MAX_TEST_BATCHES="${MAX_TEST_BATCHES:--1}"   # -1 = full test set (final); e.g. 8 to iterate
SUPERVISED="${SUPERVISED:-}"            # supervised AT reference baselines, e.g. "pgd_at trades" (item 8)
ROBUST_PROBE="${ROBUST_PROBE:-true}"    # robust linear eval (head trained on PGD) -- reveals robustness; set false for clean probe
BN_BRANCH="${BN_BRANCH:-adv}"           # dual-BN eval branch: 'adv' is where robustness lives (no-op for single-BN baselines)
PROBE_EPOCHS="${PROBE_EPOCHS:-}"        # robust-probe training epochs (empty=eval default 20; set small e.g. 1 for fast preflight)
PROBE_PER_CLASS="${PROBE_PER_CLASS:-}"  # labeled probe images/class (empty=default 500; set small e.g. 20 for fast preflight)
OUT="${OUT:-runs/${DATASET}_${BACKBONE}}"    # per-(dataset,backbone) dir so runs don't collide
PY="${PY:-python}"

read -r -a METHODS_ARR <<< "$METHODS"
read -r -a SEEDS_ARR <<< "$SEEDS"

mkdir -p "$OUT"/{upstream,robustness,mechanism,stats}
echo "[run_program] dataset=$DATASET backbone=$BACKBONE methods=(${METHODS}) seeds=(${SEEDS})"
echo "[run_program] epochs=$EPOCHS steps=$ADV_STEPS dual_bn=$DUAL_BN out=$OUT"

ckpt_path () {  # method seed epoch -> path
  echo "$OUT/upstream/${1}_seed${2}/checkpoints/upstream/${1}/seed${2}/epoch${3}/simclr_${1}_${BACKBONE}_epoch${3}_seed${2}.pt"
}

# ---- Train (skip if the final-epoch checkpoint already exists) --------------
for seed in "${SEEDS_ARR[@]}"; do
  for m in "${METHODS_ARR[@]}"; do
    final_ckpt=$(ckpt_path "$m" "$seed" "$EPOCHS")
    if [[ -f "$final_ckpt" ]]; then
      echo "==== SKIP train $m seed=$seed (final checkpoint exists) ===="
      continue
    fi
    rd="$OUT/upstream/${m}_seed${seed}"
    echo "==== TRAIN $m seed=$seed (dataset=$DATASET backbone=$BACKBONE) ===="
    beta_override=(); [[ -n "$BETA" ]] && beta_override=(adv.beta="$BETA")
    $PY simclr.py method="$m" backbone="$BACKBONE" seed="$seed" dataset="$DATASET" \
        epochs="$EPOCHS" log_interval="$SAVE_EVERY" train.warmup_epochs="$WARMUP" \
        data.subset_size=-1 train.max_steps=-1 \
        ph.source_layer="$SOURCE_LAYER" "ph.extra_layers=$EXTRA_LAYERS" ph.neg_agg="$NEG_AGG" \
        adv.steps="$ADV_STEPS" adv.eps=$($PY -c "print($EPS_PX/255)") adv.dual_bn="$DUAL_BN" \
        "${beta_override[@]}" \
        hydra.run.dir="$rd" hydra.output_subdir=.hydra hydra.job.chdir=true
  done
done

# ---- Robustness (skip if the result JSON already exists) --------------------
for seed in "${SEEDS_ARR[@]}"; do
  surr=$(ckpt_path baseline "$seed" "$EPOCHS")
  for m in "${METHODS_ARR[@]}"; do
    out_json="$OUT/robustness/${m}_seed${seed}.json"
    if [[ -f "$out_json" ]]; then
      echo "==== SKIP robustness $m seed=$seed (json exists) ===="
      continue
    fi
    ck=$(ckpt_path "$m" "$seed" "$EPOCHS")
    if [[ ! -f "$ck" ]]; then echo "  (no checkpoint for $m seed=$seed yet, skipping eval)"; continue; fi
    echo "==== ROBUSTNESS $m seed=$seed ===="
    surr_arg=()
    [[ -f "$surr" && "$m" != "baseline" ]] && surr_arg=(--surrogate_ckpt "$surr")
    # Robust linear eval (head trained on PGD) -- without it, robustness reads ~0%
    # even on a robust model. The BN branch is chosen PER METHOD just below.
    rp_arg=(); [[ "$ROBUST_PROBE" == "true" ]] && rp_arg=(--robust_probe)
    pe_arg=(); [[ -n "$PROBE_EPOCHS" ]] && pe_arg=(--probe_epochs "$PROBE_EPOCHS")
    pc_arg=(); [[ -n "$PROBE_PER_CLASS" ]] && pc_arg=(--probe_per_class "$PROBE_PER_CLASS")
    # Per-method eval branch: AdvProp methods (adv_baseline/adv_phsim/adv_swcontrol)
    # put robustness in adv-BN (trained in train mode). The CONSISTENCY methods
    # (topoacl/rawacl) run their consistency in eval mode, so their adv-BN running
    # stats never train -- their deployable representation is clean-BN. Evaluating
    # them on adv-BN reads garbage (~chance). So force clean-BN for those.
    case "$m" in
      topoacl|rawacl) mbn=clean ;;
      *)              mbn="$BN_BRANCH" ;;
    esac
    $PY eval_robustness.py --ckpt "$ck" "${surr_arg[@]}" --dataset "$DATASET" \
        --eps_px "$EPS_PX" --bn_branch "$mbn" "${rp_arg[@]}" "${pe_arg[@]}" "${pc_arg[@]}" \
        --out "$out_json" --max_test_batches "$MAX_TEST_BATCHES"
  done
done

# ---- Mechanism (skip if the result exists) ----------------------------------
for seed in "${SEEDS_ARR[@]}"; do
  mech_out="$OUT/mechanism/seed${seed}"
  if [[ -f "$mech_out/mechanism.json" ]]; then
    echo "==== SKIP mechanism seed=$seed (exists) ===="
    continue
  fi
  args=()
  for m in "${METHODS_ARR[@]}"; do
    ck=$(ckpt_path "$m" "$seed" "$EPOCHS")
    [[ -f "$ck" ]] && args+=(--ckpt "$m=$ck")
  done
  [[ ${#args[@]} -eq 0 ]] && { echo "  (no checkpoints for seed=$seed yet)"; continue; }
  $PY eval_mechanism.py "${args[@]}" --dataset "$DATASET" \
      --out "$mech_out" --eps_px "$EPS_PX" --per_class 80
done

# ---- Supervised reference baselines (PGD-AT / TRADES) -> SAME robustness table --
# These are end-to-end supervised classifiers (item 8 "reference upper bounds").
# They write $OUT/robustness/<method>_seed<seed>.json in the same format as the
# SSL evals, so the whole table merges together. RoCL / exact-AdvCL ports are
# separate (need their official repos) -- see PROGRAM.md.
read -r -a SUP_ARR <<< "$SUPERVISED"
for seed in "${SEEDS_ARR[@]}"; do
  for sm in "${SUP_ARR[@]}"; do
    out_json="$OUT/robustness/${sm}_seed${seed}.json"
    if [[ -f "$out_json" ]]; then echo "==== SKIP $sm seed=$seed (json exists) ===="; continue; fi
    echo "==== SUPERVISED $sm seed=$seed ===="
    $PY adv_supervised.py --method "$sm" --dataset "$DATASET" --backbone "$BACKBONE" \
        --epochs "$EPOCHS" --eps_px "$EPS_PX" --steps "$ADV_STEPS" --seed "$seed" \
        --ckpt_dir "$OUT/supervised/${sm}_seed${seed}" --out "$out_json" \
        --max_test_batches "$MAX_TEST_BATCHES"
  done
done

echo "[run_program] done -> $OUT"
echo "  stats: $PY scripts/merge_tier1.py --root $OUT ... ; $PY analyze_results.py ..."
