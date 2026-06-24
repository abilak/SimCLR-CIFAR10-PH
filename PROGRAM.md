# Experiment program — run order, configs, and what's ready

This is the runbook for taking PH-ACL from the pilot to a defensible paper. It
lists exactly what to run, in what order, what's gated on what, and what can run
concurrently if you get a second GPU.

## What is built + tested (trust these for real compute)
- **Datasets**: `dataset=cifar10|cifar100|stl10` (STL-10 = 96×96, SSL on the
  unlabeled split). Registry in `datasets.py`; wired through training, linear
  probe, robustness, mechanism. Default `cifar10` = original pipeline.
- **Architectures**: `backbone=resnet18|resnet34|resnet50` (PH source channels
  derived from the backbone, so RN-50 Bottleneck works).
- **Dual-BN** (`adv.dual_bn=true`) + **PH-only inner forward** speedup.
- **Resumable launcher** `scripts/run_program.sh` + **within-run resume**
  (`train.resume=true`): crash / spot-preemption safe. Re-run the same command
  and it skips finished runs and continues interrupted ones from the last epoch.
- **Supervised AT baselines** `adv_supervised.py`: **PGD-AT + TRADES** (item 8
  "reference upper bounds"), architecture-matched to the SSL encoder, evaluated by
  the SAME robustness suite → land in the SAME table. Run via `SUPERVISED="pgd_at trades"`.
- **AdvCL** is covered: `adv_baseline` + `adv.dual_bn=true` IS the AdvCL recipe
  (adversarial SimCLR + dual-BN) — label that row "AdvCL (ours, dual-BN)".

## Not yet built (see "Build next" — do NOT assume these run yet)
- **RoCL** (Kim et al. [20]) exact port + (optionally) the *official* AdvCL — need
  their GitHub repos cloned on the box (item 8 head-to-head).
- SSL frameworks MoCo / BYOL / SimSiam (item 6) — need `lightly`.
- ViT-S backbone (item 7) — needs `timm` + PH-from-tokens extraction.

## Prerequisites on the box
```bash
git pull                                  # get the dual-BN / datasets / resume code
python -m pip install -r requirements.txt # gudhi + autoattack; timm/lightly are commented out
python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

## Run order (single GPU)

### Phase 0 — pilot (RUNNING NOW). Gate.
`scripts/run_phacl.sh` with the focused config (4 methods, 1 seed, 50 ep, steps=3,
dual-BN). Decision gate before spending the big compute:
- `adv_baseline` robust acc jumps off ~3% → adversarial training + dual-BN works.
- `topoacl` ≥ `rawacl` (robustness & topology stability) and `topoacl` clean acc
  near baseline → topology earns its place.
- If both hold → proceed to Phase 1. If not → you have a clean negative (theory +
  mechanism + controls), still a paper.

### Phase 1 — CIFAR-10 home matrix (~7 days). The main result.
```bash
DATASET=cifar10 BACKBONE=resnet18 \
  METHODS="baseline adv_baseline adv_phsim adv_swcontrol topoacl rawacl" \
  SUPERVISED="pgd_at trades" \
  SEEDS="0 1 2" EPOCHS=200 ADV_STEPS=5 DUAL_BN=true MAX_TEST_BATCHES=-1 \
  bash scripts/run_program.sh
```
This produces ONE robustness table with: baseline, adv_baseline (= **AdvCL**,
dual-BN), adv_phsim, adv_swcontrol, topoacl, rawacl, **PGD-AT**, **TRADES**.
(3 seeds, not 5 — see the deadline math; 3 seeds buys a second dataset, which a
reviewer values more than 2 extra seeds.) Add the **RoCL** row once that port is
built (next).

### Phase 2 — generalization datasets, reduced (~5 days each).
```bash
DATASET=cifar100 BACKBONE=resnet18 \
  METHODS="baseline adv_baseline topoacl rawacl" \
  SEEDS="0 1 2" EPOCHS=200 ADV_STEPS=5 DUAL_BN=true MAX_TEST_BATCHES=-1 \
  bash scripts/run_program.sh

DATASET=stl10 BACKBONE=resnet18 \
  METHODS="baseline adv_baseline topoacl rawacl" \
  SEEDS="0 1 2" EPOCHS=200 ADV_STEPS=5 DUAL_BN=true MAX_TEST_BATCHES=-1 \
  bash scripts/run_program.sh
```

### Phase 3 — architecture check, reduced (~5–6 days).
```bash
DATASET=cifar10 BACKBONE=resnet50 \
  METHODS="baseline adv_baseline topoacl rawacl" \
  SEEDS="0 1 2" EPOCHS=200 ADV_STEPS=5 DUAL_BN=true MAX_TEST_BATCHES=-1 \
  bash scripts/run_program.sh
```

### Stats (cheap, after each phase)
```bash
python scripts/merge_tier1.py --root runs/cifar10_resnet18 --out runs/cifar10_resnet18/stats/merged.csv
python analyze_results.py --csv runs/cifar10_resnet18/stats/merged.csv \
    --gamma_col gamma --pgd_col pgd_acc --clean_col clean_acc --epoch_col epoch \
    --out runs/cifar10_resnet18/stats/report.json
```

## Concurrency (if you get a second GPU / can run short jobs alongside)
`run_program.sh` is **safe to run in parallel** with the same `OUT`: the
skip-completed guards prevent double-work. Two patterns:
- **Split seeds across GPUs**: `CUDA_VISIBLE_DEVICES=0 SEEDS=0 … &` and
  `CUDA_VISIBLE_DEVICES=1 SEEDS="1 2" …` — same OUT, disjoint seeds.
- **Offload short (≤1 day) jobs**: the pilot, any single reduced-dataset
  `baseline` seed (~3 h), and all the **evals** (robustness ~1–2 h/ckpt,
  mechanism, stats) are short. Once checkpoints exist, a second GPU can run the
  eval stages (re-running `run_program.sh` there skips training and does only the
  missing evals).

## Build next (offer — each will be added + tested before you run it)
Ordered by value:
1. ✅ **PGD-AT + TRADES** — DONE (`adv_supervised.py`, tested). Run via `SUPERVISED=`.
2. **RoCL** (Kim et al. [20]) exact port + optionally the *official* **AdvCL** for the
   head-to-head robust-SSL table (item 8). Needs their repos cloned on the box.
   This is the remaining credibility-critical piece (the cited-but-uncompared [20]).
3. **ViT-S** via `timm` + PH-from-tokens extraction (item 7). Needs `timm`.
4. **MoCo / BYOL / SimSiam** via `lightly` (item 6). BYOL/SimSiam have no
   negatives, so only the *consistency* variant (`topoacl`) transfers — a genuinely
   interesting boundary test. Heaviest; needs `lightly`.

Item 2 (RoCL) is the last credibility-critical gap. 3–4 strengthen but can be
scoped as "future work" if the month runs short.
