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

## Evaluation protocol — CRITICAL (read before reporting any number)
Dual-BN / AdvProp models must be evaluated through the **adv-BN branch** with a
**robust linear probe** (head trained on PGD), or robustness reads ~0% even on a
genuinely robust model. Proven on `adv_baseline` (300 ep, cifar10, seed0):

| probe × branch | clean | PGD-100 @ ε=8 |
|---|---|---|
| clean / clean-BN (naive default) | 0.757 | **0.031** ← the artifact |
| clean / adv-BN | 0.586 | 0.224 |
| robust / adv-BN | 0.567 | **0.287** ← real robustness |

So always evaluate with `--robust_probe --bn_branch adv` (the launcher now defaults
`ROBUST_PROBE=true BN_BRANCH=adv`). Report the **adv-BN operating point**
(clean≈57% / robust≈29% here) as the robust model — AdvProp/AdvCL convention; the
clean-BN branch is the high-clean / low-robust operating point. Confirm every
headline number with **AutoAttack** (PGD over-estimates): drop `--no_autoattack`,
`--max_test_batches -1`.

**Re-evaluate the 50-epoch pilot `topoacl`/`rawacl` checkpoints under this protocol**
as a quick sanity check — but they're undertrained (clean ~62%), so the *headline*
topology comparison needs fresh 300-epoch runs (below).

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

### Phase 0 — DONE. Gate passed (after fixing the eval protocol).
The 50-ep pilot read ~0% robust for everything — but that was the clean-BN/clean-probe
artifact. Re-measured correctly, `adv_baseline` at **300 ep reaches ~29% PGD-100**
(adv-BN + robust probe), confirming adversarial training + dual-BN works. The
mechanism test (`topoacl` Γ-drop −0.3% vs `rawacl` −12%) already shows the
topology-specific effect. → proceed to the headline matrix.

### Phase 1 — CIFAR-10 home matrix (THE main result). ~300 ep, corrected protocol.
**Headline config** — the proven setup where robustness actually exists (300 ep is
the confirmed point; 200 may undertrain robustness). The launcher defaults
`ROBUST_PROBE=true BN_BRANCH=adv`, so every method is measured on the protocol that
exposes robustness:
```bash
DATASET=cifar10 BACKBONE=resnet18 \
  METHODS="baseline adv_baseline adv_phsim adv_swcontrol topoacl rawacl" \
  SUPERVISED="pgd_at trades" \
  SEEDS="0 1 2" EPOCHS=300 SAVE_EVERY=100 WARMUP=10 ADV_STEPS=5 DUAL_BN=true \
  MAX_TEST_BATCHES=-1 \
  bash scripts/run_program.sh
```
Disk: checkpoints are now **slim (model-only milestones + one rolling `last.pt`** for
resume), ~67% smaller. With `SAVE_EVERY=100` the whole 6-method × 3-seed matrix is
~5–7 GB — fits the shared disk. Before launching, free space (`conda clean -a -y`,
delete the pilot + any old fat intermediate checkpoints) and check `df -h .`.
This produces ONE robustness table (full test set + AutoAttack, adv-BN + robust
probe) with: baseline, adv_baseline (= **AdvCL**, dual-BN), adv_phsim, adv_swcontrol,
topoacl, rawacl, **PGD-AT**, **TRADES**. **The decisive cells:** `topoacl` vs `rawacl`
(topology-specific?) and `topoacl` vs `adv_baseline` (beats plain ACL?) on robust
accuracy — now measurable because robustness exists in this regime.
(3 seeds, not 5 — see the deadline math; 3 seeds buys a second dataset, which a
reviewer values more than 2 extra seeds.) Add the **RoCL** row once that port is
built (next).

### Phase 2 — generalization datasets, on Lambda IN PARALLEL with local CIFAR-10.
Run these on rented Lambda **A10** GPUs ($1.29/hr; ≈ the local card's speed — do
NOT rent A100/H100) *while* CIFAR-10 runs locally, so they cost ~0 extra wall-clock.
Reduced protocol (4 methods, 2 seeds, 200 ep, steps=3) — standard for secondary
datasets. **`stl10` must be the downsized `stl10_64`** (native 96×96 makes the PH
methods ~15× costlier/epoch and blows the budget; 64px is still 4× CIFAR's pixels).
All resolutions/labels are validated end-to-end with real data.

**Preflight on each fresh Lambda box (1 epoch, ~minutes) BEFORE the real run** —
catches a bad env/download for cents instead of dollars:
```bash
export TMPDIR=$PWD/tmp && mkdir -p tmp
DATASET=cifar100 BACKBONE=resnet18 METHODS="topoacl" SEEDS="0" \
  EPOCHS=1 SAVE_EVERY=1 WARMUP=0 ADV_STEPS=2 DUAL_BN=true MAX_TEST_BATCHES=2 \
  PROBE_EPOCHS=1 PROBE_PER_CLASS=20 \
  OUT=runs/preflight bash scripts/run_program.sh && rm -rf runs/preflight
```
Then the real runs (one per Lambda box):
```bash
# box A (~$65, ~2-3 days)
DATASET=cifar100 BACKBONE=resnet18 METHODS="baseline adv_baseline topoacl rawacl" \
  SEEDS="0 1" EPOCHS=200 SAVE_EVERY=100 WARMUP=10 ADV_STEPS=3 DUAL_BN=true \
  MAX_TEST_BATCHES=-1 bash scripts/run_program.sh

# box B (~$190 at 2 seeds; ~$95 at SEEDS="0"; ~4-6 days)
DATASET=stl10_64 BACKBONE=resnet18 METHODS="baseline adv_baseline topoacl rawacl" \
  SEEDS="0 1" EPOCHS=200 SAVE_EVERY=100 WARMUP=10 ADV_STEPS=3 DUAL_BN=true \
  MAX_TEST_BATCHES=-1 bash scripts/run_program.sh
```
Budget: ~$255 of the $276 for both at 2 seeds (thin margin — drop STL to `SEEDS="0"`
for ~$160 total and a safety buffer). **Tear down each Lambda instance the moment its
run + eval finishes** (it bills idle time). Copy the result JSONs back to git/home.

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
