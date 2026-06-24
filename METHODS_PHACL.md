# Adversarial PH-ACL + improvements — what was added and why

This builds on `EXPERIMENTS_TIER1.md`. The pilot showed the clean-trained method
fails the topology-specificity control (phsim ≈ swcontrol on both robustness,
p=0.875, and topology-stability, p=0.892). The diagnosis: **with clean training,
nothing in the loop sees the adversary, so "shape the persistence diagram" and
"shape the raw point cloud" are nearly the same operation.** Topology can only
earn its place when the objective is to *preserve multiscale structure against a
worst-case perturbation* — which is exactly what the paper's theory (Γ_adv =
sup over perturbations) is about, but the original experiments never trained.

Everything below is correctness-tested (`tests/test_diffph.py`, `tests/test_adv.py`):
H0 matches GUDHI, gradchecks pass, PGD ascent increases loss within the Linf ball,
all methods give finite losses with gradient to the encoder, BN train-mode is
restored after the inner loop, and all checkpoint variants round-trip strict=True.

## Audit fixes (correctness, applied first)
1. **Sliced-Wasserstein normalization** was `mean` over points → distorted
   comparisons between *different-size* diagrams (affected the Γ eval & mechanism
   numbers). Now canonical `sum` over points, mean over directions — verified to
   match an independent SW reference exactly.
2. **`n_views==2` crash** (empty negatives) — guarded.
3. **Raw control used different random projections for positives vs negatives** —
   now shares directions (fair, reproducible control).
4. **PH trained on layer4 = 2×2 = 4-point clouds** (near-trivial topology, a
   likely reason topology added nothing). PH source layer is now configurable;
   default `layer3` (4×4 = 16 pts), `layer2` (8×8 = 64 pts) for richer/H1.

## New methods (config `method=`)
| method | objective | what it tests |
|---|---|---|
| `baseline` | clean NT-Xent | reference (high clean, ~0 robust) |
| `phsim` | clean differentiable topo separation | (clean) topology |
| `swcontrol` | clean raw-SW separation | (clean) non-topo control |
| `adv_baseline` | **adversarial NT-Xent = ACL** | does adversarial training give real robustness? (proper robust-SSL reference) |
| `adv_phsim` | adversarial topo separation | topology-specific *under attack*? |
| `adv_swcontrol` | adversarial raw-SW separation | non-topo control, adversarial regime |
| `topoacl` | **NT-Xent + β·topological adversarial consistency** (flagship) | high clean **and** robust **and** topology-specific? |
| `rawacl` | NT-Xent + β·raw consistency | topoacl's non-topological control |
| `hybrid` | α·NT-Xent + (1−α)·topo | keep clean high + add (clean) separation |

**Adversarial machinery** (`phtopo/adv.py`): a generic inner PGD that *maximizes*
any SSL/topology loss within an Linf ball (encoder in eval mode so BN stats
aren't corrupted), then a first-order outer step on the detached adversarial
input. No double-backprop. Threat-model-consistent: perturbs pixels in [0,1].

**Topological adversarial consistency** (`phtopo/losses.topo_consistency_loss`,
the flagship idea): the inner adversary finds the perturbation that most changes
each sample's H0 persistence diagram (max SW(PD(clean), PD(adv))); the encoder is
trained to make that drift small (be *topologically invariant* to attack), while
NT-Xent keeps clean accuracy high. `rawacl` is the identical objective with
persistence removed — the control that says whether the diagram-level invariance
matters or it's just generic geometry.

## Other improvements
- **Multiscale PH** (`ph.extra_layers=[layer2]`): persistence at several network
  depths, loss averaged over scales — operationalizes the paper's "multiscale"
  claim literally. Backward-compatible checkpoints.
- **Soft-negative aggregation** (`ph.neg_agg=soft`): smooth-min over all
  negatives instead of the single hardest — smoother gradients, less noise.
  (Note: smooth-min is a lower bound on the true min; use `softmin_temp≈0.01`
  to keep the bias negligible.)
- **Richer point cloud** (`ph.source_layer`, `ph.num_points`) — see audit #4.
- **Adversarial curriculum** (`adv.ramp_epochs=N`): linearly ramps eps and beta
  from ~0 over the first N epochs — stabilizes adversarial-from-scratch training.
- **EMA-teacher topological self-distillation under attack** (`adv.teacher=ema`,
  the second novel idea): a momentum teacher computes a *stable* clean-topology
  target (always eval-mode, no grad); `topoacl`/`rawacl` train the student so its
  worst-case-perturbed diagram matches the teacher's clean one. This gives a
  collapse-free target (BYOL/DINO-style) and removes the BN-mode mismatch by
  construction (teacher and the student consistency branch are both eval-mode;
  BN is still trained by the clean contrastive base). The teacher receives no
  gradient and is EMA-updated after each step.

## Correctness notes (second audit pass)
An independent review of the new adversarial code confirmed: first-order training
(no gradient leakage to params; verified `param.grad is None` after the inner
ascent), exact Linf/[0,1] projection, correct ascent, single-backward graph
reuse, and a finite-difference-correct consistency gradient (rel err 1.5e-9).
Three issues were found and fixed: (a) the consistency inner-max compared
eval-mode adv vs train-mode clean diagrams (a constant BN-mode offset) — now uses
an eval-mode clean reference so the inner objective is 0 at the clean input;
(b) the eval/train toggle is now an exception-safe `eval_mode()` context;
(c) the multiscale aggregation asserts equal map counts. All in `tests/test_adv.py`.

## Further ideas (not yet implemented — roadmap)
- **Dual-BN (AdvCL-style)**: separate BN for clean vs adversarial branches —
  the known key enabler for adversarial SSL; would likely lift every adv method.
  Higher implementation surface (BN routing + checkpoint plumbing).
- **Persistence-image / landscape auxiliary head**: a differentiable diagram
  vectorization feeding a small predictor — richer topological signal than a
  scalar SW.
- **Memory-queue neighborhoods (MoCo-style)**: build Z+/Z- from a queue of past
  embeddings so the diagrams capture class-level (not just per-image) topology,
  and H1 loops have enough points to matter.
- **Topological attack at eval**: a white-box attack that maximizes class-topology
  disruption, as a mechanism-probe robustness number.

## The decisive comparisons (what would save the paper)
1. `adv_baseline` vs `baseline`: adversarial training should lift robust accuracy
   from ~0% into the double digits. Establishes the robust regime.
2. `adv_phsim` vs `adv_swcontrol` **and** `topoacl` vs `rawacl`: the
   topology-specificity tests *in the adversarial regime*. If topo > raw here
   (≥5 seeds, AutoAttack, significance), persistence finally earns its place.
3. `topoacl` vs `adv_baseline`: does the topological consistency term add
   robustness **beyond** plain ACL?
4. clean accuracy of `topoacl`: NT-Xent should keep it near baseline — i.e. the
   gain is **not** the accuracy–robustness tradeoff.

If topo still ≈ raw after all this, that is a clean, honest negative and the
defensible paper is theory + the mechanism story + the negative control result
(see `EXPERIMENTS_TIER1.md`).

## How to run
```bash
bash scripts/run_phacl.sh          # edit knobs at top; pilot defaults (3 seeds, 50 ep)
# then merge + stats:
python scripts/merge_tier1.py --root runs/phacl --out runs/summary/phacl_merged.csv
python analyze_results.py --csv runs/summary/phacl_merged.csv \
    --gamma_col gamma --pgd_col pgd_acc --clean_col clean_acc --epoch_col epoch \
    --out runs/phacl/stats/report.json
```

## Cost
Adversarial methods cost ≈ `adv.steps`× the clean per-step time (default 5 → ~5×).
`topoacl`/`rawacl` add one clean forward on top. With layer3 + `adv.steps=5`,
budget a pilot run at a few× the clean pilot. Use `adv.steps=3` and
`--max_test_batches` during iteration; raise for final numbers. A clean warmup
(`train.warmup_epochs=5`) stabilizes the adversarial methods early in training.
