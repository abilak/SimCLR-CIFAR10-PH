# Tier-1 Experiments — what was built and how to run it

This documents the rebuild of the experimental half of the paper to address the
four reviewer-critical concerns. The common theme of the fix: the old PH path was
**CPU-only and non-differentiable** (diagrams were detached to numpy via
`ripser`/`persim`, so no gradient ever flowed through the topology — "PHSim" was
really SimCLR with PH-selected hard negatives, not a method that controls Γ). It
is replaced by `phtopo/`, a differentiable, batched, GPU-ready PH toolkit.

## `phtopo/` — the new core

| file | what |
|---|---|
| `diffph.py` | Pure-PyTorch **differentiable H0 Vietoris–Rips** (vectorized Prim's MST; H0 deaths == MST edge weights, gradient flows to point coords) + **sliced-Wasserstein** between persistence diagrams (handles the diagonal; `torch.sort` so it backprops). No `cdist` → runs natively on CPU/CUDA/MPS. |
| `losses.py` | `topo_separation_loss` (**method=phsim**): differentiable persistent-separation triplet — pulls each anchor's diagram toward its positive partner's and pushes it from its hardest negative; gradient `X→H0→SW→loss`. `raw_sw_separation_loss` (**method=swcontrol**): identical triplet but SW on the **raw point clouds, no persistence** — the matched non-topological control. |
| `descriptors.py` | GUDHI H0/H1 diagrams + Betti counts, persistence entropy, total persistence, and class-separation Γ (eval-only diagnostics). |
| `attacks.py` | PGD (configurable steps/restarts/eps) + black-box transfer. |
| `robustness.py` | Full masking-aware suite + verdict. |
| `stats.py` | Spearman+CI+p, epoch-controlled partial correlation, matched-clean control. |

Correctness is unit-tested in `tests/test_diffph.py` (H0 deaths match GUDHI
exactly; gradchecks pass; SW metric properties).

---

## The four experiments

### #1 — Real robustness + gradient-masking diagnostics  (`eval_robustness.py`)
PGD-10 alone is not credible; a TDA regularizer can mask gradients. The suite, at
ε=8/255, reports: clean acc; **PGD step curve {10,20,50,100}** (must plateau);
**PGD restart curve**; **ε-sweep to large ε** (robust acc must → 0); **Square**
(gradient-free) and **AutoAttack** (APGD-CE/T, FAB-T, Square) via the official
`autoattack` package; and **black-box transfer** from a baseline surrogate. It
emits explicit masking flags + a verdict and reports AutoAttack as the headline
robust accuracy. Run ≥5 seeds (`SEEDS` in `scripts/run_tier1.sh`) and report
mean±std.

### #2 — Topology-specificity controls  (`method=swcontrol` + `analyze_results.py`)
Two controls isolate whether *persistence* matters:
- **Non-topological functional control:** `swcontrol` swaps the persistence step
  for sliced-Wasserstein on the raw neighborhoods, matched in form/magnitude. If
  its robustness ≈ phsim's, the effect is "some Wasserstein penalty," not topology.
- **Matched-clean-accuracy control:** `matched_clean_control` finds baseline
  checkpoints whose clean acc matches phsim's and compares robustness. If the
  matched baseline is equally robust, the result is the generic accuracy–robustness
  tradeoff, not topology.

### #3 — Topology-under-attack mechanism test  (`eval_mechanism.py`)
Measures β0/β1, persistence entropy, total persistence, and class-separation Γ of
class-conditioned embeddings under **clean vs PGD** inputs, per method. The figure
the paper needs: baseline topology **collapses** under attack (Γ drops, components
merge) while phsim's is **preserved**. Quantified as relative Γ drop.

### #4 — Statistical rehabilitation  (`analyze_results.py`)
Spearman with **p-value, n, and bootstrap 95% CI**, pooled across all seeds and
checkpoints, plus **partial Spearman controlling for training epoch**.

> **Finding on the committed data (`output/gamma_adv_vs_pgd/merged_seed1.csv`,
> n=30/method):** the paper's headline Spearman could not be reproduced from this
> file. Γ_adv vs PGD for PHSim is **ρ=−0.03, p=0.87** (CI [−0.37, +0.30]) —
> indistinguishable from zero. PGD accuracy is **ρ=−0.92** with epoch, so any raw
> Γ–PGD correlation is an epoch artifact; controlling for epoch leaves nothing
> (partial ρ=+0.07, p=0.73). The only nominally significant raw correlation,
> Γ_clean vs PGD (ρ=−0.43, p=0.017), is the **wrong sign** and **also vanishes**
> under epoch control (partial ρ=+0.13, p=0.49). Until the multi-seed sweep with
> the *differentiable* Γ produces a real epoch-controlled partial correlation, the
> honest claim is "Γ above a threshold regime," not "monotonic predictor."

---

## How to run

```bash
pip install -r requirements.txt
pip install git+https://github.com/fra31/auto-attack     # AutoAttack
bash scripts/run_tier1.sh                                # edit SEEDS/EPOCHS/OUT at top
```

Outputs land in `runs/tier1/{upstream,robustness,mechanism,stats}/`.

### Local (no GPU) validation
Everything is validated locally on CPU with a toy config; `device=cpu` pins the
device (MPS works too after the `cdist` removal, but lacks some backward kernels
on older torch). The statistics (#4) run fully on CPU on existing CSVs:

```bash
python analyze_results.py --csv output/gamma_adv_vs_pgd/merged_seed1.csv \
    --gamma_col gamma_adv --pgd_col pgd_acc --epoch_col epoch --out runs/stats/seed1.json
python tests/test_diffph.py
```

## Methods
`baseline` (NT-Xent), `phsim` (differentiable persistent separation),
`swcontrol` (non-topological control), `hybrid` (α·NT-Xent + (1−α)·phsim).
