# PHSim: Persistent Homology for Adversarial Self-Supervised Learning

This repository implements **PHSim**, a topology-aware self-supervised learning method that integrates persistent homology into SimCLR-style contrastive representation learning.

The project empirically studies the central claim of the paper:

> **Adversarially stable multiscale topological separation in upstream self-supervised representations is associated with improved downstream adversarial robustness.**

PHSim is evaluated on CIFAR-10 using a ResNet-18 encoder, frozen linear probing, and PGD adversarial evaluation.

---

## 1. Problem Setting

### CIFAR-10 task

CIFAR-10 is a 10-class image classification dataset. The downstream supervised task is to classify each image into one of the following classes:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

### Two-phase SSL evaluation protocol

We follow the standard self-supervised learning pipeline.

| Phase | Description | Labels used? |
|---|---|---|
| **Upstream pretraining** | Train an encoder with SimCLR-style contrastive learning or PHSim. | No |
| **Downstream linear probing** | Freeze the encoder and train a linear classifier on CIFAR-10 labels. | Yes |

The goal is not only to maximize clean CIFAR-10 accuracy, but to test whether the upstream representation structure improves **downstream adversarial robustness**.

---

## 2. Positive and Negative Samples

In upstream SimCLR-style training, labels are not used. Positive and negative pairs are constructed from data augmentations.

For an anchor image `x`:

- A **positive sample** is another augmented view of the same original image.
- A **negative sample** is an augmented view of a different image in the batch.

Example:

```text
anchor:      crop/color-jitter of image A
positive:    another augmented view of image A
negatives:   augmented views of images B, C, D, ...
```

Standard SimCLR uses pairwise cosine similarity between embeddings. PHSim instead compares the **topological structure** of positive and negative embedding neighborhoods.

---

## 3. Persistent Separation Score Gamma

For an encoder `f` and anchor image `x`, PHSim forms two embedding neighborhoods:

```math
Z^+(x) = \{ f(x_i^+) \}_{i=1}^k,
\qquad
Z^-(x) = \{ f(x_i^-) \}_{i=1}^k.
```

Here:

- `Z⁺(x)` is the positive embedding neighborhood.
- `Z⁻(x)` is the negative embedding neighborhood.

We compute Vietoris--Rips persistence diagrams for both neighborhoods:

```math
PD(Z^+(x)),
\qquad
PD(Z^-(x)).
```

The persistent separation score is

```math
\Gamma(f;x)
=
d_{\mathrm{SW}}\left(PD(Z^+(x)), PD(Z^-(x))\right),
```

where `d_SW` is the sliced-Wasserstein distance between persistence diagrams.

### Interpretation

- **Small Gamma**: positive and negative neighborhoods have similar multiscale topology.
- **Large Gamma**: positive and negative neighborhoods are more topologically separated.

A large Gamma means the representation is not only separating samples by local pairwise distance, but also organizing positive and negative neighborhoods into more distinct multiscale structures.

We also evaluate the adversarial version:

```math
\Gamma_{\mathrm{adv}}(f;x)
=
\sup_{x' \in U_\epsilon(x)} \Gamma(f;x'),
```

where `U_epsilon(x)` is the allowed adversarial perturbation set.

---

## 4. Methods Compared

| Method | Description |
|---|---|
| `baseline` | Standard SimCLR / NT-Xent contrastive learning. |
| `phsim` | Persistent-homology-guided contrastive learning using sliced-Wasserstein separation between persistence diagrams. |

Both methods use the same CIFAR-10 data, ResNet-18 backbone, downstream linear probing protocol, and adversarial evaluation setup.

---

## 5. Experimental Setup

| Component | Setting |
|---|---|
| Dataset | CIFAR-10 |
| Encoder | ResNet-18 |
| Upstream training | SimCLR-style self-supervised pretraining |
| Downstream task | Frozen encoder + linear classifier |
| Clean metric | Standard CIFAR-10 accuracy |
| Robust metric | PGD robust accuracy |
| Attack | PGD-10 |
| Threat model | `l_inf` |
| Epsilon sweep | `{0, 2, 4, 6, 8, 10}/255` |

---

## 6. Experiments and Results

### Experiment 1: Adversarial Gamma vs. Downstream Robustness

**Goal.** Test whether adversarial persistent separation is aligned with downstream PGD robustness.

We compute `Gamma_adv` across upstream checkpoints and correlate it with downstream PGD robust accuracy.

| Method | Spearman(`Gamma_adv`, PGD accuracy) |
|---|---:|
| Baseline | -0.0494 |
| PHSim | **0.1572** |

**Takeaway.** Baseline shows essentially no monotonic relationship. PHSim shows positive monotonic alignment: larger adversarial persistent separation tends to correspond to better downstream PGD robustness.

This supports the paper's topology-to-downstream mechanism, but should be interpreted carefully as empirical alignment rather than a strong pointwise prediction law.

---

### Experiment 2: Persistent Separation and Downstream Performance

**Goal.** Test whether PHSim shifts representations into a higher-Gamma regime and whether that regime is associated with better robustness.

At best-clean checkpoints across runs, we compare mean Gamma, clean accuracy, and PGD robust accuracy.

| Method | Mean Gamma | Mean Clean Accuracy | Mean PGD Accuracy |
|---|---:|---:|---:|
| Baseline | 36.41 | 0.455 | 0.149 |
| PHSim | **37.58** | 0.387 | **0.196** |

**Takeaway.** PHSim increases mean persistent separation by about `+1.17` and improves PGD robust accuracy by about 31% relative to baseline.

However, PHSim does not improve clean accuracy. This suggests that PHSim's advantage is robustness-specific rather than a generic accuracy gain.

Important nuance: pooled Pearson correlations between Gamma and accuracy are weak, so Gamma should not be presented as a simple linear predictor of accuracy. The stronger claim is that PHSim induces a higher-Gamma, robustness-relevant representation regime.

---

### Experiment 3: Robustness Across Attack Strengths

**Goal.** Test whether PHSim remains robust as the PGD attack strength increases.

We evaluate PGD-10 robust accuracy across

```math
\epsilon \in \{0,2,4,6,8,10\}/255.
```

We report robustness curves and area under the curve (AUC).

At upstream epoch 10:

| Method | PGD Accuracy at `8/255` |
|---|---:|
| Baseline | 0.0081 |
| PHSim | **0.0559** |

**Takeaway.** At epoch 10, PHSim achieves roughly `7x` higher PGD robustness at `8/255` and a larger robustness AUC.

At later epochs, the baseline narrows the gap, suggesting that PHSim mainly accelerates early robustness formation by shaping representation topology.

---

## 7. Main Findings

1. **PHSim produces higher persistent separation.**
   PHSim has higher mean Gamma than baseline at evaluated checkpoints.

2. **PHSim improves downstream PGD robustness.**
   PHSim improves robust accuracy despite lower clean accuracy.

3. **The robustness gain is strongest early in training.**
   At epoch 10, PHSim substantially outperforms baseline under PGD attacks.

4. **Gamma is a structural signal, not a standalone accuracy predictor.**
   Gamma is useful for understanding representation topology, but it should not be overclaimed as a simple linear predictor of clean or robust accuracy.

---

## 8. Repository Structure

```text
SIMCLR-CIFAR10-PH/

├── data/
│   └── cifar-10-batches-py/
│
├── output/
│   ├── eps_robustness/
│   │   ├── eps_curve_upE5.png
│   │   ├── eps_curve_upE10.png
│   │   ├── eps_curve_upE15.png
│   │   ├── eps_curve_upE20.png
│   │   ├── eps_curve_upE25.png
│   │   ├── eps_curve_upE30.png
│   │   └── merged_all_methods_all_seeds.csv
│   │
│   ├── gamma_adv_vs_pgd/
│   │   ├── gamma_adv_vs_pgd.png
│   │   └── merged_seed1.csv
│   │
│   └── gamma_vs_acc/
│       ├── gamma_analysis_summary_table.csv
│       ├── gamma_analysis_summary.json
│       ├── gamma_vs_clean_best_clean_epoch.png
│       └── gamma_vs_pgd_best_clean_epoch.png
│
├── scripts/
│   ├── sweep_upstream_train.py
│   ├── run_downstream_then_merge.py
│   └── sweep_full_pipeline.py
│
├── utils/
│   ├── eps_robustness/
│   │   ├── sweep_eps.py
│   │   └── aggregate_eps_sweep.py
│   │
│   ├── gamma_adv_vs_pgd/
│   │   ├── eval_upstream_gamma_adv.py
│   │   ├── merge_gamma_adv_vs_pgd.py
│   │   └── run_gamma_adv_seed1.sh
│   │
│   └── gamma_vs_pgd/
│       ├── analyze_gamma_vs_robustness.py
│       └── merge_and_plot_gamma_pgd.py
│
├── visuals/
├── simclr.py              # upstream pretraining
├── simclr_lin.py          # downstream linear probing and PGD evaluation
├── models.py              # ResNet backbone and projection heads
├── simclr_config.yaml     # Hydra configuration
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 9. Running Experiments

### Upstream pretraining

```bash
python simclr.py backbone=resnet18 method=phsim seed=0
```

For the baseline:

```bash
python simclr.py backbone=resnet18 method=baseline seed=0
```

### Downstream linear probing with PGD evaluation

```bash
python simclr_lin.py \
  backbone=resnet18 \
  method=phsim \
  load_epoch=10 \
  attack.enabled=true \
  attack.pgd=true
```

### Epsilon sweep

Use the scripts under:

```text
utils/eps_robustness/
```

Expected outputs include robustness curves and merged CSV files under:

```text
output/eps_robustness/
```

### Gamma analysis

Use the scripts under:

```text
utils/gamma_adv_vs_pgd/
utils/gamma_vs_pgd/
```

Expected outputs include Gamma-vs-PGD scatter plots, Gamma-vs-clean plots, and summary tables under:

```text
output/gamma_adv_vs_pgd/
output/gamma_vs_acc/
```

---

## 10. Summary

PHSim provides an empirical testbed for the idea that persistent-homology structure in self-supervised representations is relevant to adversarial robustness.

The key experimental conclusion is:

> PHSim shifts representations toward a higher persistent-separation regime and improves downstream PGD robustness, especially during early training.

At the same time, Gamma should be interpreted as a structural representation metric rather than a direct accuracy predictor.
