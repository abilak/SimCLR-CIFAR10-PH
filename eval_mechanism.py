#!/usr/bin/env python3
"""
eval_mechanism.py  (Tier-1 #3)

Directly tests the theory's stated mechanism: "under attack, clusters merge and
components collapse even when margins hold." For each method's encoder we measure
the topology of class-conditioned embeddings under CLEAN vs PGD-attacked inputs:

  * beta0 proxy (# significant H0 features), beta1 proxy (# H1 loops),
  * total persistence (H0, H1), persistence entropy (H0, H1),
  * class-separation Gamma = mean inter-class sliced-Wasserstein between diagrams.

The compelling result the paper needs: under attack the BASELINE's topology
collapses (Gamma drops, components merge) while PHSim's is preserved. We quantify
that as the relative drop  (Gamma_clean - Gamma_adv) / Gamma_clean  and per-class
descriptor shifts, and emit a grouped-bar figure.

Attack note: PGD is run against a trained linear probe (so "attack" means a real
classification adversary), then encoder features of the adversarial inputs are
analyzed. Inputs are in [0,1].

Example
-------
  python eval_mechanism.py \
      --ckpt baseline=ckpts/baseline.pt --ckpt phsim=ckpts/phsim.pt \
      --out runs/mechanism --per_class 80 --eps_px 8
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

from eval_robustness import (
    load_encoder_from_ckpt, LinearEvalModel, train_linear_probe, make_loaders,
)
from phtopo.attacks import pgd_linf
from phtopo.descriptors import topology_descriptors, class_separation_gamma


@torch.no_grad()
def encoder_features(enc, x):
    _, h, _ = enc(x)
    return h


def gather_class_features(enc, probe, loader, device, per_class, eps, attacked):
    """Collect up to per_class encoder features per class, clean or PGD-attacked."""
    feats = defaultdict(list)
    need = {c: per_class for c in range(10)}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if attacked:
            x = pgd_linf(probe, x, y, eps=eps, steps=20, restarts=1)
        h = encoder_features(enc, x).cpu().numpy()
        yy = y.cpu().numpy()
        for i in range(len(yy)):
            c = int(yy[i])
            if need[c] > 0:
                feats[c].append(h[i]); need[c] -= 1
        if all(v <= 0 for v in need.values()):
            break
    return {c: np.stack(v, 0) for c, v in feats.items() if len(v) > 0}


def descriptor_summary(feats_by_class, maxdim=1):
    """Mean per-class descriptors + class-separation Gamma."""
    per = defaultdict(list)
    for c, X in feats_by_class.items():
        d = topology_descriptors(X, maxdim=maxdim)
        for k, v in d.items():
            per[k].append(v)
    out = {f"mean_{k}": float(np.mean(v)) for k, v in per.items()}
    out["gamma"] = class_separation_gamma(feats_by_class, maxdim=maxdim)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="method=path, repeatable (e.g. baseline=..., phsim=...)")
    ap.add_argument("--out", default="runs/mechanism")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--per_class", type=int, default=80)
    ap.add_argument("--eps_px", type=float, default=8.0)
    ap.add_argument("--probe_per_class", type=int, default=500)
    ap.add_argument("--probe_epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[mechanism] device={device}")
    eps = args.eps_px / 255.0
    Path(args.out).mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = make_loaders(args.data_dir, 256, args.probe_per_class, args.workers)

    results = {}
    for spec in args.ckpt:
        method, path = spec.split("=", 1)
        print(f"[mechanism] {method}: {path}")
        enc, feat_dim = load_encoder_from_ckpt(path, device)
        probe = LinearEvalModel(enc, feat_dim).to(device)
        train_linear_probe(probe, train_loader, device, args.probe_epochs, 0.1)

        clean = gather_class_features(enc, probe, test_loader, device, args.per_class, eps, attacked=False)
        adv = gather_class_features(enc, probe, test_loader, device, args.per_class, eps, attacked=True)
        s_clean = descriptor_summary(clean)
        s_adv = descriptor_summary(adv)
        g_drop = (s_clean["gamma"] - s_adv["gamma"]) / max(1e-9, s_clean["gamma"])
        results[method] = {
            "clean": s_clean, "adv": s_adv,
            "gamma_clean": s_clean["gamma"], "gamma_adv": s_adv["gamma"],
            "gamma_relative_drop": g_drop,
        }
        print(f"    Gamma clean={s_clean['gamma']:.4f}  adv={s_adv['gamma']:.4f}  rel.drop={g_drop:+.2%}")

    with open(os.path.join(args.out, "mechanism.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Figure: class-separation Gamma, clean vs adv, per method.
    methods = list(results.keys())
    x = np.arange(len(methods)); w = 0.35
    gc = [results[m]["gamma_clean"] for m in methods]
    ga = [results[m]["gamma_adv"] for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, gc, w, label="clean")
    ax.bar(x + w / 2, ga, w, label="PGD-attacked")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel(r"Class-separation $\Gamma$ (inter-class SW)")
    ax.set_title("Topology under attack: collapse vs preservation")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "mechanism_gamma.png"), dpi=150)
    print(f"[mechanism] wrote {args.out}/mechanism.json and mechanism_gamma.png")


if __name__ == "__main__":
    main()
