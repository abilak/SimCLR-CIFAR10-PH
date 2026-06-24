#!/usr/bin/env python3
"""
eval_robustness.py  (Tier-1 #1)

Credible adversarial-robustness evaluation for a pretrained SimCLR/PHSim encoder:
trains a linear probe on frozen features, then runs the full masking-aware suite
(PGD step/restart/eps curves, Square, AutoAttack, black-box transfer) from
phtopo.robustness.

Examples
--------
  # Single checkpoint, with a baseline surrogate for transfer
  python eval_robustness.py \
      --ckpt checkpoints/upstream/phsim/seed0/epoch100/simclr_phsim_resnet18_epoch100_seed0.pt \
      --surrogate_ckpt checkpoints/upstream/baseline/seed0/epoch100/simclr_baseline_resnet18_epoch100_seed0.pt \
      --out runs/robustness/phsim_seed0_e100.json

Install AutoAttack on the run machine for the authoritative number:
  pip install git+https://github.com/fra31/auto-attack
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

from models import SimCLR, BACKBONES
from datasets import make_eval_sets, num_classes as ds_num_classes, label_of
from phtopo.robustness import run_robustness_suite
from phtopo.attacks import pgd_linf
from phtopo.dual_bn import load_state_dict_auto, bn_route, has_dual_bn


def build_encoder(backbone, projection_dim, proj_hidden_dim, reduce_channels, device,
                  ph_source_layer="layer4", ph_extra_layers=()):
    base = BACKBONES[backbone]
    m = SimCLR(base, projection_dim=projection_dim, proj_hidden_dim=proj_hidden_dim,
               reduce_channels=reduce_channels, cifar_no_maxpool=True,
               ph_source_layer=ph_source_layer, ph_extra_layers=tuple(ph_extra_layers)).to(device)
    return m


class LinearEvalModel(nn.Module):
    """
    Frozen encoder + linear head; forward takes images in [0,1] -> logits.

    bn_branch (dual-BN models only): route every forward -- probe training, the
    attack, and the robustness suite -- through a specific BatchNorm branch
    ("clean" or "adv"). For AdvProp/AdvCL-style training the robustness lives in
    the adv-BN branch, so evaluating adversarial inputs through clean-BN (the
    default) underestimates it. None = default route (clean) / plain BN.
    """
    def __init__(self, simclr_model, feature_dim, n_classes=10, bn_branch=None):
        super().__init__()
        self.simclr = simclr_model
        self.fc = nn.Linear(feature_dim, n_classes)
        self.bn_branch = bn_branch if (bn_branch and has_dual_bn(simclr_model)) else None

    def forward(self, x):
        if self.bn_branch is not None:
            with bn_route(self.bn_branch):
                _, h, _ = self.simclr(x)
        else:
            _, h, _ = self.simclr(x)
        return self.fc(h)


def load_encoder_from_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)  # our own ckpt (has cfg/RNG)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    backbone = cfg.get("backbone", "resnet18")
    projection_dim = int(cfg.get("projection_dim", 64))
    proj_hidden_dim = int(cfg.get("model", {}).get("proj_hidden_dim", 512)) if isinstance(cfg.get("model"), dict) else 512
    ph_cfg = cfg.get("ph", {}) if isinstance(cfg.get("ph"), dict) else {}
    reduce_channels = int(ph_cfg.get("reduce_channels", 8))
    # Fallback to layer4 for old checkpoints that predate the source_layer option.
    ph_source_layer = str(ph_cfg.get("source_layer", "layer4"))
    ph_extra_layers = tuple(ph_cfg.get("extra_layers", []) or [])
    enc = build_encoder(backbone, projection_dim, proj_hidden_dim, reduce_channels, device,
                        ph_source_layer=ph_source_layer, ph_extra_layers=ph_extra_layers)
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    # Auto-converts the encoder to dual-BN iff the checkpoint is dual-BN (then the
    # clean branch -- default route -- is used for the frozen-feature eval).
    load_state_dict_auto(enc, state, strict=True)
    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()
    return enc, enc.feature_dim


def train_linear_probe(model, train_loader, device, epochs, lr,
                       robust=False, eps=8.0 / 255.0, pgd_steps=10):
    """
    Train the linear head on the FROZEN encoder. If robust=True, do *robust linear
    evaluation* -- train the head on PGD adversarial examples (the standard way
    RoCL/AdvCL report robustness). A clean-trained head sitting on robust features
    can score near-0% robust even when the features are robust, so this is the
    apples-to-apples protocol. The encoder stays frozen / BN in eval throughout
    (re-asserted after each attack, since pgd_linf toggles train mode).
    """
    params = [p for p in model.fc.parameters()]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    for ep in range(epochs):
        model.train()
        model.simclr.eval()  # keep encoder frozen / BN in eval
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if robust:
                x = pgd_linf(model, x, y, eps=eps, steps=pgd_steps, restarts=1)
                model.simclr.eval()  # pgd_linf restored train mode -> re-freeze encoder BN
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step(); sched.step()
    model.eval()
    return model


def make_loaders(data_dir, batch_size, per_class, workers, dataset="cifar10"):
    train, test = make_eval_sets(dataset, root=data_dir, download=True)  # [0,1], attacks operate here
    nc = ds_num_classes(dataset)
    # small balanced labeled probe set (reads labels directly -- no image decode)
    idx_by_class = {c: [] for c in range(nc)}
    for i in range(len(train)):
        y = label_of(train, i)
        if len(idx_by_class[y]) < per_class:
            idx_by_class[y].append(i)
        if all(len(v) >= per_class for v in idx_by_class.values()):
            break
    indices = [i for c in range(nc) for i in idx_by_class[c]]
    train_loader = DataLoader(train, batch_size=batch_size,
                              sampler=SubsetRandomSampler(indices), num_workers=workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, test_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--surrogate_ckpt", default=None, help="baseline ckpt for black-box transfer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--probe_per_class", type=int, default=500)
    ap.add_argument("--probe_epochs", type=int, default=20)
    ap.add_argument("--probe_lr", type=float, default=0.1)
    ap.add_argument("--eps_px", type=float, default=8.0)
    ap.add_argument("--max_test_batches", type=int, default=-1,
                    help="limit test batches for the (expensive) attack suite")
    ap.add_argument("--no_autoattack", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dataset", default="cifar10", help="cifar10 | cifar100 | stl10")
    ap.add_argument("--robust_probe", action="store_true",
                    help="robust linear evaluation: train the head on PGD adversarial examples")
    ap.add_argument("--bn_branch", default=None, choices=["clean", "adv"],
                    help="dual-BN only: route all forwards through this branch (default=clean)")
    ap.add_argument("--probe_pgd_steps", type=int, default=10, help="PGD steps for --robust_probe")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[robustness] device={device}")

    n_cls = ds_num_classes(args.dataset)
    eps = args.eps_px / 255.0
    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.probe_per_class,
                                             args.workers, dataset=args.dataset)

    enc, feat_dim = load_encoder_from_ckpt(args.ckpt, device)
    model = LinearEvalModel(enc, feat_dim, n_classes=n_cls, bn_branch=args.bn_branch).to(device)
    print(f"[robustness] dataset={args.dataset} ({n_cls} classes) | probe="
          f"{'robust' if args.robust_probe else 'clean'} | bn_branch={model.bn_branch or 'clean(default)'}")
    train_linear_probe(model, train_loader, device, args.probe_epochs, args.probe_lr,
                       robust=args.robust_probe, eps=eps, pgd_steps=args.probe_pgd_steps)

    surrogate = None
    if args.surrogate_ckpt:
        senc, sfeat = load_encoder_from_ckpt(args.surrogate_ckpt, device)
        surrogate = LinearEvalModel(senc, sfeat, n_classes=n_cls, bn_branch=args.bn_branch).to(device)
        print("[robustness] training surrogate probe (for transfer) ...")
        train_linear_probe(surrogate, train_loader, device, args.probe_epochs, args.probe_lr,
                           robust=args.robust_probe, eps=eps, pgd_steps=args.probe_pgd_steps)

    print("[robustness] running suite ...")
    res = run_robustness_suite(
        model, test_loader, device, eps=args.eps_px / 255.0,
        surrogate=surrogate, run_autoattack=not args.no_autoattack,
        max_batches=args.max_test_batches,
    )
    res["ckpt"] = args.ckpt
    res["surrogate_ckpt"] = args.surrogate_ckpt
    res["clean_probe_per_class"] = args.probe_per_class
    res["probe"] = "robust" if args.robust_probe else "clean"
    res["bn_branch"] = model.bn_branch or "clean"

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"[robustness] wrote {args.out}")
    print("  clean      :", round(res["clean_acc"], 4))
    print("  PGD(steps) :", {k: round(v, 4) for k, v in res["pgd_by_steps"].items()})
    print("  PGD(eps_px):", {k: round(v, 4) for k, v in res["pgd_by_eps_px"].items()})
    if "autoattack" in res:
        print("  AutoAttack :", res["autoattack"])
    if "transfer_acc" in res:
        print("  transfer   :", round(res["transfer_acc"], 4))
    print("  MASKING    :", res["masking"]["verdict"], "| headline robust =", res["masking"]["robust_acc_headline"])


if __name__ == "__main__":
    main()
