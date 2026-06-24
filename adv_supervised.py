#!/usr/bin/env python3
"""
adv_supervised.py -- supervised adversarial-training reference baselines (item 8).

Trains PGD-AT (Madry et al. 2018) and TRADES (Zhang et al. 2019) classifiers and
evaluates them with the SAME masking-aware robustness suite + AutoAttack used for
the SSL methods (phtopo.robustness.run_robustness_suite), writing the SAME JSON
format -- so they slot directly into ONE robustness table alongside baseline /
adv_baseline (=AdvCL) / topoacl / rawacl.

These are the *reference upper bounds* the reviewer asked for: supervised AT
typically gives the strongest robustness (it trains end-to-end with labels), so it
bounds what the label-free SSL methods can hope to approach.

Architecture is matched to the SSL encoder (same CIFAR stem: conv1, no maxpool),
so the comparison is apples-to-apples. Threat model is identical: Linf, eps in
[0,1] pixel space.

Examples
--------
  # train + eval a PGD-AT ResNet-18 on CIFAR-10
  python adv_supervised.py --method pgd_at --dataset cifar10 --backbone resnet18 \
      --epochs 100 --eps_px 8 --ckpt_dir runs/cifar10_resnet18/supervised/pgd_at_seed0 \
      --out runs/cifar10_resnet18/robustness/pgd_at_seed0.json --seed 0

  python adv_supervised.py --method trades --dataset cifar10 --beta 6.0 ...
"""
import argparse
import json
import os
import re
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from models import BACKBONES
from datasets import make_eval_sets, num_classes as ds_num_classes, image_size as ds_image_size
from phtopo.attacks import pgd_linf
from phtopo.robustness import run_robustness_suite


# ---------------------------------------------------------------------------
# Architecture-matched classifier (same CIFAR stem as the SSL encoder)
# ---------------------------------------------------------------------------
def make_classifier(backbone: str, n_classes: int, cifar_no_maxpool: bool = True) -> nn.Module:
    net = BACKBONES[backbone](weights=None)
    if cifar_no_maxpool:
        net.maxpool = nn.Identity()  # match SimCLR(cifar_no_maxpool=True)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


# ---------------------------------------------------------------------------
# Inner maximizations
# ---------------------------------------------------------------------------
def trades_kl_attack(model, x, eps, steps, alpha=None, random_start=True):
    """Find x_adv maximizing KL(p(x) || p(x_adv)) within the Linf ball -- the TRADES
    robustness inner-max. Model put in eval (frozen BN) during the attack."""
    if alpha is None:
        alpha = 2.5 * eps / max(1, steps)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logp_clean = F.log_softmax(model(x), dim=1)
        p_clean = logp_clean.exp()
    x0 = x.detach()
    x_adv = x0 + (torch.empty_like(x0).uniform_(-eps, eps) if random_start else 0.0)
    x_adv = x_adv.clamp(0, 1)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logp_adv = F.log_softmax(model(x_adv), dim=1)
        kl = F.kl_div(logp_adv, p_clean, reduction="batchmean")  # KL(p_clean || p_adv)
        grad = torch.autograd.grad(kl, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(0, 1)
    if was_training:
        model.train()
    return x_adv.detach()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def at_train_transform(dataset: str):
    sz = ds_image_size(dataset)
    pad = max(2, sz // 8)  # 4 for CIFAR(32), 12 for STL(96): standard crop padding
    return transforms.Compose([
        transforms.RandomCrop(sz, padding=pad),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # [0,1]
    ])


def _latest_ckpt(ckpt_dir):
    best_e, best_p = 0, None
    for p in glob.glob(os.path.join(ckpt_dir, "epoch*.pt")):
        m = re.search(r"epoch(\d+)\.pt", os.path.basename(p))
        if m and int(m.group(1)) > best_e:
            best_e, best_p = int(m.group(1)), p
    return best_e, best_p


def train(args, device):
    nc = ds_num_classes(args.dataset)
    train_set, _ = make_eval_sets(args.dataset, root=args.data_dir, download=True)
    train_set.transform = at_train_transform(args.dataset)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)

    model = make_classifier(args.backbone, nc).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    eps = args.eps_px / 255.0
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    if args.resume:
        e, p = _latest_ckpt(args.ckpt_dir)
        if p is not None and e >= args.epochs:
            print(f"[adv_sup] already complete at epoch {e}")
            return os.path.join(args.ckpt_dir, f"epoch{args.epochs}.pt")
        if p is not None:
            ck = torch.load(p, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"]); start_epoch = ck["epoch"] + 1
            print(f"[adv_sup] resume from epoch {ck['epoch']} -> {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if args.method == "pgd_at":
                x_adv = pgd_linf(model, x, y, eps=eps, steps=args.steps, restarts=1)
                model.train()
                opt.zero_grad(set_to_none=True)
                logits = model(x_adv)
                loss = F.cross_entropy(logits, y)
            elif args.method == "trades":
                x_adv = trades_kl_attack(model, x, eps=eps, steps=args.steps)
                model.train()
                opt.zero_grad(set_to_none=True)
                logits = model(x)                       # natural logits
                logp_adv = F.log_softmax(model(x_adv), dim=1)
                p_clean = F.softmax(logits, dim=1)
                loss = F.cross_entropy(logits, y) + args.beta * F.kl_div(
                    logp_adv, p_clean, reduction="batchmean")
            else:
                raise ValueError(args.method)
            loss.backward(); opt.step()
            loss_sum += float(loss) * x.size(0); tot += x.size(0)
            correct += int((logits.argmax(1) == y).sum())
        sched.step()
        print(f"[adv_sup] {args.method} epoch {epoch}/{args.epochs} "
              f"loss={loss_sum/max(1,tot):.4f} train_acc={correct/max(1,tot):.4f}")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "epoch": epoch,
                        "args": vars(args)},
                       os.path.join(args.ckpt_dir, f"epoch{epoch}.pt"))
    return os.path.join(args.ckpt_dir, f"epoch{args.epochs}.pt")


def evaluate(args, device, ckpt_path):
    nc = ds_num_classes(args.dataset)
    _, test_set = make_eval_sets(args.dataset, root=args.data_dir, download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = make_classifier(args.backbone, nc).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
    model.eval()
    res = run_robustness_suite(model, test_loader, device, eps=args.eps_px / 255.0,
                               run_autoattack=not args.no_autoattack, max_batches=args.max_test_batches)
    res["method"] = args.method
    res["ckpt"] = ckpt_path
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"[adv_sup] wrote {args.out}")
    print("  clean      :", round(res["clean_acc"], 4))
    print("  MASKING    :", res["masking"]["verdict"], "| robust =", res["masking"]["robust_acc_headline"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pgd_at", "trades"])
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--eps_px", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--beta", type=float, default=6.0, help="TRADES robustness weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out", required=True, help="robustness JSON (same format as eval_robustness)")
    ap.add_argument("--no_autoattack", action="store_true")
    ap.add_argument("--max_test_batches", type=int, default=-1)
    ap.add_argument("--train_only", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[adv_sup] method={args.method} dataset={args.dataset} backbone={args.backbone} device={device}")

    ckpt_path = train(args, device)
    if not args.train_only:
        evaluate(args, device, ckpt_path)


if __name__ == "__main__":
    main()
