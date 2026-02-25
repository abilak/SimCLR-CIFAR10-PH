# simclr_lin.py
#!/usr/bin/env python3
"""
simclr_lin.py

Linear evaluation on CIFAR-10 with optional PGD epsilon sweep.
Designed to work with Hydra run dirs like:
  logs/downstream/{method}/seed{seed}/upE{load_epoch}/

Run examples (single run):
  python simclr_lin.py backbone=resnet18 method=baseline seed=0 load_epoch=10 \
    hydra.run.dir=logs/downstream/baseline/seed0/upE10 \
    attack.enabled=true attack.sweep=true attack.eps_px=[0,2,4,6,8,10] \
    attack.pgd_steps=10 attack.pgd_alpha=-1.0 attack.pgd_random_start=true

Outputs (inside the hydra run dir):
  - logs/lin_history_{method}_{backbone}.csv
  - logs/eps_curve_{method}_{backbone}_seed{seed}_load{load_epoch}.json
  - visuals/eps_sweep_{method}_seed{seed}_load{load_epoch}.png
"""

import os
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18, resnet34
from tqdm import tqdm

from models import SimCLR

logger = logging.getLogger(__name__)


# -------------------------
# FS utils
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _clamp(x, lo=0.0, hi=1.0):
    return torch.clamp(x, lo, hi)


def _parse_eps_list(attack_cfg) -> List[float]:
    """
    Returns eps list (floats).
    Supports:
      - attack.eps_list: list of floats
      - attack.eps_px: list of ints interpreted as /255
      - attack.eps: single float fallback
    """
    eps_list = getattr(attack_cfg, "eps_list", None)
    if eps_list is not None:
        return [float(e) for e in eps_list]

    eps_px = getattr(attack_cfg, "eps_px", None)
    if eps_px is not None:
        return [float(e) / 255.0 for e in eps_px]

    eps = float(getattr(attack_cfg, "eps", 8 / 255))
    return [eps]


def resolve_upstream_ckpt(orig_cwd: str, method: str, seed: int, load_epoch: int, backbone: str) -> str:
    """
    Robustly locate upstream checkpoint even if directory layout differs.
    Tries common candidates, then falls back to recursive glob under repo root.
    """
    pat = f"simclr_{method}_{backbone}_epoch{load_epoch}_seed{seed}.pt"

    candidates = [
        Path(orig_cwd) / "checkpoints" / "upstream" / method / f"seed{seed}" / f"epoch{load_epoch}" / pat,
        Path(orig_cwd) / "checkpoints" / "upstream" / method / f"seed{seed}" / pat,
        Path(orig_cwd) / "checkpoints" / pat,
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    hits = sorted(Path(orig_cwd).glob(f"**/{pat}"), key=lambda x: str(x))
    if hits:
        return str(hits[0])

    raise FileNotFoundError(
        f"Could not find upstream checkpoint for pattern {pat} under repo root {orig_cwd}.\n"
        f"Tried: {candidates[:3]}"
    )


# -------------------------
# Attacks
# -------------------------
def pgd_attack(model, x, y, eps, alpha, steps, random_start=True, clamp_min=0.0, clamp_max=1.0):
    x0 = x.detach()
    if random_start:
        x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
        x_adv = _clamp(x_adv, clamp_min, clamp_max)
    else:
        x_adv = x0.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)

        model.zero_grad(set_to_none=True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)
            x_adv = _clamp(x_adv, clamp_min, clamp_max)

    return x_adv.detach()


# -------------------------
# Meters / LR schedule
# -------------------------
class AverageMeter:
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        v = float(val)
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def get_lr(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# -------------------------
# Logging
# -------------------------
class HistoryLogger:
    def __init__(self, out_dir: str, filename: str, viz_dir: str):
        self.out_dir = out_dir
        ensure_dir(out_dir)
        self.csv_path = os.path.join(out_dir, filename)
        self.viz_dir = viz_dir
        ensure_dir(viz_dir)
        self.rows = []
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "pgd_acc", "lr"])

    def log_epoch(self, epoch, train_loss, train_acc, test_loss, test_acc, pgd_acc, lr):
        self.rows.append((epoch, train_loss, train_acc, test_loss, test_acc, pgd_acc, lr))
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, train_loss, train_acc, test_loss, test_acc, pgd_acc, lr])

    def plot(self, tag: str):
        if not self.rows:
            return

        epochs = [r[0] for r in self.rows]
        train_losses = [r[1] for r in self.rows]
        train_accs = [r[2] for r in self.rows]
        test_losses = [r[3] for r in self.rows]
        test_accs = [r[4] for r in self.rows]
        pgd_accs = [r[5] for r in self.rows]

        plt.figure()
        plt.plot(epochs, train_losses, label="train")
        plt.plot(epochs, test_losses, label="test")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Linear Eval Loss ({tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f"lin_loss_{tag}.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(epochs, train_accs, label="train")
        plt.plot(epochs, test_accs, label="test")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"Linear Eval Accuracy ({tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f"lin_acc_{tag}.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(epochs, test_accs, label="clean test")
        plt.plot(epochs, pgd_accs, label="PGD (logged)")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"Robust Accuracy ({tag})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f"lin_robust_{tag}.png"), dpi=150)
        plt.close()


# -------------------------
# Linear eval model
# -------------------------
class LinearEvalModel(nn.Module):
    def __init__(self, simclr_model: nn.Module, feature_dim: int, n_classes: int):
        super().__init__()
        self.simclr = simclr_model
        self.fc = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        _, h, _ = self.simclr(x)
        return self.fc(h)


@torch.no_grad()
def _set_encoder_eval_and_freeze(simclr_model: nn.Module):
    simclr_model.eval()
    for p in simclr_model.parameters():
        p.requires_grad = False


def run_epoch(model, dataloader, epoch, device, optimizer=None, scheduler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")

    for x, y in tqdm(dataloader, desc=("Train" if is_train else "Test") + f" epoch {epoch}"):
        x = x.to(device, non_blocking=(device == "cuda"))
        y = y.to(device, non_blocking=(device == "cuda"))

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        acc = (logits.argmax(dim=1) == y).float().mean()
        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))

    return loss_meter.avg, acc_meter.avg


def eval_adv_metrics(model, dataloader, device, attack_cfg) -> Dict[str, Any]:
    model.eval()

    do_sweep = bool(getattr(attack_cfg, "sweep", False))
    eps_list = _parse_eps_list(attack_cfg)
    if not do_sweep:
        eps_list = [eps_list[-1]]

    steps = int(getattr(attack_cfg, "pgd_steps", 10))
    rs = bool(getattr(attack_cfg, "pgd_random_start", True))
    max_batches = int(getattr(attack_cfg, "max_test_batches", -1))

    # clean
    clean_correct, n = 0, 0
    for bidx, (x, y) in enumerate(dataloader):
        if max_batches > 0 and bidx >= max_batches:
            break
        x = x.to(device, non_blocking=(device == "cuda"))
        y = y.to(device, non_blocking=(device == "cuda"))
        with torch.no_grad():
            logits = model(x)
            clean_correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    out: Dict[str, Any] = {"adv_clean_acc": clean_correct / max(1, n)}

    # pgd sweep
    pgd_acc_by_eps: Dict[float, float] = {}
    for eps in eps_list:
        eps = float(eps)
        alpha = float(getattr(attack_cfg, "pgd_alpha", -1.0))
        if alpha <= 0:
            alpha = eps / float(max(1, steps))

        pgd_correct, n_p = 0, 0
        for bidx, (x, y) in enumerate(dataloader):
            if max_batches > 0 and bidx >= max_batches:
                break
            x = x.to(device, non_blocking=(device == "cuda"))
            y = y.to(device, non_blocking=(device == "cuda"))
            x_adv = pgd_attack(model, x, y, eps=eps, alpha=float(alpha), steps=steps, random_start=rs)
            with torch.no_grad():
                logits = model(x_adv)
                pgd_correct += (logits.argmax(1) == y).sum().item()
            n_p += x.size(0)

        pgd_acc_by_eps[eps] = pgd_correct / max(1, n_p)

    out["pgd_acc_by_eps"] = pgd_acc_by_eps
    out["adv_pgd_acc"] = pgd_acc_by_eps[max(pgd_acc_by_eps.keys())] if pgd_acc_by_eps else float("nan")
    return out


@hydra.main(version_base=None, config_path=".", config_name="simclr_config")
def finetune(args: DictConfig) -> None:
    logger.info("Config:\n" + OmegaConf.to_yaml(args))

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[SimCLR-LIN] using device = {device}")

    orig_cwd = hydra.utils.get_original_cwd()  # repo root
    run_dir = os.getcwd()                      # hydra run dir (e.g., logs/downstream/.../upE10)
    print(f"[SimCLR-LIN] hydra run dir = {run_dir}")

    # outputs inside run dir
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    viz_dir  = os.path.join(run_dir, "visuals")
    log_dir  = os.path.join(run_dir, "logs")
    ensure_dir(ckpt_dir); ensure_dir(viz_dir); ensure_dir(log_dir)

    hist = HistoryLogger(
        out_dir=log_dir,
        filename=f"lin_history_{args.method}_{args.backbone}.csv",
        viz_dir=viz_dir
    )

    # dataset
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_set = CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_set  = CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)

    # labeled subset
    n_classes = 10
    per_class = int(getattr(args.lin, "per_class", 10))
    idx_by_class = {c: [] for c in range(n_classes)}
    for i in range(len(train_set)):
        _, y = train_set[i]
        if len(idx_by_class[y]) < per_class:
            idx_by_class[y].append(i)
        if all(len(v) >= per_class for v in idx_by_class.values()):
            break
    indices = [i for c in range(n_classes) for i in idx_by_class[c]]
    sampler = SubsetRandomSampler(indices)

    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        drop_last=False,
        sampler=sampler,
        num_workers=int(args.workers),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
    )

    # backbone
    assert args.backbone in ["resnet18", "resnet34"]
    base_encoder = resnet18 if args.backbone == "resnet18" else resnet34

    pre_model = SimCLR(
        base_encoder_fn=base_encoder,
        projection_dim=int(args.projection_dim),
        proj_hidden_dim=int(args.model.proj_hidden_dim),
        reduce_channels=int(args.ph.reduce_channels),
        cifar_no_maxpool=True,
    ).to(device)

    # checkpoint path (robust)
    ckpt_path = getattr(args, "ckpt_path", None)
    if ckpt_path is not None:
        ckpt_path = hydra.utils.to_absolute_path(str(ckpt_path))
    else:
        ckpt_path = resolve_upstream_ckpt(
            orig_cwd=orig_cwd,
            method=str(args.method),
            seed=int(args.seed),
            load_epoch=int(args.load_epoch),
            backbone=str(args.backbone),
        )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Upstream checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        pre_model.load_state_dict(ckpt["model"], strict=True)
    else:
        pre_model.load_state_dict(ckpt, strict=True)

    _set_encoder_eval_and_freeze(pre_model)

    feature_dim = int(pre_model.feature_dim)  # 512
    model = LinearEvalModel(pre_model, feature_dim=feature_dim, n_classes=n_classes).to(device)

    # optimizer for linear head
    params = [p for p in model.parameters() if p.requires_grad]
    lin_lr = 0.1 * float(args.batch_size) / 256.0
    optimizer = torch.optim.SGD(
        params,
        lr=lin_lr,
        momentum=float(args.momentum),
        weight_decay=0.0,
        nesterov=True,
    )

    finetune_epochs = int(args.finetune_epochs)
    total_steps = finetune_epochs * len(train_loader)
    if total_steps <= 0:
        raise ValueError(f"Linear eval has 0 training steps. subset={len(indices)}, batch_size={int(args.batch_size)}")

    lr_max = lin_lr
    lr_min = float(args.optim.lr_min)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step, total_steps, lr_max, lr_min) / lr_max)

    attack_cfg = getattr(args, "attack", None)
    do_attack = (attack_cfg is not None) and bool(getattr(attack_cfg, "enabled", False))

    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, finetune_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, epoch, device, optimizer, scheduler)
        test_loss, test_acc = run_epoch(model, test_loader, epoch, device)

        pgd_acc_logged = float("nan")

        # run epsilon sweep once at final epoch
        if do_attack and (epoch == finetune_epochs):
            adv = eval_adv_metrics(model, test_loader, device, attack_cfg)
            pgd_acc_logged = float(adv["adv_pgd_acc"])

            eps_curve_path = os.path.join(
                log_dir,
                f"eps_curve_{args.method}_{args.backbone}_seed{int(args.seed)}_load{int(args.load_epoch)}.json"
            )
            payload = {
                "method": str(args.method),
                "backbone": str(args.backbone),
                "seed": int(args.seed),
                "load_epoch": int(args.load_epoch),
                "lin_epoch": int(epoch),
                "attack": {
                    "sweep": bool(getattr(attack_cfg, "sweep", False)),
                    "eps_list": sorted([float(e) for e in adv["pgd_acc_by_eps"].keys()]),
                    "pgd_steps": int(getattr(attack_cfg, "pgd_steps", 10)),
                    "pgd_alpha": float(getattr(attack_cfg, "pgd_alpha", -1.0)),
                    "pgd_random_start": bool(getattr(attack_cfg, "pgd_random_start", True)),
                },
                "adv_clean_acc": float(adv["adv_clean_acc"]),
                "pgd_acc_by_eps": {str(float(k)): float(v) for k, v in adv["pgd_acc_by_eps"].items()},
            }
            with open(eps_curve_path, "w") as f:
                json.dump(payload, f, indent=2)

            eps_vals = sorted(adv["pgd_acc_by_eps"].keys())
            acc_vals = [adv["pgd_acc_by_eps"][e] for e in eps_vals]
            plt.figure()
            plt.plot(eps_vals, acc_vals, marker="o")
            plt.xlabel(r"$\epsilon$ ($\ell_\infty$)")
            plt.ylabel("PGD robust accuracy")
            plt.title(f"Epsilon Sweep ({args.method}, seed={args.seed}, up_epoch={int(args.load_epoch)})")
            plt.tight_layout()
            plt.savefig(
                os.path.join(viz_dir, f"eps_sweep_{args.method}_seed{int(args.seed)}_load{int(args.load_epoch)}.png"),
                dpi=150,
            )
            plt.close()

        current_lr = optimizer.param_groups[0]["lr"]
        tag = f"{args.method}_{args.backbone}_seed{int(args.seed)}_upE{int(args.load_epoch)}"
        hist.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc, pgd_acc_logged, current_lr)
        hist.plot(tag)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_path = os.path.join(ckpt_dir, f"simclr_lin_{tag}_best.pth")
            try:
                torch.save(model.state_dict(), best_path)
            except Exception as e:
                logger.warning(f"[WARN] Failed to save best linear head: {e}")

    logger.info(f"Best Test Acc: {best_test_acc:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    finetune()