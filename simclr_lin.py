"""
simclr_lin.py

Run examples:
  python simclr_lin.py backbone=resnet18 method=baseline load_epoch=10
  python simclr_lin.py backbone=resnet18 method=phsim    load_epoch=10
  python simclr_lin.py backbone=resnet18 method=hybrid   load_epoch=10
"""

import os
import csv
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import numpy as np

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
# Utilities
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

class HistoryLogger:
    def __init__(self, out_dir: str, filename: str = "lin_history.csv", viz_dir: str = None):
        self.out_dir = out_dir
        ensure_dir(out_dir)
        self.csv_path = os.path.join(out_dir, filename)
        self.viz_dir = viz_dir
        self.rows = []
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "pgd_acc", "lr"])

    def log_epoch(self, epoch, train_loss, train_acc, test_loss, test_acc, pgd_acc, lr):
        self.rows.append((epoch, train_loss, train_acc, test_loss, test_acc, pgd_acc, lr))
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, train_loss, train_acc, test_loss, test_acc, pgd_acc, lr])

    def plot(self, tag):
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
        plt.savefig(os.path.join(self.viz_dir, f"lin_loss_{tag}.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(epochs, train_accs, label="train")
        plt.plot(epochs, test_accs, label="test")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"Linear Eval Accuracy ({tag})")
        plt.legend()
        plt.savefig(os.path.join(self.viz_dir, f"lin_acc_{tag}.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(epochs, test_accs, label="clean test")
        plt.plot(epochs, pgd_accs, label="PGD-10 test")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"Robust Accuracy ({tag})")
        plt.legend()
        plt.savefig(os.path.join(self.viz_dir, f"lin_robust_{tag}.png"), dpi=150)
        plt.close()

def _clamp(x, lo=0.0, hi=1.0):
    return torch.clamp(x, lo, hi)

# FGSM and PGD implementations adapted for adversarial attack evaluation 
def fgsm_attack(model, x, y, eps, clamp_min=0.0, clamp_max=1.0):
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    model.zero_grad(set_to_none=True)
    if x_adv.grad is not None:
        x_adv.grad.zero_()
    loss.backward()
    with torch.no_grad():
        x_adv = x_adv + eps * x_adv.grad.sign()
        x_adv = _clamp(x_adv, clamp_min, clamp_max)
    return x_adv.detach()

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
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
    """Cosine annealing schedule (returns absolute LR)."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# -------------------------
# Linear eval head on pooled features h
# -------------------------
class LinearEvalModel(nn.Module):
    """
    Wraps the pretrained SimCLR model and trains a linear classifier on top of pooled feature h.

    SimCLR.forward(x) returns: (h_map_small, h, rep)
    We use h as feature for linear eval.

    Only self.fc is trainable.
    """
    def __init__(self, simclr_model: nn.Module, feature_dim: int, n_classes: int):
        super().__init__()
        self.simclr = simclr_model
        self.fc = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        # We only want pooled features h
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
    loader_bar = tqdm(dataloader)

    for x, y in loader_bar:
        x = x.to(device, non_blocking=(device == "cuda"))
        y = y.to(device, non_blocking=(device == "cuda"))

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        acc = (logits.argmax(dim=1) == y).float().mean()

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc.item(), x.size(0))

        phase = "Train" if is_train else "Test"
        loader_bar.set_description(
            f"{phase} epoch {epoch} | loss {loss_meter.avg:.4f} | acc {acc_meter.avg:.4f}"
        )

    return loss_meter.avg, acc_meter.avg

def eval_adv_metrics(model, dataloader, device, attack_cfg):
    model.eval()
    eps = float(getattr(attack_cfg, "eps", 8/255))
    do_fgsm = bool(getattr(attack_cfg, "fgsm", True))
    do_pgd = bool(getattr(attack_cfg, "pgd", True))
    steps = int(getattr(attack_cfg, "pgd_steps", 10))
    alpha = float(getattr(attack_cfg, "pgd_alpha", 2/255))
    rs = bool(getattr(attack_cfg, "pgd_random_start", True))
    max_batches = int(getattr(attack_cfg, "max_test_batches", -1))

    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0
    n = 0

    pbar = tqdm(dataloader, desc="AdvEval", leave=False)
    for bidx, (x, y) in enumerate(pbar):
        if max_batches > 0 and bidx >= max_batches:
            break

        x = x.to(device, non_blocking=(device == "cuda"))
        y = y.to(device, non_blocking=(device == "cuda"))
        bs = x.size(0)
        n += bs

        with torch.no_grad():
            logits = model(x)
            clean_correct += (logits.argmax(1) == y).sum().item()

        if do_fgsm:
            x_f = fgsm_attack(model, x, y, eps=eps)
            with torch.no_grad():
                logits_f = model(x_f)
                fgsm_correct += (logits_f.argmax(1) == y).sum().item()

        if do_pgd:
            x_p = pgd_attack(model, x, y, eps=eps, alpha=alpha, steps=steps, random_start=rs)
            with torch.no_grad():
                logits_p = model(x_p)
                pgd_correct += (logits_p.argmax(1) == y).sum().item()

        msg = f"clean {clean_correct/n:.3f}"
        if do_fgsm:
            msg += f" | fgsm {fgsm_correct/n:.3f}"
        if do_pgd:
            msg += f" | pgd {pgd_correct/n:.3f}"
        pbar.set_postfix_str(msg)

    out = {"adv_clean_acc": clean_correct / max(1, n)}
    out["adv_fgsm_acc"] = fgsm_correct / max(1, n) if do_fgsm else float("nan")
    out["adv_pgd_acc"]  = pgd_correct / max(1, n) if do_pgd else float("nan")
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

    # Hydra run dir for outputs
    out_dir = os.getcwd()

    ckpt_dir = os.path.join(out_dir, "checkpoints", "downstream", args.method, f"seed{args.seed}")
    viz_dir  = os.path.join(out_dir, "visuals",     "downstream", args.method, f"seed{args.seed}")
    log_dir  = os.path.join(out_dir, "logs",        "downstream", args.method, f"seed{args.seed}")
    ensure_dir(ckpt_dir)
    ensure_dir(viz_dir)
    ensure_dir(log_dir)

    hist = HistoryLogger(log_dir, filename=f"lin_history_{args.method}_{args.backbone}.csv", viz_dir=viz_dir)

    # Simple transforms for linear eval (common practice: light aug on train, standard on test)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_set = CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)

    # labeled subset: 10 per class (like your original)
    n_classes = 10
    rng = np.random.default_rng(seed=int(getattr(args, "seed", 0) or 0))
    indices = rng.choice(len(train_set), 10 * n_classes, replace=False)
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

    # Build the same SimCLR backbone as training
    assert args.backbone in ["resnet18", "resnet34"]
    base_encoder = resnet18 if args.backbone == "resnet18" else resnet34

    pre_model = SimCLR(
        base_encoder_fn=base_encoder,
        projection_dim=int(args.projection_dim),
        proj_hidden_dim=int(args.model.proj_hidden_dim),
        reduce_channels=int(args.ph.reduce_channels),
        cifar_no_maxpool=True,  # must match simclr.py
    ).to(device)

    # Load checkpoint produced by simclr.py
    ckpt_path = f"checkpoints/upstream/{args.method}/seed{args.seed}/epoch{int(args.load_epoch)}/simclr_{args.method}_{args.backbone}_epoch{int(args.load_epoch)}_seed{int(args.seed)}.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Make sure you're running from the directory containing the checkpoint, "
            f"or pass an absolute path / adjust naming."
        )

    ckpt = torch.load(ckpt_path, map_location=device)

    # simclr.py saves ckpt["model"] = model.state_dict()
    if isinstance(ckpt, dict) and "model" in ckpt:
        pre_model.load_state_dict(ckpt["model"], strict=True)
    else:
        pre_model.load_state_dict(ckpt, strict=True)

    # Freeze encoder (SimCLR model)
    _set_encoder_eval_and_freeze(pre_model)

    # Linear eval model on pooled features h
    feature_dim = int(pre_model.feature_dim)  # 512 for resnet18/34
    model = LinearEvalModel(pre_model, feature_dim=feature_dim, n_classes=n_classes).to(device)

    # Optimizer only for linear layer
    parameters = [p for p in model.parameters() if p.requires_grad]
    lin_lr = 0.1 * float(args.batch_size) / 256.0  # SimCLR paper heuristic

    optimizer = torch.optim.SGD(
        parameters,
        lr=lin_lr,
        momentum=float(args.momentum),
        weight_decay=0.0,
        nesterov=True,
    )

    # Cosine annealing LR for finetuning epochs
    finetune_epochs = int(args.finetune_epochs)
    total_steps = finetune_epochs * len(train_loader)
    if total_steps <= 0:
        raise ValueError(
            f"Linear eval has 0 training steps. "
            f"Check labeled subset size vs batch_size. "
            f"(subset={len(indices)}, batch_size={int(args.batch_size)}, drop_last={False})"
        )
    lr_max = lin_lr
    lr_min = float(args.optim.lr_min)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(step, total_steps, lr_max, lr_min) / lr_max
    )

    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, finetune_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, epoch, device, optimizer, scheduler)
        test_loss, test_acc = run_epoch(model, test_loader, epoch, device)

        attack_cfg = getattr(args, "attack", None)
        pgd_acc = float("nan")
        if attack_cfg is not None and bool(getattr(attack_cfg, "enabled", False)):
            adv = eval_adv_metrics(model, test_loader, device, attack_cfg)
            pgd_acc = adv["adv_pgd_acc"]

        current_lr = optimizer.param_groups[0]["lr"]
        tag = f"{args.method}_{args.backbone}_bs{args.batch_size}_seed{args.seed}"

        hist.log_epoch(epoch, train_loss, train_acc, test_loss, test_acc, pgd_acc, current_lr)
        hist.plot(tag=tag)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            logger.info("==> New best test acc")
            best_name = f"simclr_lin_{tag}_best.pth"
            best_path = os.path.join(ckpt_dir, best_name)
            torch.save(model.state_dict(), best_path)

    logger.info(f"Best Test Acc: {best_test_acc:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    finetune()