"""
simclr.py

Methods (yaml / CLI):
  - method=baseline : standard SimCLR NT-Xent on rep
  - method=phsim    : PH-guided contrastive (PH defines soft positives; reps learn them)
  - method=hybrid   : alpha*baseline + (1-alpha)*phsim

Run examples:
  python simclr.py backbone=resnet18 method=baseline
  python simclr.py backbone=resnet18 method=phsim
  python simclr.py backbone=resnet18 method=hybrid loss.alpha=0.9
"""

import os
import csv
import logging
import warnings
from typing import Optional

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34
from tqdm import tqdm

from models import SimCLR
from phtopo.losses import topo_separation_loss, raw_sw_separation_loss
from phtopo.descriptors import class_separation_gamma

logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", UserWarning)

# -------------------------
# Utilities
# -------------------------
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class HistoryLogger:
    def __init__(self, out_dir: str, filename: str = "train_history.csv"):
        self.out_dir = out_dir
        ensure_dir(out_dir)
        self.csv_path = os.path.join(out_dir, filename)
        self.rows = []
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss", "lr", "gamma"])

    def log_epoch(self, epoch: int, loss: float, lr: float, gamma: float = float("nan")):
        self.rows.append((epoch, loss, lr, gamma))
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, loss, lr, gamma])

    def plot(self, tag: str, out_dir: Optional[str] = None):
        """
        Saves 3 plots (loss, lr, gamma) into out_dir (defaults to self.out_dir).
        Each plot covers the full logged history so far.
        """
        if not self.rows:
            return
        out_dir = out_dir or self.out_dir
        ensure_dir(out_dir)

        epochs = [r[0] for r in self.rows]
        losses = [r[1] for r in self.rows]
        lrs = [r[2] for r in self.rows]
        gammas = [r[3] for r in self.rows]

        plt.figure()
        plt.plot(epochs, losses)
        plt.xlabel("epoch")
        plt.ylabel("train loss")
        plt.title(f"Train Loss ({tag})")
        plt.savefig(os.path.join(out_dir, f"loss_{tag}.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(epochs, lrs)
        plt.xlabel("epoch")
        plt.ylabel("learning rate")
        plt.title(f"Learning Rate ({tag})")
        plt.savefig(os.path.join(out_dir, f"lr_{tag}.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(epochs, gammas)
        plt.xlabel("epoch")
        plt.ylabel("Gamma (PH separation)")
        plt.title(f"Topological Separation Γ vs Epoch ({tag})")
        plt.savefig(os.path.join(out_dir, f"gamma_{tag}.png"), dpi=150)
        plt.close()


@torch.no_grad()
def eval_gamma_class_separation(
    model: nn.Module,
    device: str,
    data_dir: str,
    per_class: int = 50,
    batch_size: int = 256,
    w_h0: float = 0.2,
    w_h1: float = 1.0,
    maxdim: int = 1,
) -> float:
    """
    Evaluation-only proxy for Γ(f):
    - Build a small labeled set from CIFAR10 test split (per_class examples per class).
    - Compute pooled features h for each example.
    - For each class, compute persistence diagrams (H0/H1) on the class point cloud in feature space.
    - Return average weighted Wasserstein distance over all class pairs.
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)

    idx_by_class = {c: [] for c in range(10)}
    for idx in range(len(test_set)):
        _, y = test_set[idx]
        if len(idx_by_class[y]) < per_class:
            idx_by_class[y].append(idx)
        if all(len(v) >= per_class for v in idx_by_class.values()):
            break

    indices = [i for c in range(10) for i in idx_by_class[c]]
    subset = Subset(test_set, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

    feats = {c: [] for c in range(10)}
    model.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.numpy()
        _, h, _ = model(x)  # pooled backbone feature h
        h_np = h.detach().cpu().numpy().astype(np.float32)
        for i, c in enumerate(y):
            feats[int(c)].append(h_np[i])

    feats_by_class = {c: np.stack(feats[c], axis=0) for c in range(10) if len(feats[c]) > 0}
    # Fast GUDHI (H0/H1) + sliced-Wasserstein; same SW functional as training.
    return class_separation_gamma(
        feats_by_class, w_h0=float(w_h0), w_h1=float(w_h1), maxdim=maxdim
    )


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    """Cosine annealing schedule: returns ABSOLUTE lr."""
    if total_steps <= 1:
        return lr_min
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def get_color_distortion(s=0.5):
    """Color distortion = color jitter + grayscale (SimCLR appendix)."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    return transforms.Compose([rnd_color_jitter, rnd_gray])


# -------------------------
# Dataloader
# -------------------------
class CIFAR10Pair(CIFAR10):
    """Generate mini-batch pairs on CIFAR10 training set."""
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # (2,C,H,W), y


# -------------------------
# Losses
# -------------------------
def nt_xent(x: torch.Tensor, t=0.5) -> torch.Tensor:
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    sim = (x @ x.t()).clamp(min=1e-7)
    sim = sim / t
    sim = sim - torch.eye(sim.size(0), device=sim.device) * 1e5

    targets = torch.arange(sim.size(0), device=sim.device)
    targets[::2] += 1
    targets[1::2] -= 1
    return F.cross_entropy(sim, targets.long())


# Differentiable persistent-separation objectives live in phtopo.losses:
#   topo_separation_loss   -> method=phsim   (gradient flows through topology)
#   raw_sw_separation_loss -> method=swcontrol (non-topological control, Tier-1 #2)


# -------------------------
# Train
# -------------------------
@hydra.main(version_base=None, config_path=".", config_name="simclr_config")
def train(args: DictConfig) -> None:
    logger.info("Config:\n" + OmegaConf.to_yaml(args))

    device_cfg = str(getattr(args, "device", "auto")).lower()
    if device_cfg in ("cuda", "mps", "cpu"):
        device = device_cfg
    else:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"[SimCLR] using device = {device}")
    if device == "cuda":
        cudnn.benchmark = True

    seed = int(getattr(args, "seed", 0))
    set_seed(seed)

    # Hydra run dir (sweep scripts set hydra.run.dir)
    out_dir = os.getcwd()

    ckpt_dir = os.path.join(out_dir, "checkpoints", "upstream", str(args.method), f"seed{int(args.seed)}")
    ensure_dir(ckpt_dir)

    log_dir = os.path.join(out_dir, "logs", "upstream", str(args.method), f"seed{int(args.seed)}")
    ensure_dir(log_dir)

    viz_root = os.path.join(out_dir, "visuals", "upstream", str(args.method), f"seed{int(args.seed)}")
    ensure_dir(viz_root)

    hist = HistoryLogger(out_dir=log_dir, filename=f"{args.method}_seed{args.seed}_train_history.csv")

    # Data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        get_color_distortion(s=float(args.aug.color_strength)),
        transforms.ToTensor(),
    ])

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_set = CIFAR10Pair(root=data_dir, train=True, transform=train_transform, download=True)

    if int(args.data.subset_size) > 0:
        train_set = Subset(train_set, range(int(args.data.subset_size)))

    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True,
    )

    # Model
    assert args.backbone in ["resnet18", "resnet34"]
    base_encoder = resnet18 if args.backbone == "resnet18" else resnet34

    model = SimCLR(
        base_encoder,
        projection_dim=int(args.projection_dim),
        proj_hidden_dim=int(args.model.proj_hidden_dim),
        reduce_channels=int(args.ph.reduce_channels),
        cifar_no_maxpool=True,  # IMPORTANT for PH on CIFAR
    ).to(device)

    logger.info(f"Base model: {args.backbone}")
    logger.info(f"feature dim: {model.feature_dim}, projection dim: {args.projection_dim}")
    logger.info(f"method: {args.method}")

    # Optimizer
    optimizer = torch.optim.SGD(
        list(model.parameters()),
        float(args.learning_rate),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        nesterov=True,
    )

    # Scheduler
    max_steps = int(args.train.max_steps) if int(args.train.max_steps) > 0 else None
    steps_per_epoch = min(len(train_loader), max_steps) if max_steps is not None else len(train_loader)
    total_steps = int(args.epochs) * int(steps_per_epoch)

    lr_max = float(args.learning_rate)
    lr_min = float(args.optim.lr_min)

    def lr_mult(step: int) -> float:
        lr_abs = get_lr(step, total_steps, lr_max, lr_min)
        return lr_abs / max(lr_max, 1e-12)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_mult)

    model.train()

    temperature = float(args.temperature)
    tau_student = float(args.loss.student_temperature)

    # PH loss hyperparameters (differentiable separation objective)
    ph_num_points = int(getattr(args.ph, "num_points", 25))
    ph_neg_k = int(getattr(args.ph, "neg_k", 4))
    ph_margin = float(getattr(args.ph, "margin", 1.0))
    ph_ndir = int(getattr(args.ph, "n_directions", 32))
    ph_lambda = float(getattr(args.ph, "ph_lambda", 1.0))  # weight when combined with nt_xent

    warmup_epochs = int(getattr(args.train, "warmup_epochs", 0))

    # -------------------------
    # Early stopping (gamma-based)
    # -------------------------
    early_stop = bool(getattr(args.train, "early_stop", False))

    # IMPORTANT: default eval_every == log_interval so gamma aligns with checkpoint epochs
    save_every = int(args.log_interval)
    eval_every = int(getattr(args.train, "eval_every", save_every))
    if eval_every <= 0:
        eval_every = save_every

    es_patience = int(getattr(args.train, "es_patience", 4))
    es_min_delta = float(getattr(args.train, "es_min_delta", 1e-3))
    es_warmup = int(getattr(args.train, "es_warmup_epochs", 0))

    best_gamma = -1e18
    best_epoch = 0
    bad_count = 0

    tag = f"{args.method}_{args.backbone}_seed{args.seed}"

    for epoch in range(1, int(args.epochs) + 1):
        loss_meter = AverageMeter("loss")
        bar = tqdm(train_loader, total=steps_per_epoch)

        for step, (x, _) in enumerate(bar):
            if max_steps is not None and step >= max_steps:
                break

            B = x.size(0)
            x = x.view(B * 2, x.size(2), x.size(3), x.size(4)).to(
                device, non_blocking=(device == "cuda")
            )

            optimizer.zero_grad(set_to_none=True)
            h_map_small, _, rep = model(x)

            method = str(args.method).lower()
            ph_methods = ["phsim", "swcontrol", "hybrid"]
            if method in ph_methods and warmup_epochs > 0 and epoch <= warmup_epochs:
                method_eff = "baseline"
            else:
                method_eff = method

            sep_fn = topo_separation_loss if method_eff != "swcontrol" else raw_sw_separation_loss

            if method_eff == "baseline":
                loss = nt_xent(rep, temperature)

            elif method_eff in ("phsim", "swcontrol"):
                # Differentiable persistent-separation (or non-topological control)
                loss, _ = sep_fn(
                    h_map_small,
                    num_points=ph_num_points,
                    neg_k=ph_neg_k,
                    margin=ph_margin,
                    n_directions=ph_ndir,
                )

            elif method_eff == "hybrid":
                alpha = float(args.loss.alpha)
                loss_cos = nt_xent(rep, temperature)
                loss_ph, _ = topo_separation_loss(
                    h_map_small,
                    num_points=ph_num_points,
                    neg_k=ph_neg_k,
                    margin=ph_margin,
                    n_directions=ph_ndir,
                )
                loss = alpha * loss_cos + (1.0 - alpha) * ph_lambda * loss_ph

            else:
                raise ValueError(f"Unknown method={args.method}. Use baseline|phsim|swcontrol|hybrid.")

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), n=x.size(0))
            bar.set_description(f"epoch {epoch} | loss {loss_meter.avg:.4f}")

        # -------------------------
        # Save checkpoint on schedule
        # -------------------------
        if epoch % save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "config": OmegaConf.to_container(args, resolve=True),
                "epoch": epoch,
            }
            ckpt_name = f"simclr_{args.method}_{args.backbone}_epoch{epoch}_seed{args.seed}.pt"
            epoch_ckpt_dir = os.path.join(ckpt_dir, f"epoch{epoch}")
            ensure_dir(epoch_ckpt_dir)
            ckpt_path = os.path.join(epoch_ckpt_dir, ckpt_name)
            logger.info(f"==> Save checkpoint: {ckpt_path}")
            torch.save(ckpt, ckpt_path)

        # -------------------------
        # End-of-epoch eval/log/plots
        # -------------------------
        current_lr = optimizer.param_groups[0]["lr"]

        # Evaluate gamma only on eval cadence (aligned to save cadence by default)
        do_eval = (epoch % eval_every == 0)

        gamma = float("nan")
        if do_eval:
            gamma = eval_gamma_class_separation(
                model=model,
                device=device,
                data_dir=data_dir,
                per_class=int(getattr(args.eval, "gamma_per_class", 50)),
                batch_size=int(getattr(args.eval, "gamma_batch_size", 256)),
                w_h0=float(getattr(args.ph, "w_h0", 0.2)),
                w_h1=float(getattr(args.ph, "w_h1", 1.0)),
                maxdim=1,
            )
            model.train()

        hist.log_epoch(epoch, loss_meter.avg, current_lr, gamma)

        # Keep plots in ONE place to avoid huge duplicate files.
        # (Still compatible with your sweep scripts—nothing depends on per-epoch plot folders.)
        hist.plot(tag=tag, out_dir=viz_root)

        # -------------------------
        # Early stopping (maximize gamma)
        # -------------------------
        if early_stop and do_eval and epoch > es_warmup:
            improved = (gamma > best_gamma + es_min_delta)
            if improved:
                best_gamma = gamma
                best_epoch = epoch
                bad_count = 0

                # Save best checkpoint in TWO ways:
                # (1) A best/ checkpoint for convenience
                best_dir = os.path.join(ckpt_dir, "best")
                ensure_dir(best_dir)
                best_path = os.path.join(
                    best_dir, f"simclr_{args.method}_{args.backbone}_best_seed{args.seed}.pt"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                                "config": OmegaConf.to_container(args, resolve=True),
                        "best_gamma": best_gamma,
                        "best_epoch": best_epoch,
                    },
                    best_path,
                )

                # (2) Also save a discoverable epoch-style checkpoint if this epoch
                # isn't already saved by save_every (so sweeps can find it).
                if epoch % save_every != 0:
                    ckpt = {
                        "model": model.state_dict(),
                                "config": OmegaConf.to_container(args, resolve=True),
                        "epoch": epoch,
                        "best_gamma": best_gamma,
                        "best_epoch": best_epoch,
                    }
                    ckpt_name = f"simclr_{args.method}_{args.backbone}_epoch{epoch}_seed{args.seed}.pt"
                    epoch_ckpt_dir = os.path.join(ckpt_dir, f"epoch{epoch}")
                    ensure_dir(epoch_ckpt_dir)
                    ckpt_path = os.path.join(epoch_ckpt_dir, ckpt_name)
                    logger.info(f"==> Save (best-epoch) checkpoint: {ckpt_path}")
                    torch.save(ckpt, ckpt_path)

            else:
                bad_count += 1
                if bad_count >= es_patience:
                    logger.info(
                        f"[EarlyStop] Stop at epoch={epoch}. best_gamma={best_gamma:.6f} (best_epoch={best_epoch})"
                    )
                    break


if __name__ == "__main__":
    train()