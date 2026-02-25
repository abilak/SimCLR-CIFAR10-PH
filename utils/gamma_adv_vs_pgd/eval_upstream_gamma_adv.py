"""
eval_upstream_gamma_adv.py

Compute upstream adversarial gamma (Gamma_adv) for saved SimCLR checkpoints.

Attack objective (label-free):
  maximize representation distortion in pooled feature space h:
    loss = mean(1 - cos(h(x_adv), h(x_clean)))

Then compute Gamma_adv using the same class-separation PH proxy as eval_gamma_class_separation,
but on adversarial features h(x_adv) instead of clean h(x).

Example:
  python eval_upstream_gamma_adv.py \
    --ckpt checkpoints/upstream/phsim/seed0/epoch10/simclr_phsim_resnet18_epoch10_seed0.pt \
    --data_dir data \
    --eps_px 8 --steps 5 --alpha_px 2 \
    --per_class 50 --batch_size 256 \
    --out_csv logs/upstream/phsim/seed0/phsim_seed0_gamma_adv.csv
"""

import os
import csv
import argparse
import itertools
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from persim import wasserstein
from ripser import ripser
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18, resnet34

from omegaconf import OmegaConf
from models import SimCLR


def _sanitize_dgm_np(dgm: np.ndarray) -> np.ndarray:
    if dgm is None or dgm.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    dgm = dgm.astype(np.float32, copy=False)
    mask = np.isfinite(dgm).all(axis=1)
    dgm = dgm[mask]
    if dgm.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pers = dgm[:, 1] - dgm[:, 0]
    dgm = dgm[pers > 1e-8]
    if dgm.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return dgm


def _standardize_np(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    x = x / (x.std(axis=0, keepdims=True) + 1e-6)
    return x.astype(np.float32)


@torch.no_grad()
def _select_balanced_indices(test_set: CIFAR10, per_class: int = 50) -> list:
    idx_by_class = {c: [] for c in range(10)}
    for idx in range(len(test_set)):
        _, y = test_set[idx]
        if len(idx_by_class[y]) < per_class:
            idx_by_class[y].append(idx)
        if all(len(v) >= per_class for v in idx_by_class.values()):
            break
    return [i for c in range(10) for i in idx_by_class[c]]


def pgd_rep_attack(
    model: nn.Module,
    x: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    """
    Label-free upstream PGD:
      maximize 1 - cosine_similarity(h_adv, h_clean)

    Assumes x in [0,1] and uses l_inf PGD with projection + clamp.
    """
    model.eval()

    x0 = x.detach()
    x_adv = x0.clone().detach()
    x_adv.requires_grad_(True)

    # Get clean pooled feature once (detached target)
    with torch.no_grad():
        _, h_clean, _ = model(x0)
        h_clean = h_clean.detach()

    for _ in range(int(steps)):
        _, h_adv, _ = model(x_adv)

        cos = F.cosine_similarity(h_adv, h_clean, dim=1)  # (B,)
        loss = (1.0 - cos).mean()  # ascend this

        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        x_adv = x_adv + alpha * grad.sign()

        # project to l_inf ball around x0
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = x_adv.detach()
        x_adv.requires_grad_(True)

    return x_adv.detach()


@torch.no_grad()
def eval_gamma_class_separation_on_features(
    feats: Dict[int, np.ndarray],
    w_h0: float = 0.2,
    w_h1: float = 1.0,
    maxdim: int = 1,
) -> float:
    """
    Same as your eval_gamma_class_separation, but consumes pre-collected per-class features.
    feats[c] should be shape (per_class, D) for c in 0..9.
    """
    dgms0, dgms1 = {}, {}
    for c in range(10):
        Xc = feats[c]
        Xc = _standardize_np(Xc)
        dgms = ripser(Xc, maxdim=maxdim)["dgms"]
        d0 = _sanitize_dgm_np(dgms[0])
        d1 = _sanitize_dgm_np(dgms[1]) if len(dgms) > 1 else np.zeros((0, 2), dtype=np.float32)
        dgms0[c] = d0
        dgms1[c] = d1

    dists = []
    for a, b in itertools.combinations(range(10), 2):
        d0 = wasserstein(dgms0[a], dgms0[b], matching=False)
        d1 = wasserstein(dgms1[a], dgms1[b], matching=False)
        d = float(w_h0) * float(d0) + float(w_h1) * float(d1)
        dists.append(d)

    return float(np.mean(dists)) if dists else 0.0


def build_model_from_ckpt_config(cfg: Dict[str, Any]) -> SimCLR:
    backbone_name = str(cfg.get("backbone", "resnet18"))
    base_encoder = resnet18 if backbone_name == "resnet18" else resnet34

    projection_dim = int(cfg.get("projection_dim", 64))
    model_cfg = cfg.get("model", {}) or {}
    ph_cfg = cfg.get("ph", {}) or {}

    proj_hidden_dim = int(model_cfg.get("proj_hidden_dim", 512))
    reduce_channels = int(ph_cfg.get("reduce_channels", 32))

    model = SimCLR(
        base_encoder,
        projection_dim=projection_dim,
        proj_hidden_dim=proj_hidden_dim,
        reduce_channels=reduce_channels,
        cifar_no_maxpool=True,
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)

    ap.add_argument("--eps_px", type=float, default=8.0)
    ap.add_argument("--alpha_px", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=5)

    ap.add_argument("--per_class", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--w_h0", type=float, default=0.2)
    ap.add_argument("--w_h1", type=float, default=1.0)
    ap.add_argument("--maxdim", type=int, default=1)

    ap.add_argument("--out_csv", type=str, default="gamma_adv.csv")
    args = ap.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    if not isinstance(cfg, dict):
        # Sometimes OmegaConf container might appear already resolved; keep it safe:
        cfg = OmegaConf.to_container(cfg, resolve=True)

    model = build_model_from_ckpt_config(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    eps = float(args.eps_px) / 255.0
    alpha = float(args.alpha_px) / 255.0

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = CIFAR10(root=args.data_dir, train=False, transform=test_transform, download=True)
    indices = _select_balanced_indices(test_set, per_class=int(args.per_class))
    subset = Subset(test_set, indices)
    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    # collect adversarial pooled features per class
    feats_adv = {c: [] for c in range(10)}

    for x, y in loader:
        x = x.to(device)
        y_np = y.numpy()

        x_adv = pgd_rep_attack(model, x, eps=eps, alpha=alpha, steps=int(args.steps))

        with torch.no_grad():
            _, h_adv, _ = model(x_adv)
        h_adv_np = h_adv.detach().cpu().numpy().astype(np.float32)

        for i, c in enumerate(y_np):
            feats_adv[int(c)].append(h_adv_np[i])

    # stack into arrays
    feats_adv_np = {c: np.stack(feats_adv[c], axis=0) for c in range(10)}
    gamma_adv = eval_gamma_class_separation_on_features(
        feats_adv_np,
        w_h0=float(args.w_h0),
        w_h1=float(args.w_h1),
        maxdim=int(args.maxdim),
    )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["ckpt", "epoch", "eps_px", "steps", "alpha_px", "gamma_adv"])
        w.writerow([
            args.ckpt,
            int(ckpt.get("epoch", -1)),
            float(args.eps_px),
            int(args.steps),
            float(args.alpha_px),
            float(gamma_adv),
        ])

    print(f"[OK] gamma_adv = {gamma_adv:.6f}")
    print(f"[OK] wrote: {args.out_csv}")


if __name__ == "__main__":
    main()