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
from torchvision.models import resnet18, resnet34, resnet50
from tqdm import tqdm

from models import SimCLR, BACKBONES
from datasets import make_ssl_trainset, make_eval_sets, num_classes as ds_num_classes
from phtopo.losses import (
    topo_separation_loss, raw_sw_separation_loss,
    topo_consistency_loss, raw_consistency_loss,
)
from phtopo.adv import pgd_ascent_on_loss, eval_mode
from phtopo.dual_bn import convert_to_dual_bn, bn_route, count_dual_bn, sync_adv_bn_from_clean
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
    dataset: str = "cifar10",
) -> float:
    """
    Evaluation-only proxy for Γ(f):
    - Build a small labeled set from CIFAR10 test split (per_class examples per class).
    - Compute pooled features h for each example.
    - For each class, compute persistence diagrams (H0/H1) on the class point cloud in feature space.
    - Return average weighted Wasserstein distance over all class pairs.
    """
    nc = ds_num_classes(dataset)
    _, test_set = make_eval_sets(dataset, root=data_dir, download=True)

    idx_by_class = {c: [] for c in range(nc)}
    for idx in range(len(test_set)):
        _, y = test_set[idx]
        if len(idx_by_class[y]) < per_class:
            idx_by_class[y].append(idx)
        if all(len(v) >= per_class for v in idx_by_class.values()):
            break

    indices = [i for c in range(nc) for i in idx_by_class[c]]
    subset = Subset(test_set, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

    feats = {c: [] for c in range(nc)}
    model.eval()
    for x, y in loader:
        x = x.to(device)
        y = y.numpy()
        _, h, _ = model(x)  # pooled backbone feature h
        h_np = h.detach().cpu().numpy().astype(np.float32)
        for i, c in enumerate(y):
            feats[int(c)].append(h_np[i])

    feats_by_class = {c: np.stack(feats[c], axis=0) for c in range(nc) if len(feats[c]) > 0}
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


# Method taxonomy
CLEAN_METHODS = {"baseline", "phsim", "swcontrol", "hybrid"}
ADV_WRAP_METHODS = {"adv_baseline", "adv_phsim", "adv_swcontrol"}  # adversarial SSL (ACL-style)
CONSIST_METHODS = {"topoacl", "rawacl"}                            # clean NT-Xent + adv consistency
ALL_METHODS = CLEAN_METHODS | ADV_WRAP_METHODS | CONSIST_METHODS


def _ph_forward(model, x, P):
    """Return (maps, rep): a LIST of PH feature maps (>1 if multiscale) + projection."""
    if P.get("multiscale"):
        maps, _, rep = model.ph_maps(x)
        return maps, rep
    h_map, _, rep = model(x)
    return [h_map], rep


def _ph_maps_only(model, x, P):
    """
    PH feature maps ONLY, skipping layer4 + avgpool + projector (which aren't in
    the PH-map graph). Numerically identical to `_ph_forward(model, x, P)[0]` but
    avoids the wasted deep-layer compute -- a meaningful speedup for the inner PGD
    of the separation/consistency methods, where this runs once per ascent step.
    """
    return model.ph_maps_only(x)


def _sep_loss_maps(core, maps, P):
    """Mean separation loss over the (multiscale) list of maps. Returns (loss, stats)."""
    fn = topo_separation_loss if core == "phsim" else raw_sw_separation_loss
    losses, st = [], {}
    for hm in maps:
        l, st = fn(hm, num_points=P["num_points"], neg_k=P["neg_k"], margin=P["margin"],
                   n_directions=P["ndir"], neg_agg=P["neg_agg"], softmin_temp=P["softmin_temp"])
        losses.append(l)
    return torch.stack(losses).mean(), st


def _consistency_maps(method, maps_clean, maps_adv, P):
    """Mean clean-vs-adv consistency over the (multiscale) list of map pairs."""
    assert len(maps_clean) == len(maps_adv), \
        f"multiscale map count mismatch: {len(maps_clean)} vs {len(maps_adv)}"
    cons_fn = topo_consistency_loss if method == "topoacl" else raw_consistency_loss
    losses, st = [], {}
    for hc, ha in zip(maps_clean, maps_adv):
        l, st = cons_fn(hc, ha, num_points=P["num_points"], n_directions=P["ndir"])
        losses.append(l)
    return torch.stack(losses).mean(), st


def compute_training_loss(model, x, method, P, teacher_model=None):
    """
    Compute the upstream training loss for a batch x (2B interleaved views) under
    any supported method, with optional multiscale PH (mean over network depths).

    Adversarial methods run an inner PGD maximization (encoder in eval mode so
    BatchNorm stats aren't corrupted by the inner pass), then a first-order outer
    step on the detached adversarial input.

    teacher_model: if given (EMA teacher), topoacl/rawacl distill against the
    teacher's stable clean topology (eval-mode, no grad) instead of the online
    model's own clean diagram -- a collapse-free target with no BN-mode mismatch.

    Dual-BN (P["dual_bn"]=True, AdvProp/AdvCL): clean inputs route through clean-BN
    and adversarial inputs through adv-BN. For adv_* methods the loss becomes
    clean_loss + adv_loss (both branches trained every step); for topoacl/rawacl
    the clean base trains clean-BN and the adversarial-consistency branch trains
    adv-BN. Eval/inference uses clean-BN (default route). Requires the model to have
    been converted with phtopo.dual_bn.convert_to_dual_bn (train() does this).
    Returns (loss, stats_dict).
    """
    method = method.lower()

    # ----- clean objectives -----
    if method == "baseline":
        _, _, rep = model(x)
        return nt_xent(rep, P["temperature"]), {}
    if method in ("phsim", "swcontrol"):
        maps = _ph_maps_only(model, x, P)
        return _sep_loss_maps(method, maps, P)
    if method == "hybrid":
        maps, rep = _ph_forward(model, x, P)
        loss_cos = nt_xent(rep, P["temperature"])
        loss_ph, st = _sep_loss_maps("phsim", maps, P)
        return P["alpha"] * loss_cos + (1.0 - P["alpha"]) * P["ph_lambda"] * loss_ph, st

    eps, alpha, steps = P["adv_eps"], P["adv_alpha"], P["adv_steps"]
    dual_bn = bool(P.get("dual_bn", False))

    # ----- adversarial SSL (ACL-style): maximize the SSL loss, then train on it -----
    if method in ADV_WRAP_METHODS:
        core = method[len("adv_"):]

        def closure(xp):
            if core == "baseline":
                _, _, rep = model(xp)
                return nt_xent(rep, P["temperature"])
            maps = _ph_maps_only(model, xp, P)
            return _sep_loss_maps(core, maps, P)[0]

        if dual_bn:
            # AdvProp / AdvCL: attack and consume adversarial inputs through the
            # adv-BN branch, and ALSO train the clean-BN branch on the clean inputs
            # in the same step (clean_loss + adv_loss). Both branches see data every
            # step, so neither branch's running stats go stale, and eval (clean
            # branch, default route) sees properly trained clean statistics.
            with eval_mode(model), bn_route("adv"):
                x_adv = pgd_ascent_on_loss(closure, x, eps, alpha, steps, P["adv_random_start"])
            with bn_route("clean"):
                l_clean = closure(x)          # trains clean_bn
            with bn_route("adv"):
                l_adv = closure(x_adv)        # trains adv_bn
            loss = l_clean + l_adv
            return loss, {"adv_loss": float(l_adv.detach()), "clean_loss": float(l_clean.detach())}

        # Single-BN: inner PGD in eval mode (BN frozen), outer train-mode step.
        with eval_mode(model):
            x_adv = pgd_ascent_on_loss(closure, x, eps, alpha, steps, P["adv_random_start"])
        loss = closure(x_adv)  # outer step, train-mode BN, on detached x_adv
        return loss, {"adv_loss": float(loss.detach())}

    # ----- topological adversarial consistency (flagship) + raw control -----
    if method in CONSIST_METHODS:
        if dual_bn:
            # The clean NT-Xent base trains clean-BN -- the branch used at
            # inference/eval -- on clean inputs only (clean-BN is never exposed to
            # adversarial-batch statistics). The adversarial CONSISTENCY is measured
            # ENTIRELY WITHIN the adv branch, in eval mode, so that the clean target
            # and the adversarial forward share the IDENTICAL normalization. The loss
            # then reflects ONLY perturbation-induced topology drift (exactly 0 when
            # x_adv == x), NOT the (large, learned) gap between the clean-BN and
            # adv-BN branches -- comparing clean-BN(x) to adv-BN(x_adv) would
            # contaminate the signal with that branch gap and perversely push the
            # two branches back together, defeating dual-BN. adv-BN's affine
            # parameters are trained by this consistency gradient.
            with bn_route("clean"):
                _, rep_clean = _ph_forward(model, x, P)
                base = nt_xent(rep_clean, P["temperature"])
            with eval_mode(model), bn_route("adv"):
                if teacher_model is not None:
                    # Stable EMA target, same (adv) branch & mode as the student's
                    # adversarial forward, so the only difference is the perturbation.
                    with torch.no_grad():
                        maps_ref = [m.detach() for m in _ph_maps_only(teacher_model, x, P)]
                else:
                    maps_ref = [m.detach() for m in _ph_maps_only(model, x, P)]
                def closure(xp):
                    maps_adv = _ph_maps_only(model, xp, P)
                    return _consistency_maps(method, maps_ref, maps_adv, P)[0]
                x_adv = pgd_ascent_on_loss(closure, x, eps, alpha, steps, P["adv_random_start"])
                maps_adv = _ph_maps_only(model, x_adv, P)   # eval-mode adv-BN; grad flows
                cons, st = _consistency_maps(method, maps_ref, maps_adv, P)
            loss = base + P["adv_beta"] * cons
            st = {**st, "base_ntxent": float(base.detach()), "consistency": float(cons.detach())}
            return loss, st

        maps_clean, rep_clean = _ph_forward(model, x, P)   # train-mode clean forward (grad kept)
        base = nt_xent(rep_clean, P["temperature"])         # keeps clean accuracy high

        if teacher_model is not None:
            # EMA-teacher self-distillation: stable clean topology target (eval, no grad).
            # Student's consistency forwards run in eval mode to match the teacher's
            # BN mode (no offset); BN is still trained by the clean contrastive base.
            with torch.no_grad():
                maps_target = [m.detach() for m in _ph_maps_only(teacher_model, x, P)]
            with eval_mode(model):
                def closure(xp):
                    maps_adv = _ph_maps_only(model, xp, P)
                    return _consistency_maps(method, maps_target, maps_adv, P)[0]
                x_adv = pgd_ascent_on_loss(closure, x, eps, alpha, steps, P["adv_random_start"])
                maps_adv = _ph_maps_only(model, x_adv, P)         # student eval-mode (grad flows)
                cons, st = _consistency_maps(method, maps_target, maps_adv, P)
        else:
            # Self-consistency: EVAL-mode clean reference for the inner ascent (else a
            # constant BN-mode offset pollutes the target); train-vs-train for the outer.
            with eval_mode(model):
                maps_clean_ref = [m.detach() for m in _ph_maps_only(model, x, P)]
                def closure(xp):
                    maps_adv = _ph_maps_only(model, xp, P)
                    return _consistency_maps(method, maps_clean_ref, maps_adv, P)[0]
                x_adv = pgd_ascent_on_loss(closure, x, eps, alpha, steps, P["adv_random_start"])
            maps_adv = _ph_maps_only(model, x_adv, P)
            cons, st = _consistency_maps(method, maps_clean, maps_adv, P)

        loss = base + P["adv_beta"] * cons
        st = {**st, "base_ntxent": float(base.detach()), "consistency": float(cons.detach())}
        return loss, st

    raise ValueError(f"Unknown method={method}. Use one of {sorted(ALL_METHODS)}.")


def _find_latest_ckpt(ckpt_dir, method, backbone, seed):
    """
    Return (epoch, path) of the highest-epoch checkpoint under ckpt_dir for this
    (method, backbone, seed), or (0, None) if none. Used to resume a run that was
    interrupted (crash / spot preemption) from its last saved epoch instead of
    restarting from scratch.
    """
    import glob, re
    pat = os.path.join(ckpt_dir, "epoch*", f"simclr_{method}_{backbone}_epoch*_seed{seed}.pt")
    best_e, best_p = 0, None
    for p in glob.glob(pat):
        mobj = re.search(r"epoch(\d+)_seed", os.path.basename(p))
        if mobj and int(mobj.group(1)) > best_e:
            best_e, best_p = int(mobj.group(1)), p
    return best_e, best_p


@torch.no_grad()
def ema_update(ema_model, model, momentum: float):
    """
    EMA update of teacher params; BN buffers (running stats) copied directly.

    Teacher and student must have identical structure (the teacher is a deepcopy
    of the student made AFTER any dual-BN conversion), so the positional zips align
    -- including each dual-BN branch's running_mean/var/num_batches_tracked. Assert
    equal counts so a future structural divergence fails loudly instead of silently
    truncating (zip stops at the shorter sequence).
    """
    ep_list, p_list = list(ema_model.parameters()), list(model.parameters())
    eb_list, b_list = list(ema_model.buffers()), list(model.buffers())
    assert len(ep_list) == len(p_list), f"EMA param count mismatch: {len(ep_list)} vs {len(p_list)}"
    assert len(eb_list) == len(b_list), f"EMA buffer count mismatch: {len(eb_list)} vs {len(b_list)}"
    for ep, p in zip(ep_list, p_list):
        ep.mul_(momentum).add_(p.detach(), alpha=1.0 - momentum)
    for eb, b in zip(eb_list, b_list):
        eb.copy_(b)


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

    # Data. The two-views SSL set is built per-dataset (crop sized to the dataset);
    # default cifar10 reproduces the original 32x32 pipeline. Attacks operate in
    # [0,1] (ToTensor only), unchanged across datasets.
    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    dataset_name = str(getattr(args, "dataset", "cifar10")).lower()
    train_set = make_ssl_trainset(
        dataset_name, root=data_dir,
        color_strength=float(args.aug.color_strength), download=True,
    )

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
    assert args.backbone in BACKBONES, f"backbone must be one of {sorted(BACKBONES)}"
    base_encoder = BACKBONES[args.backbone]

    ph_extra_layers = tuple(getattr(args.ph, "extra_layers", []) or [])
    model = SimCLR(
        base_encoder,
        projection_dim=int(args.projection_dim),
        proj_hidden_dim=int(args.model.proj_hidden_dim),
        reduce_channels=int(args.ph.reduce_channels),
        cifar_no_maxpool=True,  # IMPORTANT for PH on CIFAR
        ph_source_layer=str(getattr(args.ph, "source_layer", "layer3")),
        ph_extra_layers=ph_extra_layers,
    ).to(device)

    logger.info(f"Base model: {args.backbone}")
    logger.info(f"feature dim: {model.feature_dim}, projection dim: {args.projection_dim}")
    logger.info(f"method: {args.method}")

    # Dual-BN (AdvProp/AdvCL): only meaningful for adversarial methods, where clean
    # and adversarial inputs are routed through separate BN branches. Convert BEFORE
    # the optimizer (so adv-BN params are optimized) and BEFORE the EMA deepcopy (so
    # the teacher matches). Default off => zero change to the single-BN path.
    _adv_cfg0 = getattr(args, "adv", None)
    _method_lc0 = str(args.method).lower()
    _adv_method = _method_lc0 in (ADV_WRAP_METHODS | CONSIST_METHODS)
    use_dual_bn = (bool(getattr(_adv_cfg0, "dual_bn", False)) and _adv_method) if _adv_cfg0 is not None else False
    if use_dual_bn:
        convert_to_dual_bn(model)
        logger.info(f"Dual-BN enabled: converted {count_dual_bn(model)} BatchNorm2d -> DualBatchNorm2d")

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

    # Loss hyperparameters bundle (passed to compute_training_loss)
    adv_cfg = getattr(args, "adv", None)
    adv_eps = float(getattr(adv_cfg, "eps", 8.0 / 255.0)) if adv_cfg is not None else 8.0 / 255.0
    adv_steps = int(getattr(adv_cfg, "steps", 5)) if adv_cfg is not None else 5
    _adv_alpha_cfg = float(getattr(adv_cfg, "alpha", -1.0)) if adv_cfg is not None else -1.0
    adv_alpha = _adv_alpha_cfg if _adv_alpha_cfg > 0 else 2.5 * adv_eps / max(1, adv_steps)
    P = {
        "temperature": float(args.temperature),
        "num_points": int(getattr(args.ph, "num_points", 64)),
        "neg_k": int(getattr(args.ph, "neg_k", 4)),
        "margin": float(getattr(args.ph, "margin", 1.0)),
        "ndir": int(getattr(args.ph, "n_directions", 32)),
        "ph_lambda": float(getattr(args.ph, "ph_lambda", 1.0)),
        "neg_agg": str(getattr(args.ph, "neg_agg", "hard")),
        "softmin_temp": float(getattr(args.ph, "softmin_temp", 0.1)),
        "alpha": float(getattr(args.loss, "alpha", 0.9)),
        "multiscale": len(ph_extra_layers) > 0,
        "adv_eps": adv_eps,
        "adv_alpha": adv_alpha,
        "adv_steps": adv_steps,
        "adv_beta": float(getattr(adv_cfg, "beta", 1.0)) if adv_cfg is not None else 1.0,
        "adv_random_start": bool(getattr(adv_cfg, "random_start", True)) if adv_cfg is not None else True,
        "dual_bn": use_dual_bn,
    }

    warmup_epochs = int(getattr(args.train, "warmup_epochs", 0))

    # EMA teacher (optional, for topoacl/rawacl self-distillation under attack)
    import copy as _copy
    method_lc = str(args.method).lower()
    use_ema = (str(getattr(adv_cfg, "teacher", "none")).lower() == "ema"
               and method_lc in CONSIST_METHODS) if adv_cfg is not None else False
    ema_model = None
    if use_ema:
        ema_model = _copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()
        logger.info("EMA teacher enabled for topological self-distillation.")
    ema_momentum = float(getattr(adv_cfg, "ema_momentum", 0.996)) if adv_cfg is not None else 0.996

    # Adversarial curriculum: linearly ramp eps & beta over `ramp_epochs`.
    base_adv_eps, base_adv_beta = P["adv_eps"], P["adv_beta"]
    adv_ramp_epochs = int(getattr(adv_cfg, "ramp_epochs", 0)) if adv_cfg is not None else 0

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

    # -------------------------
    # Resume (crash / spot-preemption safe): continue from the latest checkpoint
    # for this (method, backbone, seed). Restores model, optimizer, scheduler, EMA
    # teacher, and RNG so the LR schedule and momentum buffers continue cleanly.
    # -------------------------
    start_epoch = 1
    if bool(getattr(args.train, "resume", True)):
        last_e, last_p = _find_latest_ckpt(ckpt_dir, args.method, args.backbone, args.seed)
        if last_p is not None and last_e >= int(args.epochs):
            logger.info(f"[resume] run already complete at epoch {last_e}; nothing to do.")
            return
        if last_p is not None:
            ck = torch.load(last_p, map_location=device, weights_only=False)  # our own ckpt (has cfg/RNG)
            load_state_dict_auto(model, ck["model"], strict=True)
            if ck.get("optimizer") is not None:
                optimizer.load_state_dict(ck["optimizer"])
            if ck.get("scheduler") is not None:
                scheduler.load_state_dict(ck["scheduler"])
            if ema_model is not None and ck.get("ema") is not None:
                load_state_dict_auto(ema_model, ck["ema"], strict=True)
            rng = ck.get("rng")
            if rng is not None:
                try:
                    torch.set_rng_state(rng["torch"])
                    np.random.set_state(rng["numpy"])
                    if device == "cuda" and rng.get("cuda") is not None:
                        torch.cuda.set_rng_state_all(rng["cuda"])
                except Exception as e:
                    logger.warning(f"[resume] could not restore RNG state: {e}")
            start_epoch = int(ck["epoch"]) + 1
            logger.info(f"[resume] from epoch {ck['epoch']} ({last_p}); continuing at epoch {start_epoch}")

    for epoch in range(start_epoch, int(args.epochs) + 1):
        loss_meter = AverageMeter("loss")
        bar = tqdm(train_loader, total=steps_per_epoch)

        # Dual-BN warm start: at the first adversarial epoch (right after a clean
        # warmup, during which only clean-BN was trained), copy the warmed-up
        # clean-BN stats into adv-BN so the adv branch doesn't begin from stale init.
        if use_dual_bn and warmup_epochs > 0 and epoch == warmup_epochs + 1:
            n_sync = sync_adv_bn_from_clean(model)
            if ema_model is not None:
                ema_update(ema_model, model, momentum=0.0)  # hard-sync teacher to the warm-started student
            logger.info(f"Dual-BN warm start: synced adv-BN from clean-BN ({n_sync} branches) at epoch {epoch}")

        # Adversarial curriculum: scale eps/alpha/beta by the ramp factor.
        if adv_ramp_epochs > 0:
            ramp = min(1.0, epoch / float(adv_ramp_epochs))
            P["adv_eps"] = base_adv_eps * ramp
            P["adv_alpha"] = 2.5 * P["adv_eps"] / max(1, P["adv_steps"])
            P["adv_beta"] = base_adv_beta * ramp

        for step, (x, _) in enumerate(bar):
            if max_steps is not None and step >= max_steps:
                break

            B = x.size(0)
            x = x.view(B * 2, x.size(2), x.size(3), x.size(4)).to(
                device, non_blocking=(device == "cuda")
            )

            optimizer.zero_grad(set_to_none=True)

            method = str(args.method).lower()
            # Warmup: train clean baseline for the first warmup_epochs (stabilizes
            # the encoder before topology/adversarial objectives kick in).
            if method != "baseline" and warmup_epochs > 0 and epoch <= warmup_epochs:
                method_eff = "baseline"
            else:
                method_eff = method

            loss, _ = compute_training_loss(model, x, method_eff, P,
                                            teacher_model=ema_model if method_eff in CONSIST_METHODS else None)

            loss.backward()
            optimizer.step()
            scheduler.step()
            if ema_model is not None:
                ema_update(ema_model, model, ema_momentum)

            loss_meter.update(loss.item(), n=x.size(0))
            bar.set_description(f"epoch {epoch} | loss {loss_meter.avg:.4f}")

        # -------------------------
        # Save checkpoint on schedule
        # -------------------------
        if epoch % save_every == 0:
            # Persist optimizer/scheduler/EMA/RNG too so a resumed run continues the
            # LR schedule + momentum buffers exactly. Eval loaders read only ck["model"],
            # so these extra keys are backward-compatible.
            ckpt = {
                "model": model.state_dict(),
                "config": OmegaConf.to_container(args, resolve=True),
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema_model.state_dict() if ema_model is not None else None,
                "rng": {
                    "torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                },
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
                dataset=dataset_name,
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