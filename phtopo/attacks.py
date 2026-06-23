"""
phtopo.attacks
==============

L-infinity attacks for the robustness evaluation (Tier-1 #1). A TDA regularizer
is exactly the kind of objective that can induce *gradient masking* and inflate
PGD numbers, so this module is built to expose masking, not hide it:

  * `pgd_linf`        - PGD with configurable steps, step size, and N random
                        restarts (worst-case over restarts).
  * `transfer_attack` - black-box transfer: craft adversarial examples on a
                        surrogate model, evaluate on the target.

AutoAttack (APGD-CE/APGD-T/FAB-T/Square) is run via the official `autoattack`
package in `phtopo.robustness` when available; that is the authoritative
gradient-free + ensemble check. These primitives provide the white-box PGD
step/eps/restart curves whose *plateauing* (or lack thereof) is the core masking
diagnostic.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


@torch.enable_grad()
def pgd_linf(
    model: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int = 10,
    alpha: Optional[float] = None,
    restarts: int = 1,
    random_start: bool = True,
) -> torch.Tensor:
    """
    L-inf PGD maximizing cross-entropy, taking the worst adversarial example over
    `restarts` random restarts (an example counts as robust only if it survives
    every restart). Returns adversarial inputs (detached).

    alpha defaults to 2.5 * eps / steps (a standard choice that does not vanish
    for large step counts, so the step-count curve is meaningful).
    """
    if alpha is None:
        alpha = 2.5 * eps / max(1, steps)
    x0 = x.detach()
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    # Track the worst (still-misclassified-preferring) example per input.
    best_adv = x0.clone()
    # Initialize "best loss" very low so the first restart always writes.
    best_loss = torch.full((x0.shape[0],), -1e30, device=x0.device)

    for _ in range(max(1, restarts)):
        if random_start:
            delta = torch.empty_like(x0).uniform_(-eps, eps)
            x_adv = _clamp01(x0 + delta).detach()
        else:
            x_adv = x0.clone()

        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y, reduction="sum")
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv + alpha * grad.sign()
                x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps)
                x_adv = _clamp01(x_adv).detach()

        with torch.no_grad():
            logits = model(x_adv)
            per = F.cross_entropy(logits, y, reduction="none")
            improved = per > best_loss
            best_loss = torch.where(improved, per, best_loss)
            best_adv[improved] = x_adv[improved]

    if was_training and hasattr(model, "train"):
        model.train()
    return best_adv.detach()


@torch.no_grad()
def accuracy_under(model: Callable, x_adv: torch.Tensor, y: torch.Tensor) -> float:
    logits = model(x_adv)
    return float((logits.argmax(1) == y).float().mean())


def transfer_attack(
    target: Callable,
    surrogate: Callable,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int = 50,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """
    Black-box transfer: build PGD adversarials on `surrogate`, return them (to be
    evaluated on `target`). If a model is genuinely robust, transfer accuracy is
    similar to white-box; if it is masking gradients, transfer (and Square) break
    it far more than its own white-box PGD does.
    """
    return pgd_linf(surrogate, x, y, eps=eps, steps=steps, alpha=alpha, restarts=1)
