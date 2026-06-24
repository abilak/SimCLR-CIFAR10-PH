"""
phtopo.adv
==========

Adversarial-training machinery for PH-ACL. This is the change that makes the
EXPERIMENTS match the paper's THEORY: the theory is built on the adversarial
persistent-separation risk Gamma_adv(f;x) = sup_{x' in U(x)} Gamma(f;x'), but the
original experiments trained the non-adversarial special case (clean SSL + a
topology term). Clean training is also why `phsim` was statistically
indistinguishable from the non-topological control `swcontrol`: with no
adversary in the loop, "shape the persistence diagram" and "shape the raw point
cloud" are nearly the same operation. Topology should only earn its keep when the
objective is to PRESERVE multiscale structure against a worst-case perturbation.

This module provides one generic primitive:

  pgd_ascent_on_loss(loss_closure, x, eps, alpha, steps)
      -> x_adv that (approximately) MAXIMIZES loss_closure within the Linf
         ball of radius eps around x, clamped to [0,1].

It is loss-agnostic: pass any closure x -> scalar (NT-Xent, topo separation, or
the topological-consistency disruption) and it returns the inner-maximizer. The
returned tensor is detached, so the outer optimization step is first-order
(standard adversarial training; no double-backprop).

BatchNorm note: the caller should put the encoder in eval() during the inner
ascent (so BN uses running stats and is not updated by adversarial inputs) and
back in train() for the outer step. `simclr.py` does this.
"""

from __future__ import annotations

import contextlib
from typing import Callable

import torch


@contextlib.contextmanager
def eval_mode(model):
    """
    Temporarily put a module in eval() (so BatchNorm uses running stats and does
    NOT update them during the adversarial inner pass), restoring the previous
    mode even if an exception is raised. Used to guard the inner PGD.
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def _project_linf(x_adv: torch.Tensor, x0: torch.Tensor, eps: float,
                  lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps)
    return x_adv.clamp(lo, hi)


@torch.enable_grad()
def pgd_ascent_on_loss(
    loss_closure: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float,
    alpha: float = None,
    steps: int = 5,
    random_start: bool = True,
) -> torch.Tensor:
    """
    Inner maximization for adversarial training. Returns x_adv (detached) that
    approximately maximizes `loss_closure` over the Linf ball of radius `eps`.

    Args:
        loss_closure: x' -> scalar loss to MAXIMIZE. Must be differentiable
            w.r.t. x' (it runs the encoder + the SSL/topology loss). Param grads
            are NOT accumulated here (we take grad only w.r.t. x').
        eps, alpha, steps: Linf budget, step size (default 2.5*eps/steps),
            and number of ascent steps.
    """
    if eps <= 0 or steps <= 0:
        return x.detach()
    if alpha is None:
        alpha = 2.5 * eps / steps
    x0 = x.detach()
    if random_start:
        x_adv = _project_linf(x0 + torch.empty_like(x0).uniform_(-eps, eps), x0, eps)
    else:
        x_adv = x0.clone()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = loss_closure(x_adv)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()      # ASCENT (maximize loss)
            x_adv = _project_linf(x_adv, x0, eps)
    return x_adv.detach()
