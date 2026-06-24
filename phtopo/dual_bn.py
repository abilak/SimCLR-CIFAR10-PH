"""
phtopo.dual_bn
==============

Dual BatchNorm (AdvProp / AdvCL style) for adversarial self-supervised training.

Clean and adversarial inputs have genuinely DIFFERENT feature statistics. Forcing
both through a single BatchNorm corrupts the running statistics -- the adversarial
batch shifts the mean/variance that the clean-eval path relies on -- which is the
well-documented reason naive adversarial training hurts clean accuracy and makes
robust SSL unstable (Xie et al. 2020, "Adversarial Examples Improve Image
Recognition"; Jiang et al. 2020, AdvCL). The fix is to keep TWO BatchNorm
branches: clean inputs flow through ``clean_bn``, adversarial inputs through
``adv_bn``. EVERY OTHER parameter (conv weights, etc.) is shared, so robustness
must still be earned in the shared weights -- BN only absorbs the distribution
shift between the clean and adversarial input distributions.

Routing
-------
A process-wide ``contextvars.ContextVar`` selects the active branch. ``forward``
reads it; the value at FORWARD time is what matters (autograd replays the ops
that actually ran, so the route at backward time is irrelevant). The default
route is ``"clean"``, so any code that is unaware of dual-BN -- the linear-probe
trainer, the robustness/mechanism evaluators, plain inference -- automatically
uses the clean branch. That is exactly the AdvProp inference convention (report
with the clean-BN statistics).

    from phtopo.dual_bn import convert_to_dual_bn, bn_route
    convert_to_dual_bn(model)            # replace every nn.BatchNorm2d in place
    with bn_route("adv"):
        ... forward adversarial inputs ...   # uses adv_bn
    ... forward clean inputs ...             # uses clean_bn (default route)

Self-describing checkpoints
---------------------------
After conversion, BatchNorm state lives under ``*.clean_bn.*`` / ``*.adv_bn.*``
keys, so a checkpoint announces whether it is dual-BN by its key names. Use
:func:`load_state_dict_auto` to load into a fresh (plain) model and have it
convert itself iff the checkpoint is dual-BN -- no config plumbing required, and
it cannot silently load a dual-BN checkpoint into a single-BN model.

When a model has NO ``DualBatchNorm2d`` modules, :func:`bn_route` is a harmless
no-op and nothing in the forward path changes (zero overhead, identical numerics).
"""

from __future__ import annotations

import contextlib
import contextvars

import torch
import torch.nn as nn

# Process-/coroutine-local active branch. ContextVar (not a plain global) so the
# route is correctly scoped and thread/async-safe, and always restored on exit.
_BN_ROUTE: "contextvars.ContextVar[str]" = contextvars.ContextVar(
    "phtopo_bn_route", default="clean"
)

VALID_ROUTES = ("clean", "adv")


@contextlib.contextmanager
def bn_route(route: str):
    """
    Context manager selecting the active BatchNorm branch for any
    :class:`DualBatchNorm2d` modules forwarded inside the block. Restores the
    previous route on exit, even on exception. No-op for models without dual-BN.
    """
    if route not in VALID_ROUTES:
        raise ValueError(f"bn_route must be one of {VALID_ROUTES}, got {route!r}")
    token = _BN_ROUTE.set(route)
    try:
        yield
    finally:
        _BN_ROUTE.reset(token)


def current_bn_route() -> str:
    """The branch the next DualBatchNorm2d.forward would use (``"clean"``/``"adv"``)."""
    return _BN_ROUTE.get()


class DualBatchNorm2d(nn.Module):
    """
    Two independent ``nn.BatchNorm2d`` branches; the active one is chosen by the
    :func:`bn_route` context (default ``"clean"``). The branches share NOTHING
    (separate affine params and running stats), so the clean path is never
    perturbed by adversarial-batch statistics and vice-versa.

    State layout (self-describing):
        clean_bn.{weight,bias,running_mean,running_var,num_batches_tracked}
        adv_bn.{...}
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.clean_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum,
                                       affine=affine, track_running_stats=track_running_stats)
        self.adv_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum,
                                     affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        # Read the route at forward time. Only one branch runs -> identical FLOPs
        # to a single BatchNorm (the other branch is untouched this pass).
        if _BN_ROUTE.get() == "adv":
            return self.adv_bn(x)
        return self.clean_bn(x)

    @classmethod
    def from_batchnorm(cls, bn: nn.BatchNorm2d) -> "DualBatchNorm2d":
        """
        Build a DualBatchNorm2d whose BOTH branches are EXACT clones of ``bn``
        (affine params, running stats, num_batches_tracked, eps, momentum,
        affine/track flags, device, dtype). Immediately after this, either branch
        reproduces ``bn`` bit-for-bit, so converting a model and forwarding through
        the clean branch in eval mode is identical to the original model.
        """
        if not isinstance(bn, nn.BatchNorm2d):
            raise TypeError(f"from_batchnorm expects nn.BatchNorm2d, got {type(bn)}")
        dual = cls(bn.num_features, eps=bn.eps, momentum=bn.momentum,
                   affine=bn.affine, track_running_stats=bn.track_running_stats)
        # Exact clone of buffers + params into both branches via state_dict
        # (covers running_mean/var and num_batches_tracked even when affine=False).
        sd = bn.state_dict()
        dual.clean_bn.load_state_dict(sd, strict=True)
        dual.adv_bn.load_state_dict(sd, strict=True)
        # Match the source BN's device/dtype (state_dict load preserves the source
        # tensors' device/dtype already, but be explicit for the no-buffer case).
        ref = next((t for t in list(bn.parameters()) + list(bn.buffers())), None)
        if ref is not None:
            dual = dual.to(device=ref.device, dtype=ref.dtype)
        return dual


def _is_other_batchnorm(m: nn.Module) -> bool:
    """A normalization layer we do NOT handle (BatchNorm1d/3d, SyncBatchNorm)."""
    from torch.nn.modules.batchnorm import _BatchNorm
    return isinstance(m, _BatchNorm) and not isinstance(m, (nn.BatchNorm2d, DualBatchNorm2d))


def convert_to_dual_bn(module: nn.Module, _allow_other_bn: bool = False) -> nn.Module:
    """
    Recursively replace every ``nn.BatchNorm2d`` in ``module`` (IN PLACE) with a
    :class:`DualBatchNorm2d` whose branches are exact clones. Returns ``module``.

    Idempotent: already-converted modules are left alone, so calling it twice is
    safe. By default it raises if it encounters a BatchNorm1d/3d/Sync layer (none
    occur in the ResNet18/34 encoders here) -- a guard so a normalization layer is
    never silently left outside the clean/adv split. Pass ``_allow_other_bn=True``
    to skip such layers instead.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, DualBatchNorm2d):
            continue  # already converted
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, DualBatchNorm2d.from_batchnorm(child))
        elif _is_other_batchnorm(child):
            if not _allow_other_bn:
                raise NotImplementedError(
                    f"convert_to_dual_bn only handles BatchNorm2d; found "
                    f"{type(child).__name__} at '{name}'. Pass _allow_other_bn=True "
                    f"to leave it single-BN."
                )
        else:
            convert_to_dual_bn(child, _allow_other_bn=_allow_other_bn)
    return module


@torch.no_grad()
def sync_adv_bn_from_clean(module: nn.Module) -> int:
    """
    Copy each DualBatchNorm2d's ``clean_bn`` state into its ``adv_bn`` (a warm
    start). Intended to be called ONCE at the warmup->adversarial transition: with
    a clean-baseline warmup, only ``clean_bn`` is trained, so ``adv_bn`` would
    otherwise begin adversarial training from a stale clone-of-init and have to
    re-estimate statistics from scratch (a transient instability spike right at the
    transition). Re-syncing gives the adv branch the warmed-up clean distribution as
    its starting point, then it specializes. Returns the number of branches synced.
    """
    n = 0
    for m in module.modules():
        if isinstance(m, DualBatchNorm2d):
            m.adv_bn.load_state_dict(m.clean_bn.state_dict())
            n += 1
    return n


def has_dual_bn(module: nn.Module) -> bool:
    """True if ``module`` contains at least one DualBatchNorm2d."""
    return any(isinstance(m, DualBatchNorm2d) for m in module.modules())


def count_dual_bn(module: nn.Module) -> int:
    """Number of DualBatchNorm2d modules in ``module``."""
    return sum(isinstance(m, DualBatchNorm2d) for m in module.modules())


def state_dict_is_dual_bn(state_dict) -> bool:
    """
    True if ``state_dict`` was saved from a dual-BN model. Requires BOTH a
    ``*.clean_bn.*`` and a paired ``*.adv_bn.*`` key, so a single module that
    happens to be named ``clean_bn`` cannot trigger a false positive (which would
    otherwise wrongly convert a plain model -- caught loudly by strict load, but
    better to never trigger it).
    """
    keys = list(state_dict.keys())
    has_clean = any(".clean_bn." in k or k.startswith("clean_bn.") for k in keys)
    has_adv = any(".adv_bn." in k or k.startswith("adv_bn.") for k in keys)
    return has_clean and has_adv


def load_state_dict_auto(model: nn.Module, state_dict, strict: bool = True):
    """
    Load ``state_dict`` into ``model``, first converting ``model`` to dual-BN iff
    the checkpoint is a dual-BN checkpoint (and the model isn't already). This is
    the single entry point every loader should use: it makes loaders dual-BN-aware
    with no config plumbing and refuses to silently mismatch BN layouts.
    """
    if state_dict_is_dual_bn(state_dict) and not has_dual_bn(model):
        convert_to_dual_bn(model)
    return model.load_state_dict(state_dict, strict=strict)


__all__ = [
    "DualBatchNorm2d",
    "bn_route",
    "current_bn_route",
    "convert_to_dual_bn",
    "sync_adv_bn_from_clean",
    "has_dual_bn",
    "count_dual_bn",
    "state_dict_is_dual_bn",
    "load_state_dict_auto",
    "VALID_ROUTES",
]
