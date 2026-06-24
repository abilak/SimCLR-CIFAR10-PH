"""
RIGOROUS correctness tests for dual-BN (AdvProp/AdvCL).

These are deliberately harsh. Dual-BN is invisible when it works and silently
destroys clean accuracy when it doesn't (a stale or cross-contaminated branch
produces garbage running stats that only show up at eval). So we check, with
exact (bit-identical) comparisons wherever possible:

  1.  Conversion is bit-identical to the original model (eval, clean branch) and
      both branches are exact clones at init.
  2.  Conversion is bit-identical in train mode on the first forward, and updates
      running stats identically to the original.
  3.  Every BatchNorm2d is converted; count matches; conversion is idempotent.
  4.  NO cross-contamination: a clean-routed forward updates ONLY clean_bn stats,
      an adv-routed forward updates ONLY adv_bn stats (mean, var, num_batches).
  5.  Gradient isolation: only the routed branch receives gradient.
  6.  Routing actually changes the output once branches differ; eval uses clean.
  7.  bn_route restores the previous route on exception; no-op on plain models.
  8.  Checkpoints are self-describing and round-trip strict=True; a dual ckpt
      auto-converts a fresh plain model; a plain ckpt does NOT trigger conversion.
  9.  Optimizer sees the adv-BN parameters after conversion.
  10. End-to-end: every adversarial method with dual_bn=True gives a finite loss,
      gradient to the shared encoder AND to BOTH BN branches, advances BOTH
      branches' running stats by exactly one step, leaves no param-grad from the
      inner attack, and restores train mode. EMA-teacher path included.
  11. The other-BatchNorm guard fires (BatchNorm1d/3d) unless explicitly allowed.

Run: python tests/test_dual_bn.py
"""
import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision.models import resnet18

from models import SimCLR
from phtopo.dual_bn import (
    DualBatchNorm2d, bn_route, current_bn_route, convert_to_dual_bn,
    has_dual_bn, count_dual_bn, state_dict_is_dual_bn, load_state_dict_auto,
    sync_adv_bn_from_clean,
)
from simclr import compute_training_loss, ema_update, ADV_WRAP_METHODS, CONSIST_METHODS

P_SINGLE = dict(temperature=0.5, num_points=16, neg_k=4, margin=1.0, ndir=16, ph_lambda=1.0,
                neg_agg="hard", softmin_temp=0.1, alpha=0.9, multiscale=False,
                adv_eps=8/255, adv_alpha=2.5*(8/255)/3, adv_steps=3, adv_beta=1.0,
                adv_random_start=True, dual_bn=True)


def _model():
    return SimCLR(resnet18, projection_dim=64, reduce_channels=8, ph_source_layer="layer3")


def _count_bn2d(m):
    return sum(isinstance(x, nn.BatchNorm2d) for x in m.modules())


def _first_dual(m):
    return next(x for x in m.modules() if isinstance(x, DualBatchNorm2d))


# ---------------------------------------------------------------------------
# 1. Bit-identical conversion (eval mode) + both branches are clones
# ---------------------------------------------------------------------------
def test_conversion_bit_identical_eval():
    torch.manual_seed(0)
    m = _model()
    m_orig = copy.deepcopy(m).eval()
    n_bn = _count_bn2d(m)
    convert_to_dual_bn(m)
    m.eval()
    assert count_dual_bn(m) == n_bn, f"converted {count_dual_bn(m)} != {n_bn} BatchNorm2d"
    # every remaining BatchNorm2d must be a branch of a dual module (2 per dual),
    # i.e. none of the original BNs was left unconverted.
    assert _count_bn2d(m) == 2 * n_bn, \
        f"an unconverted BatchNorm2d survived: {_count_bn2d(m)} BN != 2*{n_bn} branches"

    x = torch.rand(8, 3, 32, 32)
    with torch.no_grad():
        o_orig = m_orig(x)[2]
        o_clean = m(x)[2]                       # default route == clean
        with bn_route("clean"):
            o_clean2 = m(x)[2]
        with bn_route("adv"):
            o_adv = m(x)[2]                      # adv branch is a clone at init
    assert torch.equal(o_orig, o_clean), "clean branch not bit-identical to original (eval)"
    assert torch.equal(o_clean, o_clean2), "default route differs from explicit clean route"
    assert torch.equal(o_orig, o_adv), "adv branch (clone at init) not identical to original"
    print(f"[OK] conversion bit-identical in eval; {n_bn} BN -> dual; both branches == original")


# ---------------------------------------------------------------------------
# 2. Bit-identical in train mode (first forward) + identical stat updates
# ---------------------------------------------------------------------------
def test_conversion_bit_identical_train_and_stats():
    torch.manual_seed(1)
    m = _model()
    m_orig = copy.deepcopy(m)
    convert_to_dual_bn(m)
    m.train(); m_orig.train()
    x = torch.rand(16, 3, 32, 32)
    o_orig = m_orig(x)[2]
    o_dual = m(x)[2]                            # clean route, train mode
    assert torch.equal(o_orig, o_dual), "train-mode first forward not bit-identical"
    # running stats must have moved identically: compare a representative BN.
    bn_orig = next(x for x in m_orig.modules() if isinstance(x, nn.BatchNorm2d))
    d = _first_dual(m)
    assert torch.equal(bn_orig.running_mean, d.clean_bn.running_mean), "clean_bn running_mean drifted"
    assert torch.equal(bn_orig.running_var, d.clean_bn.running_var), "clean_bn running_var drifted"
    assert int(d.adv_bn.num_batches_tracked) == 0, "adv_bn updated on a clean forward"
    print("[OK] conversion bit-identical in train; clean_bn stats track original; adv_bn untouched")


# ---------------------------------------------------------------------------
# 3. Idempotent conversion
# ---------------------------------------------------------------------------
def test_conversion_idempotent():
    m = _model()
    n = _count_bn2d(m)
    convert_to_dual_bn(m)
    c1 = count_dual_bn(m)
    convert_to_dual_bn(m)                       # second pass must be a no-op
    c2 = count_dual_bn(m)
    assert c1 == c2 == n, f"idempotency broken: {n} -> {c1} -> {c2}"
    # no DualBatchNorm2d nested inside another (no double-wrap)
    for mod in m.modules():
        if isinstance(mod, DualBatchNorm2d):
            assert not has_dual_bn(mod.clean_bn) and not has_dual_bn(mod.adv_bn), "double-wrapped BN"
    print(f"[OK] conversion idempotent ({n} dual-BN, no double-wrap)")


# ---------------------------------------------------------------------------
# 4. No cross-contamination of running statistics
# ---------------------------------------------------------------------------
def test_no_stat_cross_contamination():
    torch.manual_seed(2)
    m = _model(); convert_to_dual_bn(m); m.train()
    d = _first_dual(m)

    cm0, cv0 = d.clean_bn.running_mean.clone(), d.clean_bn.running_var.clone()
    am0, av0 = d.adv_bn.running_mean.clone(), d.adv_bn.running_var.clone()
    cn0 = int(d.clean_bn.num_batches_tracked); an0 = int(d.adv_bn.num_batches_tracked)

    # CLEAN-routed forward: only clean_bn may change.
    with bn_route("clean"):
        m(torch.rand(16, 3, 32, 32))
    assert not torch.equal(d.clean_bn.running_mean, cm0), "clean_bn stats did NOT update on clean route"
    assert torch.equal(d.adv_bn.running_mean, am0) and torch.equal(d.adv_bn.running_var, av0), \
        "adv_bn stats CHANGED on a clean-routed forward (cross-contamination)"
    assert int(d.adv_bn.num_batches_tracked) == an0, "adv_bn num_batches_tracked moved on clean route"
    assert int(d.clean_bn.num_batches_tracked) == cn0 + 1

    cm1 = d.clean_bn.running_mean.clone()
    # ADV-routed forward: only adv_bn may change; clean_bn must stay put.
    with bn_route("adv"):
        m(torch.rand(16, 3, 32, 32))
    assert not torch.equal(d.adv_bn.running_mean, am0), "adv_bn stats did NOT update on adv route"
    assert torch.equal(d.clean_bn.running_mean, cm1), "clean_bn stats CHANGED on an adv-routed forward"
    assert int(d.clean_bn.num_batches_tracked) == cn0 + 1, "clean_bn num_batches moved on adv route"
    assert int(d.adv_bn.num_batches_tracked) == an0 + 1
    print("[OK] no cross-contamination: clean/adv running stats + num_batches fully independent")


# ---------------------------------------------------------------------------
# 5. Gradient isolation: only the routed branch gets gradient
# ---------------------------------------------------------------------------
def test_gradient_isolation():
    torch.manual_seed(3)
    m = _model(); convert_to_dual_bn(m); m.train()
    x = torch.rand(8, 3, 32, 32)

    m.zero_grad(set_to_none=True)
    with bn_route("clean"):
        m(x)[2].pow(2).sum().backward()
    d = _first_dual(m)
    assert d.clean_bn.weight.grad is not None and float(d.clean_bn.weight.grad.abs().sum()) > 0, \
        "clean_bn got no gradient on a clean-routed backward"
    assert d.adv_bn.weight.grad is None, "adv_bn got gradient on a clean-routed backward"

    m.zero_grad(set_to_none=True)
    with bn_route("adv"):
        m(x)[2].pow(2).sum().backward()
    assert d.adv_bn.weight.grad is not None and float(d.adv_bn.weight.grad.abs().sum()) > 0, \
        "adv_bn got no gradient on an adv-routed backward"
    assert d.clean_bn.weight.grad is None, "clean_bn got gradient on an adv-routed backward"
    print("[OK] gradient isolation: only the routed branch receives gradient")


# ---------------------------------------------------------------------------
# 6. Routing changes output once branches differ; eval uses clean branch
# ---------------------------------------------------------------------------
def test_routing_effect_and_eval_uses_clean():
    torch.manual_seed(4)
    m = _model(); convert_to_dual_bn(m)
    # Make adv_bn genuinely different from clean_bn (as it would be after training).
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, DualBatchNorm2d):
                mod.adv_bn.weight.add_(0.5)
                mod.adv_bn.running_mean.add_(0.3)
    m.eval()
    x = torch.rand(8, 3, 32, 32)
    with torch.no_grad():
        o_clean = m(x)[2]
        with bn_route("adv"):
            o_adv = m(x)[2]
        o_default = m(x)[2]
    assert not torch.allclose(o_clean, o_adv), "routing has no effect though branches differ"
    assert torch.equal(o_clean, o_default), "eval/default route did NOT use the clean branch"
    print("[OK] routing changes output; eval/default route uses the clean branch (AdvProp convention)")


# ---------------------------------------------------------------------------
# 7. Context safety: restore on exception; no-op on plain model
# ---------------------------------------------------------------------------
def test_context_safety():
    assert current_bn_route() == "clean"
    try:
        with bn_route("adv"):
            assert current_bn_route() == "adv"
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert current_bn_route() == "clean", "bn_route did not restore route after exception"
    # nested
    with bn_route("adv"):
        with bn_route("clean"):
            assert current_bn_route() == "clean"
        assert current_bn_route() == "adv"
    assert current_bn_route() == "clean"
    # no-op on a plain (unconverted) model
    m = _model().eval()
    x = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        a = m(x)[2]
        with bn_route("adv"):
            b = m(x)[2]
    assert torch.equal(a, b), "bn_route changed a plain model's output (should be a no-op)"
    print("[OK] context safety: restores on exception, nests correctly, no-op without dual-BN")


# ---------------------------------------------------------------------------
# 8. Self-describing checkpoints + round-trip
# ---------------------------------------------------------------------------
def test_checkpoint_roundtrip_and_detection():
    m = _model(); convert_to_dual_bn(m)
    sd = m.state_dict()
    assert state_dict_is_dual_bn(sd), "dual-BN state_dict not detected"
    assert any(".clean_bn." in k for k in sd) and any(".adv_bn." in k for k in sd)

    # fresh plain model auto-converts and loads strict=True
    fresh = _model()
    assert not has_dual_bn(fresh)
    load_state_dict_auto(fresh, sd, strict=True)
    assert has_dual_bn(fresh), "load_state_dict_auto did not convert for a dual ckpt"

    # a plain checkpoint must NOT trigger conversion
    plain = _model(); plain_sd = plain.state_dict()
    assert not state_dict_is_dual_bn(plain_sd)
    fresh2 = _model()
    load_state_dict_auto(fresh2, plain_sd, strict=True)
    assert not has_dual_bn(fresh2), "load_state_dict_auto wrongly converted for a plain ckpt"

    # loading a dual ckpt into an already-dual model also works (no double convert)
    again = _model(); convert_to_dual_bn(again)
    load_state_dict_auto(again, sd, strict=True)
    assert count_dual_bn(again) == count_dual_bn(m)
    print("[OK] checkpoints self-describing; round-trip strict=True; no false-positive conversion")


# ---------------------------------------------------------------------------
# 9. Optimizer sees adv-BN parameters
# ---------------------------------------------------------------------------
def test_optimizer_sees_adv_params():
    m = _model()
    n_before = sum(p.numel() for p in m.parameters())
    convert_to_dual_bn(m)
    n_after = sum(p.numel() for p in m.parameters())
    # exactly one extra BN's worth of affine params (weight+bias) per converted BN
    extra = sum(d.adv_bn.weight.numel() + d.adv_bn.bias.numel()
                for d in m.modules() if isinstance(d, DualBatchNorm2d))
    assert n_after == n_before + extra, f"param count off: {n_before}+{extra} != {n_after}"
    ids = {id(p) for p in m.parameters()}
    for d in m.modules():
        if isinstance(d, DualBatchNorm2d):
            assert id(d.adv_bn.weight) in ids and id(d.adv_bn.bias) in ids, "adv_bn params not in parameters()"
    print(f"[OK] optimizer sees adv-BN params (+{extra} params across {count_dual_bn(m)} branches)")


# ---------------------------------------------------------------------------
# 10. End-to-end through compute_training_loss for every adversarial method
# ---------------------------------------------------------------------------
def _branch_grad_sums(m):
    c = sum(float(d.clean_bn.weight.grad.abs().sum()) for d in m.modules()
            if isinstance(d, DualBatchNorm2d) and d.clean_bn.weight.grad is not None)
    a = sum(float(d.adv_bn.weight.grad.abs().sum()) for d in m.modules()
            if isinstance(d, DualBatchNorm2d) and d.adv_bn.weight.grad is not None)
    return c, a


def test_end_to_end_methods_dual_bn():
    torch.manual_seed(5)
    x = torch.rand(8, 3, 32, 32)
    for meth in sorted(ADV_WRAP_METHODS | CONSIST_METHODS):
        m = _model(); convert_to_dual_bn(m); m.train(); m.zero_grad(set_to_none=True)
        d = _first_dual(m)
        cn0, an0 = int(d.clean_bn.num_batches_tracked), int(d.adv_bn.num_batches_tracked)

        loss, st = compute_training_loss(m, x, meth, P_SINGLE)
        assert torch.isfinite(loss), f"{meth}: non-finite loss"
        # the inner attack must not have accumulated any parameter gradient
        assert all(p.grad is None for p in m.parameters()), \
            f"{meth}: inner attack leaked gradient into parameters"
        assert m.training, f"{meth}: train mode not restored after the inner loop"
        # clean-BN always trains in TRAIN mode on the clean inputs (one step / call).
        assert int(d.clean_bn.num_batches_tracked) == cn0 + 1, \
            f"{meth}: clean_bn not stepped once ({int(d.clean_bn.num_batches_tracked)-cn0})"
        # adv-BN running-stat policy differs by method family:
        #   ADV_WRAP (AdvProp): outer adv forward is TRAIN mode -> adv-BN steps once.
        #   CONSIST (topoacl/rawacl): the consistency runs the adv branch in EVAL
        #     mode (to mode-match the clean target and isolate perturbation drift),
        #     so adv-BN running stats do NOT advance -- its AFFINE params still train.
        if meth in ADV_WRAP_METHODS:
            assert int(d.adv_bn.num_batches_tracked) == an0 + 1, \
                f"{meth}: adv_bn not stepped once ({int(d.adv_bn.num_batches_tracked)-an0})"
        else:
            assert int(d.adv_bn.num_batches_tracked) == an0, \
                f"{meth}: adv_bn running stats moved though consistency is eval-mode"

        loss.backward()
        g = m.stem[0].weight.grad
        assert g is not None and float(g.abs().sum()) > 0, f"{meth}: no grad to shared encoder"
        c, a = _branch_grad_sums(m)
        assert c > 0, f"{meth}: clean-BN branch received no gradient"
        assert a > 0, f"{meth}: adv-BN branch received no gradient (affine must train)"
        if meth in ADV_WRAP_METHODS:
            assert "clean_loss" in st and "adv_loss" in st, f"{meth}: missing AdvProp clean/adv loss split"
    print(f"[OK] end-to-end dual-BN for {len(ADV_WRAP_METHODS|CONSIST_METHODS)} methods: "
          "finite loss, both branches trained, correct adv-BN stat policy, no attack grad leak, train restored")


def test_consistency_zero_at_zero_perturbation():
    """
    Regression guard for the branch-gap contamination bug: with the attack disabled
    (adv_steps=0, no random start), x_adv == x, so the dual-BN consistency MUST be
    ~0. The buggy version compared clean-BN(x) to adv-BN(x) and charged a large
    constant (the learned branch gap) even at zero perturbation.
    """
    torch.manual_seed(8)
    x = torch.rand(8, 3, 32, 32)
    P0 = {**P_SINGLE, "adv_steps": 0, "adv_random_start": False}
    for meth in ["topoacl", "rawacl"]:
        m = _model(); convert_to_dual_bn(m); m.train()
        # Force the branches genuinely apart (as after training) so a cross-branch
        # comparison WOULD be large; a same-branch comparison stays ~0.
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, DualBatchNorm2d):
                    mod.adv_bn.weight.add_(0.4); mod.adv_bn.bias.add_(0.2)
                    mod.adv_bn.running_mean.add_(0.3); mod.adv_bn.running_var.add_(0.5)
        _, st = compute_training_loss(m, x, meth, P0)
        assert st["consistency"] < 1e-4, \
            f"{meth}: consistency={st['consistency']:.4f} at zero perturbation " \
            f"(branch-gap contamination -- should be ~0)"
    print("[OK] consistency ~0 at zero perturbation (no clean/adv branch-gap leak into the loss)")


def test_end_to_end_ema_teacher_dual_bn():
    torch.manual_seed(6)
    m = _model(); convert_to_dual_bn(m); m.train()
    teacher = copy.deepcopy(m)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    assert has_dual_bn(teacher), "EMA teacher should also be dual-BN (deepcopy after convert)"
    x = torch.rand(8, 3, 32, 32)
    for meth in ["topoacl", "rawacl"]:
        m.zero_grad(set_to_none=True)
        loss, st = compute_training_loss(m, x, meth, P_SINGLE, teacher_model=teacher)
        assert torch.isfinite(loss) and "consistency" in st
        loss.backward()
        assert float(m.stem[0].weight.grad.abs().sum()) > 0, f"{meth}+ema+dualbn: no grad to encoder"
        c, a = _branch_grad_sums(m)
        assert c > 0 and a > 0, f"{meth}+ema+dualbn: a BN branch got no gradient"
        assert all(p.grad is None for p in teacher.parameters()), "teacher received gradient"
    # EMA update over a dual-BN structure (params + both BN branch buffers)
    before = teacher.stem[0].weight.clone()
    for p in m.parameters():
        if p.grad is not None:
            p.data.add_(0.1)
    ema_update(teacher, m, momentum=0.9)
    assert not torch.equal(before, teacher.stem[0].weight), "ema_update did not move teacher"
    print("[OK] EMA-teacher + dual-BN: topoacl/rawacl finite, both branches trained, teacher frozen & EMA-updated")


# ---------------------------------------------------------------------------
# 11. Guard against unsupported BatchNorm variants
# ---------------------------------------------------------------------------
def test_other_batchnorm_guard():
    net = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    raised = False
    try:
        convert_to_dual_bn(net)
    except NotImplementedError:
        raised = True
    assert raised, "convert_to_dual_bn should refuse BatchNorm1d by default"
    # explicit opt-out leaves it single-BN, untouched
    convert_to_dual_bn(net, _allow_other_bn=True)
    assert isinstance(net[1], nn.BatchNorm1d) and not has_dual_bn(net)
    print("[OK] other-BatchNorm guard fires (BatchNorm1d); _allow_other_bn=True leaves it alone")


def test_warm_start_sync():
    """sync_adv_bn_from_clean copies warmed clean-BN stats into adv-BN exactly."""
    torch.manual_seed(9)
    m = _model(); convert_to_dual_bn(m); m.train()
    # 'warm up' only the clean branch so the branches diverge.
    with bn_route("clean"):
        for _ in range(4):
            m(torch.rand(16, 3, 32, 32))
    d = _first_dual(m)
    assert not torch.equal(d.clean_bn.running_mean, d.adv_bn.running_mean), "branches should differ pre-sync"
    n = sync_adv_bn_from_clean(m)
    assert n == count_dual_bn(m)
    for mod in m.modules():
        if isinstance(mod, DualBatchNorm2d):
            assert torch.equal(mod.adv_bn.running_mean, mod.clean_bn.running_mean)
            assert torch.equal(mod.adv_bn.running_var, mod.clean_bn.running_var)
            assert torch.equal(mod.adv_bn.weight, mod.clean_bn.weight)
            assert torch.equal(mod.adv_bn.bias, mod.clean_bn.bias)
            assert int(mod.adv_bn.num_batches_tracked) == int(mod.clean_bn.num_batches_tracked)
            # and they remain INDEPENDENT objects (no aliasing) after the sync
            assert mod.adv_bn.running_mean is not mod.clean_bn.running_mean
    # mutating clean afterwards must not touch adv (proves a real copy, not a view)
    with torch.no_grad():
        d.clean_bn.running_mean.add_(1.0)
    assert not torch.equal(d.adv_bn.running_mean, d.clean_bn.running_mean), "sync aliased the buffers"
    print(f"[OK] warm-start sync copies clean-BN -> adv-BN exactly and independently ({n} branches)")


def test_from_batchnorm_preserves_config():
    """from_batchnorm must clone eps/momentum(/None)/affine/track flags + dtype exactly."""
    for affine in (True, False):
        for track in (True, False):
            for momentum in (0.1, None):
                bn = nn.BatchNorm2d(6, eps=7e-4, momentum=momentum, affine=affine,
                                    track_running_stats=track)
                if track:  # move running stats off init so cloning is observable
                    bn.train()
                    bn(torch.randn(4, 6, 5, 5))
                bn = bn.to(torch.float64)
                d = DualBatchNorm2d.from_batchnorm(bn)
                for branch in (d.clean_bn, d.adv_bn):
                    assert branch.eps == bn.eps and branch.momentum == bn.momentum
                    assert branch.affine == affine and branch.track_running_stats == track
                    assert branch.weight is None if not affine else branch.weight is not None
                    assert next(branch.parameters(), torch.empty(0, dtype=torch.float64)).dtype == torch.float64
                    if track:
                        assert torch.equal(branch.running_mean, bn.running_mean)
                        assert torch.equal(branch.running_var, bn.running_var)
                        assert int(branch.num_batches_tracked) == int(bn.num_batches_tracked)
                    else:
                        assert branch.running_mean is None and branch.running_var is None
    print("[OK] from_batchnorm preserves eps/momentum(None)/affine/track/dtype/running-stats across all flag combos")


if __name__ == "__main__":
    test_conversion_bit_identical_eval()
    test_conversion_bit_identical_train_and_stats()
    test_conversion_idempotent()
    test_no_stat_cross_contamination()
    test_gradient_isolation()
    test_routing_effect_and_eval_uses_clean()
    test_context_safety()
    test_checkpoint_roundtrip_and_detection()
    test_optimizer_sees_adv_params()
    test_end_to_end_methods_dual_bn()
    test_consistency_zero_at_zero_perturbation()
    test_end_to_end_ema_teacher_dual_bn()
    test_other_batchnorm_guard()
    test_warm_start_sync()
    test_from_batchnorm_preserves_config()
    print("\nAll dual-BN correctness tests passed.")
