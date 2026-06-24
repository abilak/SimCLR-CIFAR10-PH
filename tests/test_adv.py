"""
Correctness tests for adversarial PH-ACL, the consistency objectives, multiscale
PH, and checkpoint round-trips. Needs torch + torchvision (resnet18).

Run: python tests/test_adv.py
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import SimCLR
from torchvision.models import resnet18
from simclr import compute_training_loss, nt_xent, ALL_METHODS
from phtopo.adv import pgd_ascent_on_loss
from phtopo.losses import topo_consistency_loss

P_SINGLE = dict(temperature=0.5, num_points=16, neg_k=4, margin=1.0, ndir=16, ph_lambda=1.0,
                neg_agg="hard", softmin_temp=0.1, alpha=0.9, multiscale=False,
                adv_eps=8/255, adv_alpha=2.5*(8/255)/3, adv_steps=3, adv_beta=1.0, adv_random_start=True)


def _model(extra=()):
    return SimCLR(resnet18, projection_dim=64, reduce_channels=8,
                  ph_source_layer="layer3", ph_extra_layers=extra)


def test_pgd_ascent_increases_and_bounded():
    torch.manual_seed(0)
    m = _model().eval()
    x = torch.rand(8, 3, 32, 32)
    closure = lambda xp: nt_xent(m(xp)[2], 0.5)
    l0 = float(closure(x))
    xa = pgd_ascent_on_loss(closure, x, eps=8/255, alpha=2.5*(8/255)/5, steps=5)
    l1 = float(closure(xa))
    assert l1 > l0, f"ascent did not increase loss: {l0}->{l1}"
    assert float((xa - x).abs().max()) <= 8/255 + 1e-6, "Linf budget violated"
    assert float(xa.min()) >= 0 and float(xa.max()) <= 1, "left [0,1]"
    print(f"[OK] PGD ascent: {l0:.3f}->{l1:.3f}, Linf<=8/255, in [0,1]")


def test_all_methods_finite_grad_bn_restored():
    torch.manual_seed(1)
    x = torch.rand(8, 3, 32, 32)
    for meth in sorted(ALL_METHODS):
        m = _model(); m.train(); m.zero_grad()
        loss, st = compute_training_loss(m, x, meth, P_SINGLE)
        assert torch.isfinite(loss), f"{meth}: non-finite loss"
        loss.backward()
        g = m.stem[0].weight.grad
        assert g is not None and float(g.abs().sum()) > 0, f"{meth}: no grad to encoder"
        assert m.training, f"{meth}: train mode not restored after adversarial inner loop"
    print(f"[OK] all {len(ALL_METHODS)} methods: finite loss, grad->encoder, BN train-mode restored")


def test_multiscale():
    torch.manual_seed(2)
    P = {**P_SINGLE, "multiscale": True, "neg_agg": "soft"}
    m = _model(extra=("layer2",))
    maps, _, _ = m.ph_maps(torch.rand(4, 3, 32, 32))
    assert len(maps) == 2 and maps[0].shape[-1] == 4 and maps[1].shape[-1] == 8
    x = torch.rand(8, 3, 32, 32)
    for meth in ["phsim", "topoacl", "adv_phsim", "hybrid"]:
        m.zero_grad()
        loss, _ = compute_training_loss(m, x, meth, P)
        assert torch.isfinite(loss)
        loss.backward()
        assert float(m.stem[0].weight.grad.abs().sum()) > 0
    print("[OK] multiscale (layer3+layer2): maps + finite loss + grad for 4 methods")


def test_ph_maps_only_equivalence_and_skips_layer4():
    torch.manual_seed(11)
    x = torch.rand(6, 3, 32, 32, requires_grad=True)
    # single-scale (layer3): ph_maps_only must equal the full forward's PH map exactly
    m = _model().eval()
    full = m(x)[0]
    only = m.ph_maps_only(x)
    assert len(only) == 1 and torch.equal(full, only[0]), "ph_maps_only != full forward (single scale)"
    # and it must NOT execute layer4 (the skipped, expensive block)
    calls = {"n": 0}
    h = m.layer4.register_forward_hook(lambda *a: calls.__setitem__("n", calls["n"] + 1))
    try:
        _ = m.ph_maps_only(x); assert calls["n"] == 0, "ph_maps_only ran layer4 (no speedup)"
        _ = m(x);              assert calls["n"] == 1, "full forward should run layer4"
    finally:
        h.remove()
    # multiscale (layer3+layer2): identical to ph_maps()'s map list
    mm = _model(extra=("layer2",)).eval()
    ref = mm.ph_maps(x)[0]
    got = mm.ph_maps_only(x)
    assert len(got) == 2 and all(torch.equal(a, b) for a, b in zip(ref, got)), "multiscale maps differ"
    # gradient flows to the input through ph_maps_only
    mm.ph_maps_only(x)[0].pow(2).sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0, "no grad through ph_maps_only"
    print("[OK] ph_maps_only: bit-identical to full forward, skips layer4, grad flows (single+multiscale)")


def test_consistency_semantics():
    torch.manual_seed(3)
    h = torch.randn(6, 8, 4, 4)
    same, _ = topo_consistency_loss(h, h, num_points=16, n_directions=16)
    diff, _ = topo_consistency_loss(h, h + 0.5 * torch.randn_like(h), num_points=16, n_directions=16)
    assert float(same) < 1e-6, f"consistency(h,h) should be ~0, got {float(same)}"
    assert float(diff) > float(same), "perturbed topology should drift more"
    print(f"[OK] consistency: self={float(same):.2e}, perturbed={float(diff):.3f}")


def test_checkpoint_roundtrips():
    from eval_robustness import build_encoder
    # multiscale
    m = _model(extra=("layer2",))
    build_encoder("resnet18", 64, 512, 8, "cpu", "layer3", ("layer2",)).load_state_dict(m.state_dict(), strict=True)
    # single layer3
    build_encoder("resnet18", 64, 512, 8, "cpu", "layer3").load_state_dict(_model().state_dict(), strict=True)
    # old layer4 (backward compat)
    m4 = SimCLR(resnet18, projection_dim=64, reduce_channels=8, ph_source_layer="layer4")
    build_encoder("resnet18", 64, 512, 8, "cpu", "layer4").load_state_dict(m4.state_dict(), strict=True)
    print("[OK] checkpoint round-trips: multiscale / single / old-layer4 (strict=True)")


def test_ema_teacher_consistency():
    import copy
    from simclr import compute_training_loss, ema_update
    torch.manual_seed(7)
    m = _model(); m.train()
    teacher = copy.deepcopy(m)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    x = torch.rand(8, 3, 32, 32)
    for meth in ["topoacl", "rawacl"]:
        m.zero_grad()
        loss, st = compute_training_loss(m, x, meth, P_SINGLE, teacher_model=teacher)
        assert torch.isfinite(loss) and "consistency" in st
        loss.backward()
        assert float(m.stem[0].weight.grad.abs().sum()) > 0, f"{meth}+ema: no grad to encoder"
        # teacher must receive NO gradient
        assert all(p.grad is None for p in teacher.parameters()), "teacher got gradient"
    # ema_update moves teacher toward the (now-updated) student and keeps it frozen
    before = teacher.stem[0].weight.clone()
    for p in m.parameters():
        if p.grad is not None:
            p.data.add_(0.1)  # simulate an optimizer step
    ema_update(teacher, m, momentum=0.9)
    assert not torch.equal(before, teacher.stem[0].weight), "ema_update did not move teacher"
    assert all(not p.requires_grad for p in teacher.parameters()), "teacher should stay frozen"
    print("[OK] EMA teacher: topoacl/rawacl finite + grad to student, teacher frozen & EMA-updated")


if __name__ == "__main__":
    test_pgd_ascent_increases_and_bounded()
    test_all_methods_finite_grad_bn_restored()
    test_multiscale()
    test_ph_maps_only_equivalence_and_skips_layer4()
    test_consistency_semantics()
    test_checkpoint_roundtrips()
    test_ema_teacher_consistency()
    print("\nAll adversarial / consistency / multiscale / EMA tests passed.")
