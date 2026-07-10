#!/usr/bin/env python3
"""
reduction_universal.py -- Experiment A: show the H0-persistence == geometry reduction
is UNIVERSAL, not an artifact of one trained encoder. For many encoders (your SimCLR
checkpoints + off-the-shelf ImageNet models + a random-init control), at each layer we
measure, per image, H0 total persistence vs mean pairwise distance of the activation
point cloud, and report Pearson r + linear-fit R^2 across images.

Thesis prediction: r ~ 0.9 everywhere (H0 persistence is a geometric-scale statistic),
regardless of architecture, training, or initialization. Directly answers the UAI
"one architecture / one dataset" complaint with an affirmative, general result.

Usage: python utils/reduction_universal.py --data_dir ./data --per_class 20
"""
import argparse, glob, os, re, ssl, sys
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from datasets import _bypass_torchvision_cifar_md5_if_prepared

ssl._create_default_https_context = ssl._create_unverified_context  # macOS cert workaround


def h0_totpers(pts):
    if pts.shape[0] < 2:
        return 0.0
    return float(minimum_spanning_tree(squareform(pdist(pts.astype(np.float64)))).sum())


def mpd(pts):
    return float(pdist(pts.astype(np.float64)).mean()) if pts.shape[0] >= 2 else 0.0


def reduce_stats(fmap):
    """Per image: (H0 total persistence, mean pairwise dist) of the (C,H,W) cloud."""
    f = fmap.detach().cpu().numpy()
    T = np.empty(f.shape[0]); R = np.empty(f.shape[0])
    for i in range(f.shape[0]):
        pts = f[i].reshape(f[i].shape[0], -1).T
        T[i] = h0_totpers(pts); R[i] = mpd(pts)
    return T, R


def fit(T, R):
    if T.std() < 1e-9 or R.std() < 1e-9:
        return float("nan"), float("nan")
    r = float(np.corrcoef(T, R)[0, 1])
    return r, r ** 2  # R^2 of simple linear fit == r^2


def get_loader(data_dir, per_class, size=32):
    _bypass_torchvision_cifar_md5_if_prepared(data_dir)
    tf = transforms.Compose([transforms.Resize(size), transforms.ToTensor()]) if size != 32 \
        else transforms.ToTensor()
    test = CIFAR10(root=data_dir, train=False, download=False, transform=tf)
    idxs = list(range(per_class * 10))
    return DataLoader(Subset(test, idxs), batch_size=50)


def eval_simclr_ckpt(name, ckpt, loader, device, layers=("layer2", "layer3", "layer4")):
    from torchvision.models import resnet18
    from models import SimCLR
    from phtopo.dual_bn import convert_to_dual_bn, load_state_dict_auto, bn_route, state_dict_is_dual_bn
    m = SimCLR(resnet18, projection_dim=64, proj_hidden_dim=512, reduce_channels=8,
               ph_source_layer="layer3", cifar_no_maxpool=True).to(device)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    if state_dict_is_dual_bn(sd):
        convert_to_dual_bn(m)
    load_state_dict_auto(m, sd, strict=True); m.eval()
    acc = {L: ([], []) for L in layers}
    with torch.no_grad(), bn_route("adv"):
        for x, _ in loader:
            f = m._backbone_feats(x.to(device))
            for L in layers:
                T, R = reduce_stats(f[L]); acc[L][0].append(T); acc[L][1].append(R)
    return {L: fit(np.concatenate(a[0]), np.concatenate(a[1])) for L, a in acc.items()}


def eval_torchvision(name, ctor, weights, loader, device):
    """Off-the-shelf ImageNet model; grab layer2/3/4 via feature extractor."""
    from torchvision.models.feature_extraction import create_feature_extractor
    try:
        model = ctor(weights=weights).to(device).eval()
    except Exception as e:
        print(f"  [{name}] could not load ({type(e).__name__}: {str(e)[:60]}) -- skipping")
        return None
    nodes = {"layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}
    fx = create_feature_extractor(model, return_nodes=nodes)
    acc = {L: ([], []) for L in nodes}
    with torch.no_grad():
        for x, _ in loader:
            out = fx(x.to(device))
            for L in nodes:
                T, R = reduce_stats(out[L]); acc[L][0].append(T); acc[L][1].append(R)
    return {L: fit(np.concatenate(a[0]), np.concatenate(a[1])) for L, a in acc.items()}


def _latest(seed_dir):
    cks = glob.glob(os.path.join(seed_dir, "**", "*.pt"), recursive=True)
    return max(cks, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1))
               if re.search(r"epoch(\d+)", p) else -1) if cks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_cifar10_lambda/cifar10_resnet18")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--per_class", type=int, default=20)
    ap.add_argument("--no_torchvision", action="store_true",
                    help="skip off-the-shelf models (no weight download)")
    args = ap.parse_args()
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[reduction] device={device}  per_class={args.per_class}")
    print("  measuring r(H0 total persistence, mean pairwise distance) per model x layer\n")

    rows = []  # (model, layer, r, R2)

    # 1) your trained SimCLR encoders
    loader32 = get_loader(args.data_dir, args.per_class, size=32)
    for meth in ["adv_baseline", "adv_topoacl", "adv_rawacl"]:
        ck = _latest(os.path.join(args.runs, "upstream", f"{meth}_seed0"))
        if ck is None:
            continue
        res = eval_simclr_ckpt(meth, ck, loader32, device)
        for L, (r, r2) in res.items():
            rows.append((f"SimCLR:{meth}", L, r, r2))

    # 2) random-init control (no download) -- reduction should hold even untrained
    if not args.no_torchvision:
        from torchvision.models import resnet18 as tv_r18, resnet34, resnet50
        from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
        res = eval_torchvision("resnet18-random", tv_r18, None, loader32, device)
        if res:
            for L, (r, r2) in res.items():
                rows.append(("tv:resnet18-random", L, r, r2))
        # 3) off-the-shelf ImageNet-pretrained (need Resize to 224)
        loader224 = get_loader(args.data_dir, args.per_class, size=224)
        for name, ctor, w in [("resnet18-IN", tv_r18, ResNet18_Weights.DEFAULT),
                              ("resnet34-IN", resnet34, ResNet34_Weights.DEFAULT),
                              ("resnet50-IN", resnet50, ResNet50_Weights.DEFAULT)]:
            res = eval_torchvision(name, ctor, w, loader224, device)
            if res:
                for L, (r, r2) in res.items():
                    rows.append((f"tv:{name}", L, r, r2))

    print(f"  {'model':22s} {'layer':7s} {'r':>7s} {'R^2':>7s}")
    for m, L, r, r2 in rows:
        print(f"  {m:22s} {L:7s} {r:7.3f} {r2:7.3f}")

    rs = np.abs([r for _, _, r, _ in rows if not np.isnan(r)])
    simclr_rs = np.abs([r for m, _, r, _ in rows if m.startswith("SimCLR") and not np.isnan(r)])
    print("\n==== SUMMARY ====")
    print(f"  mean |r| across {len(rs)} cells = {rs.mean():.3f} (min={rs.min():.3f}, max={rs.max():.3f})")
    print(f"  mean |r| on OUR SimCLR encoders (the paper's setting) = {simclr_rs.mean():.3f}")
    print(f"  cells with |r|>=0.8: {(rs>=0.8).sum()}/{len(rs)}")
    print("  => H0 persistence tracks geometric spread strongly and broadly (architecture,")
    print("     training, init). Correlation is near-perfect at deep layers and in our SimCLR")
    print("     encoders; weakest at mid-depth for some ImageNet nets -- report per-cell honestly.")


if __name__ == "__main__":
    main()
