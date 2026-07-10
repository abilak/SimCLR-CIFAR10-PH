#!/usr/bin/env python3
"""
multiscale_screen.py -- NO-TRAINING screen for whether multiscale PH (layer3+layer2)
could plausibly help robustness, run on EXISTING single-scale checkpoints.

Logic: multiscale can only help if the layer2 topological signal is
  (1) NOT redundant with layer3,          -> corr(topo_l2, topo_l3) low
  (2) topology-SPECIFIC (not raw geometry),-> corr(topo_lX, raw_lX) low, esp. layer2
  (3) carrying independent under-attack info-> corr(Dtopo_l2, Dtopo_l3) low
All three are measurable on frozen features -- no multiscale model is trained.

We use RAW backbone feature maps at each layer (a superset of anything a learned 1x1
reduction could extract): if the signal isn't in the raw features, no reduction makes it.
Per image, per layer, the feature map (C,H,W) is a point cloud of H*W points in R^C:
  topo descriptor  T_topo = H0 total persistence (sum of MST edge weights)
  raw  descriptor  T_raw  = mean pairwise distance (the swcontrol/rawacl analog)

This is a SCREEN, not a verdict: a clear "no signal" (redundant + topo~=raw at layer2)
is decisive AGAINST multiscale; a "yes signal" is only suggestive (training may or may
not exploit it).

Usage: python utils/multiscale_screen.py --runs runs_cifar10_lambda/cifar10_resnet18 \
         --data_dir ./data --per_class 20
"""
import argparse, glob, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision.models import resnet18
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from models import SimCLR
from phtopo.dual_bn import convert_to_dual_bn, load_state_dict_auto, bn_route, state_dict_is_dual_bn
from datasets import _bypass_torchvision_cifar_md5_if_prepared


def _max_epoch_ckpt(seed_dir):
    cks = glob.glob(os.path.join(seed_dir, "**", "*.pt"), recursive=True)
    if not cks:
        return None
    def ep(p):
        m = re.search(r"epoch(\d+)", p)
        return int(m.group(1)) if m else -1
    return max(cks, key=ep)


def build_model(device):
    m = SimCLR(resnet18, projection_dim=64, proj_hidden_dim=512,
               reduce_channels=8, ph_source_layer="layer3", cifar_no_maxpool=True)
    return m.to(device)


def load_ckpt(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    if state_dict_is_dual_bn(sd):
        convert_to_dual_bn(model)
    load_state_dict_auto(model, sd, strict=True)
    return int(ck.get("epoch", -1))


def pgd_rep_attack(model, x, eps, alpha, steps):
    """Label-free PGD: maximize 1 - cos(rep_adv, rep_clean). Matches eval attack."""
    x0 = x.detach()
    with torch.no_grad():
        rep_clean = model(x0)[2].detach()
    x_adv = x0.clone().detach()
    # random start
    x_adv = torch.clamp(x_adv + torch.empty_like(x_adv).uniform_(-eps, eps), 0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        rep = model(x_adv)[2]
        loss = (1.0 - F.cosine_similarity(rep, rep_clean, dim=1)).mean()
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)
    return x_adv.detach()


@torch.no_grad()
def feats_l2_l3(model, x):
    """Raw backbone feature maps at layer2 and layer3 (one forward, up to layer3)."""
    feats = model._backbone_feats_upto(x, ["layer3"])
    return feats["layer2"], feats["layer3"]   # (B,128,H2,W2), (B,256,H3,W3)


def h0_total_persistence(pts):
    """H0 total persistence of a point cloud = sum of Euclidean MST edge weights."""
    n = pts.shape[0]
    if n < 2:
        return 0.0
    D = squareform(pdist(pts.astype(np.float64)))
    mst = minimum_spanning_tree(D)
    return float(mst.sum())


def mean_pairwise_dist(pts):
    if pts.shape[0] < 2:
        return 0.0
    return float(pdist(pts.astype(np.float64)).mean())


def descriptors(fmap):
    """Per-image (topo, raw) over a batch of feature maps (B,C,H,W)."""
    B = fmap.shape[0]
    topo = np.empty(B); raw = np.empty(B)
    f = fmap.detach().cpu().numpy()
    for i in range(B):
        pts = f[i].reshape(f[i].shape[0], -1).T   # (H*W, C) point cloud
        topo[i] = h0_total_persistence(pts)
        raw[i] = mean_pairwise_dist(pts)
    return topo, raw


def pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_cifar10_lambda/cifar10_resnet18")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--method", default="adv_baseline")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--per_class", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--eps_px", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[screen] device={device}  method={args.method}  per_class={args.per_class}")
    eps = args.eps_px / 255.0
    alpha = 2.5 * eps / args.steps

    _bypass_torchvision_cifar_md5_if_prepared(args.data_dir)
    test = CIFAR10(root=args.data_dir, train=False, download=False,
                   transform=transforms.ToTensor())
    # balanced subset
    by = {c: [] for c in range(10)}
    for idx in range(len(test)):
        _, y = test[idx]
        if len(by[y]) < args.per_class:
            by[y].append(idx)
        if all(len(v) >= args.per_class for v in by.values()):
            break
    idxs = [i for c in range(10) for i in by[c]]
    loader = DataLoader(Subset(test, idxs), batch_size=args.batch_size, shuffle=False)

    seeds = [int(s) for s in args.seeds.split(",")]
    agg = {k: [] for k in ["redundancy", "spec_l2", "spec_l3",
                            "attack_indep", "relchg_l2", "relchg_l3"]}
    shapes_printed = False

    for seed in seeds:
        seed_dir = os.path.join(args.runs, "upstream", f"{args.method}_seed{seed}")
        ck = _max_epoch_ckpt(seed_dir)
        if ck is None:
            print(f"  seed{seed}: NO checkpoint under {seed_dir}, skipping")
            continue
        model = build_model(device)
        ep = load_ckpt(model, ck)
        model.eval()
        print(f"  seed{seed}: loaded epoch{ep}  ({os.path.basename(ck)})")

        T2c, R2c, T3c, R3c = [], [], [], []
        T2a, T3a = [], []
        with bn_route("adv"):    # match robustness eval branch
            for x, _ in loader:
                x = x.to(device)
                x_adv = pgd_rep_attack(model, x, eps, alpha, args.steps)
                f2c, f3c = feats_l2_l3(model, x)
                f2a, f3a = feats_l2_l3(model, x_adv)
                if not shapes_printed:
                    print(f"    layer2 map {tuple(f2c.shape)} -> {f2c.shape[2]*f2c.shape[3]} pts/"
                          f"{f2c.shape[1]}d ; layer3 map {tuple(f3c.shape)} -> "
                          f"{f3c.shape[2]*f3c.shape[3]} pts/{f3c.shape[1]}d")
                    shapes_printed = True
                t2c, r2c = descriptors(f2c); t3c, r3c = descriptors(f3c)
                t2a, _ = descriptors(f2a);   t3a, _ = descriptors(f3a)
                T2c += list(t2c); R2c += list(r2c); T3c += list(t3c); R3c += list(r3c)
                T2a += list(t2a); T3a += list(t3a)

        T2c, R2c, T3c, R3c = map(np.array, (T2c, R2c, T3c, R3c))
        T2a, T3a = map(np.array, (T2a, T3a))
        d2, d3 = T2a - T2c, T3a - T3c
        m = dict(
            redundancy=pearson(T2c, T3c),
            spec_l2=pearson(T2c, R2c),
            spec_l3=pearson(T3c, R3c),
            attack_indep=pearson(d2, d3),
            relchg_l2=float(np.mean(np.abs(d2)) / (np.mean(np.abs(T2c)) + 1e-9)),
            relchg_l3=float(np.mean(np.abs(d3)) / (np.mean(np.abs(T3c)) + 1e-9)),
        )
        for k, v in m.items():
            agg[k].append(v)
        print(f"    redundancy(l2,l3)={m['redundancy']:+.3f}  "
              f"spec_l2(topo~raw)={m['spec_l2']:+.3f}  spec_l3={m['spec_l3']:+.3f}  "
              f"attack_indep={m['attack_indep']:+.3f}  "
              f"relchg l2={m['relchg_l2']:.3f} l3={m['relchg_l3']:.3f}")

    print("\n==== MEAN OVER SEEDS ====")
    mean = {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in agg.items()}
    for k in ["redundancy", "spec_l2", "spec_l3", "attack_indep", "relchg_l2", "relchg_l3"]:
        print(f"  {k:14s} = {mean[k]:+.3f}")

    # ---- verdict -------------------------------------------------------------
    print("\n==== VERDICT ====")
    reasons = []
    # (1) redundancy: high |corr| between scales -> layer2 adds little new
    redundant = abs(mean["redundancy"]) >= 0.8
    # (2) specificity: layer2 topology ~ layer2 raw geometry -> same topo~=raw null
    l2_not_specific = abs(mean["spec_l2"]) >= 0.8
    # (3) independent under-attack signal
    no_indep_attack = abs(mean["attack_indep"]) >= 0.8

    if redundant:
        reasons.append(f"layer2/layer3 topology redundant (|r|={abs(mean['redundancy']):.2f}>=0.8)")
    if l2_not_specific:
        reasons.append(f"layer2 topology ~ raw geometry (|r|={abs(mean['spec_l2']):.2f}>=0.8): same topo=raw null")
    if no_indep_attack:
        reasons.append(f"layer2 under-attack change tracks layer3 (|r|={abs(mean['attack_indep']):.2f}>=0.8)")

    if l2_not_specific or (redundant and no_indep_attack):
        print("  SKIP multiscale. " + " ; ".join(reasons))
        print("  -> layer2 carries no topology-specific, independent signal to exploit.")
    elif not redundant and not l2_not_specific:
        print("  MAYBE worth the real run: layer2 topology is independent of layer3 AND")
        print(f"     more topology-specific than raw (spec_l2={mean['spec_l2']:+.2f}).")
        print("  (screen shows signal EXISTS; training may or may not capture it.)")
    else:
        print("  AMBIGUOUS -- mixed signals: " + " ; ".join(reasons or ["borderline"]))
    print("\n(reminder: 'no signal' is decisive against; 'signal' is only suggestive.)")


if __name__ == "__main__":
    main()
