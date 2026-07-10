#!/usr/bin/env python3
"""
topo_detect.py -- PIVOT A: adversarial-example DETECTION via activation topology,
run on EXISTING checkpoints. Different claim from robust training: we don't need
topology to be attack-invariant, we exploit whatever consistent topological SHIFT
an attack induces as a detection signal (cf. Gebhart & Schrater; Lacombe et al.).

The decisive, control-disciplined question (same as the whole project):
  does a TOPOLOGY detector beat a RAW-GEOMETRY detector at clean-vs-adv detection?

Protocol
--------
For balanced CIFAR-10 test images, build clean + PGD-adversarial (rep-space,
label-free) versions. At layers {layer2,layer3,layer4} treat each activation map
(C,H,W) as a point cloud (H*W points in R^C) and extract:
  * TOPO features  : H0 persistence-diagram summaries (sum/mean/std/max/entropy of
                     MST death times + a fixed-bin death histogram)   [per layer]
  * RAW  features  : geometry of the SAME cloud (pairwise-dist stats, point-norm
                     stats, centroid norm)                             [per layer]
Train logistic regression clean-vs-adv on a train split; report held-out ROC-AUC.
Compare TOPO vs RAW vs COMBINED vs a trivial baseline (global feature L2 norm).

Verdict: TOPO AUC clearly > RAW AUC  -> topology-specific detection win (positive
pivot). TOPO ~= RAW -> detection works but isn't topology-specific. Both ~0.5 ->
no detection signal.

Limitation (spike): attack is rep-space PGD on the SSL encoder, not an attack on a
downstream classifier. A real detection paper would attack the deployed task head.

Usage: python utils/topo_detect.py --runs runs_cifar10_lambda/cifar10_resnet18 \
         --data_dir ./data --per_class 50
"""
import argparse, glob, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision.models import resnet18
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from models import SimCLR
from phtopo.dual_bn import convert_to_dual_bn, load_state_dict_auto, bn_route, state_dict_is_dual_bn
from datasets import _bypass_torchvision_cifar_md5_if_prepared

LAYERS = ["layer2", "layer3", "layer4"]
N_DEATH_BINS = 8


def _max_epoch_ckpt(seed_dir):
    cks = glob.glob(os.path.join(seed_dir, "**", "*.pt"), recursive=True)
    if not cks:
        return None
    return max(cks, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1))
               if re.search(r"epoch(\d+)", p) else -1)


def build_model(device):
    return SimCLR(resnet18, projection_dim=64, proj_hidden_dim=512, reduce_channels=8,
                  ph_source_layer="layer3", cifar_no_maxpool=True).to(device)


def load_ckpt(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    if state_dict_is_dual_bn(sd):
        convert_to_dual_bn(model)
    load_state_dict_auto(model, sd, strict=True)
    return int(ck.get("epoch", -1))


def pgd_rep_attack(model, x, eps, alpha, steps):
    x0 = x.detach()
    with torch.no_grad():
        rep_clean = model(x0)[2].detach()
    x_adv = torch.clamp(x0 + torch.empty_like(x0).uniform_(-eps, eps), 0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = (1.0 - F.cosine_similarity(model(x_adv)[2], rep_clean, dim=1)).mean()
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * g.sign()
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps).clamp(0, 1)
    return x_adv.detach()


def _h0_deaths(pts):
    """H0 persistence diagram death times = Euclidean MST edge weights (births=0)."""
    if pts.shape[0] < 2:
        return np.zeros(1)
    D = squareform(pdist(pts.astype(np.float64)))
    mst = minimum_spanning_tree(D).toarray()
    d = mst[mst > 0]
    return d if d.size else np.zeros(1)


def _entropy(w):
    p = w / (w.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def topo_feats(pts):
    d = _h0_deaths(pts)
    dn = d / (d.max() + 1e-9)
    hist, _ = np.histogram(dn, bins=N_DEATH_BINS, range=(0, 1), density=False)
    hist = hist / (hist.sum() + 1e-9)
    return np.concatenate([[d.sum(), d.mean(), d.std(), d.max(), _entropy(d), float(len(d))], hist])


def raw_feats(pts):
    """Geometry of the SAME cloud -- matched-richness raw-geometry control."""
    dists = pdist(pts.astype(np.float64)) if pts.shape[0] >= 2 else np.zeros(1)
    norms = np.linalg.norm(pts, axis=1)
    centroid = pts.mean(0)
    q = np.quantile(dists, [0.1, 0.5, 0.9])
    return np.array([dists.mean(), dists.std(), dists.max(), dists.min(),
                     q[0], q[1], q[2], norms.mean(), norms.std(),
                     float(np.linalg.norm(centroid))])


@torch.no_grad()
def extract(model, x):
    """Return per-layer (topo_vec, raw_vec) concatenated across LAYERS + global norm."""
    feats = model._backbone_feats(x)
    B = x.shape[0]
    T = [[] for _ in range(B)]
    R = [[] for _ in range(B)]
    gnorm = np.zeros(B)
    for L in LAYERS:
        f = feats[L].detach().cpu().numpy()
        for i in range(B):
            pts = f[i].reshape(f[i].shape[0], -1).T
            T[i].append(topo_feats(pts))
            R[i].append(raw_feats(pts))
    gnorm = torch.flatten(feats["layer4"], 1).norm(dim=1).cpu().numpy()
    T = np.array([np.concatenate(t) for t in T])
    R = np.array([np.concatenate(r) for r in R])
    return T, R, gnorm


def auc_cv(X, y, seed):
    """5-split held-out mean ROC-AUC with a standardized logistic regressor."""
    aucs = []
    for k in range(5):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4,
                                              random_state=seed * 10 + k, stratify=y)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Xtr, ytr)
        aucs.append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_cifar10_lambda/cifar10_resnet18")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--method", default="adv_baseline")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--per_class", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--eps_px", type=float, default=8.0)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[detect] device={device} method={args.method} per_class={args.per_class} "
          f"eps={args.eps_px}/255 steps={args.steps}")
    eps = args.eps_px / 255.0
    alpha = 2.5 * eps / args.steps

    _bypass_torchvision_cifar_md5_if_prepared(args.data_dir)
    test = CIFAR10(root=args.data_dir, train=False, download=False,
                   transform=transforms.ToTensor())
    by = {c: [] for c in range(10)}
    for idx in range(len(test)):
        _, yy = test[idx]
        if len(by[yy]) < args.per_class:
            by[yy].append(idx)
        if all(len(v) >= args.per_class for v in by.values()):
            break
    idxs = [i for c in range(10) for i in by[c]]
    loader = DataLoader(Subset(test, idxs), batch_size=args.batch_size, shuffle=False)

    res = {k: [] for k in ["topo", "raw", "combined", "gnorm"]}
    for seed in [int(s) for s in args.seeds.split(",")]:
        ck = _max_epoch_ckpt(os.path.join(args.runs, "upstream", f"{args.method}_seed{seed}"))
        if ck is None:
            print(f"  seed{seed}: no checkpoint, skip"); continue
        model = build_model(device); ep = load_ckpt(model, ck); model.eval()
        print(f"  seed{seed}: epoch{ep}")

        Tc, Rc, Gc, Ta, Ra, Ga = [], [], [], [], [], []
        with bn_route("adv"):
            for x, _ in loader:
                x = x.to(device)
                xadv = pgd_rep_attack(model, x, eps, alpha, args.steps)
                tc, rc, gc = extract(model, x)
                ta, ra, ga = extract(model, xadv)
                Tc.append(tc); Rc.append(rc); Gc.append(gc)
                Ta.append(ta); Ra.append(ra); Ga.append(ga)
        Tc, Rc, Ta, Ra = map(np.vstack, (Tc, Rc, Ta, Ra))
        Gc, Ga = np.concatenate(Gc), np.concatenate(Ga)
        n = Tc.shape[0]
        y = np.concatenate([np.zeros(n), np.ones(n)])  # 0=clean 1=adv
        Xtopo = np.vstack([Tc, Ta]); Xraw = np.vstack([Rc, Ra])
        Xcomb = np.hstack([Xtopo, Xraw]); Xg = np.concatenate([Gc, Ga]).reshape(-1, 1)

        at = auc_cv(Xtopo, y, seed); ar = auc_cv(Xraw, y, seed)
        ac = auc_cv(Xcomb, y, seed); ag = auc_cv(Xg, y, seed)
        res["topo"].append(at[0]); res["raw"].append(ar[0])
        res["combined"].append(ac[0]); res["gnorm"].append(ag[0])
        print(f"    AUC  topo={at[0]:.3f}±{at[1]:.3f}  raw={ar[0]:.3f}±{ar[1]:.3f}  "
              f"combined={ac[0]:.3f}  gnorm-baseline={ag[0]:.3f}   (n={n} clean + {n} adv)")

    print("\n==== MEAN OVER SEEDS (ROC-AUC, clean vs adv) ====")
    for k in ["topo", "raw", "combined", "gnorm"]:
        v = res[k]
        print(f"  {k:9s} = {np.mean(v):.3f}" if v else f"  {k:9s} = n/a")

    print("\n==== VERDICT ====")
    if not res["topo"]:
        print("  no seeds ran"); return
    topo, raw = np.mean(res["topo"]), np.mean(res["raw"])
    if topo < 0.6 and raw < 0.6:
        print(f"  NO detection signal (topo={topo:.3f}, raw={raw:.3f} ~ chance).")
    elif topo >= raw + 0.05:
        print(f"  TOPOLOGY-SPECIFIC detection WIN: topo={topo:.3f} > raw={raw:.3f} "
              f"(+{topo-raw:.3f}). Positive pivot -- worth pursuing.")
    elif abs(topo - raw) < 0.05:
        print(f"  Detection works (topo={topo:.3f}) but NOT topology-specific "
              f"(raw={raw:.3f}). Same topo=raw pattern as robustness.")
    else:
        print(f"  RAW geometry detects better ({raw:.3f}) than topology ({topo:.3f}).")
    print("\n(spike: rep-space attack, not a downstream-classifier attack.)")


if __name__ == "__main__":
    main()
