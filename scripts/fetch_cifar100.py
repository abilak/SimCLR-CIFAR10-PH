#!/usr/bin/env python3
"""
fetch_cifar100.py -- CIFAR-100 from the HuggingFace uoft-cs/cifar100 parquet mirror,
written to the pickle layout torchvision's CIFAR100 expects. Same approach/rationale
as scripts/fetch_cifar10.py (bypasses the flaky toronto host; str keys; md5 bypass).

Usage: python scripts/fetch_cifar100.py --data_dir ./data
"""
import argparse, io, pickle, ssl, sys, urllib.request
from pathlib import Path

HF_TRAIN = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet"
HF_TEST  = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet"


def _fetch(url):
    print(f"  GET {url}")
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, timeout=180, context=ctx) as r:
        return r.read()


def _parquet(pq_bytes):
    import pyarrow.parquet as pq
    import numpy as np
    from PIL import Image
    d = pq.read_table(io.BytesIO(pq_bytes)).to_pydict()
    img_col = d["img"] if "img" in d else d["image"]
    fine = list(d["fine_label"]); coarse = list(d.get("coarse_label", [0] * len(fine)))
    N = len(fine)
    data = np.empty((N, 3072), dtype=np.uint8)
    for i, e in enumerate(img_col):
        b = e["bytes"] if isinstance(e, dict) else e
        arr = np.array(Image.open(io.BytesIO(b)).convert("RGB"))
        data[i] = arr.transpose(2, 0, 1).reshape(-1)
    return data, fine, coarse


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", default="./data")
    root = Path(ap.parse_args().data_dir).expanduser().resolve()
    out = root / "cifar-100-python"; out.mkdir(parents=True, exist_ok=True)

    print("[cifar100] fetching parquet from HF uoft-cs/cifar100 ...")
    trd, trf, trc = _parquet(_fetch(HF_TRAIN))
    ted, tef, tec = _parquet(_fetch(HF_TEST))
    assert trd.shape == (50000, 3072) and ted.shape == (10000, 3072), (trd.shape, ted.shape)

    # str keys (encoding="latin1" decodes values not keys) -- see fetch_cifar10.py
    with open(out / "train", "wb") as f:
        pickle.dump({"data": trd, "fine_labels": trf, "coarse_labels": trc,
                     "filenames": [f"img_{i}.png" for i in range(len(trf))],
                     "batch_label": "training"}, f, protocol=2)
    with open(out / "test", "wb") as f:
        pickle.dump({"data": ted, "fine_labels": tef, "coarse_labels": tec,
                     "filenames": [f"img_{i}.png" for i in range(len(tef))],
                     "batch_label": "testing"}, f, protocol=2)
    with open(out / "meta", "wb") as f:
        pickle.dump({"fine_label_names": [f"class_{i}" for i in range(100)],
                     "coarse_label_names": [f"super_{i}" for i in range(20)]}, f, protocol=2)
    print(f"[cifar100] wrote {out}/ (train + test + meta)")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from datasets import _bypass_torchvision_cifar_md5_if_prepared
    _bypass_torchvision_cifar_md5_if_prepared(str(root))
    from torchvision.datasets import CIFAR100
    tr = CIFAR100(root=str(root), train=True, download=False)
    te = CIFAR100(root=str(root), train=False, download=False)
    _, y = tr[0]
    assert len(tr) == 50000 and len(te) == 10000 and 0 <= y < 100
    print(f"[cifar100] verified via torchvision: train={len(tr)} test={len(te)}, sample label={y}")


if __name__ == "__main__":
    main()
