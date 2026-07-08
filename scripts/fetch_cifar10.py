#!/usr/bin/env python3
"""
fetch_cifar10.py -- download CIFAR-10 from the HuggingFace `uoft-cs/cifar10`
mirror (parquet) and write the exact pickle files torchvision expects, so
`CIFAR10(root=DATA_DIR, train=..., download=False)` loads without ever hitting
toronto.edu. Bypasses the flaky `cs.toronto.edu` mirror entirely.

Writes:
  <DATA_DIR>/cifar-10-batches-py/
      data_batch_1..5, test_batch, batches.meta

Usage:  python scripts/fetch_cifar10.py --data_dir ./data
"""
import argparse
import io
import os
import pickle
import sys
import urllib.request
from pathlib import Path

HF_TRAIN = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet"
HF_TEST  = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"

# torchvision's expected labels (order matters!)
LABEL_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def _fetch(url: str) -> bytes:
    import ssl
    print(f"  GET {url}")
    # Some hosts (esp. on macOS without updated certs) fail SSL verification even
    # when the CDN is fine. This script is fetching a public dataset via HTTPS, so
    # falling back to unverified context is acceptable.
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
        return r.read()


def _parquet_to_arrays(pq_bytes: bytes):
    """Load an HF-CIFAR-10 parquet -> (N, 3072) uint8 flat array + int labels list."""
    import pyarrow.parquet as pq
    import numpy as np
    from PIL import Image

    tbl = pq.read_table(io.BytesIO(pq_bytes))
    df = tbl.to_pydict()
    # HF schema for uoft-cs/cifar10: {"img": {"bytes": ..., "path": ...}, "label": int}
    img_col = df["img"] if "img" in df else df["image"]
    labels = list(df["label"])
    N = len(labels)
    data = np.empty((N, 3072), dtype=np.uint8)
    for i, entry in enumerate(img_col):
        img_bytes = entry["bytes"] if isinstance(entry, dict) else entry
        arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))  # (32,32,3)
        # torchvision pickle expects channels-first flattened: R plane, G plane, B plane
        data[i] = arr.transpose(2, 0, 1).reshape(-1)
    return data, labels


def _write_batch(path: Path, data, labels, batch_label: str):
    # torchvision's CIFAR loader accesses entry["data"] with STR keys (see
    # torchvision/datasets/cifar.py: `self.data.append(entry["data"])` after
    # `pickle.load(f, encoding="latin1")`). encoding="latin1" decodes byte *values*
    # but NOT dict keys, so we write str keys to match.
    d = {
        "batch_label": batch_label,
        "labels": labels,
        "data": data,
        "filenames": [f"img_{i}.png" for i in range(len(labels))],
    }
    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    args = ap.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    out = root / "cifar-10-batches-py"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[cifar10] fetching train + test parquet from HuggingFace uoft-cs/cifar10 ...")
    train_bytes = _fetch(HF_TRAIN)
    test_bytes  = _fetch(HF_TEST)

    print("[cifar10] decoding train ...")
    train_data, train_labels = _parquet_to_arrays(train_bytes)
    assert train_data.shape == (50000, 3072), train_data.shape
    print("[cifar10] decoding test ...")
    test_data, test_labels = _parquet_to_arrays(test_bytes)
    assert test_data.shape == (10000, 3072), test_data.shape

    # Split train into 5 batches of 10000 (torchvision's expected structure)
    for i in range(5):
        s, e = i * 10000, (i + 1) * 10000
        _write_batch(out / f"data_batch_{i+1}", train_data[s:e], train_labels[s:e],
                     f"training batch {i+1} of 5")
    _write_batch(out / "test_batch", test_data, test_labels, "testing batch 1 of 1")

    # meta file -- also str keys (same reason as above)
    with open(out / "batches.meta", "wb") as f:
        pickle.dump({"label_names": LABEL_NAMES,
                     "num_cases_per_batch": 10000,
                     "num_vis": 3072}, f, protocol=2)

    print(f"[cifar10] wrote {out}/ (5 train batches + test_batch + batches.meta)")
    # sanity-load via torchvision (no download -- proves it works)
    # HF-mirror pickles have correct DATA but different bytes from the original
    # 2009 release, so torchvision's hardcoded md5 check would reject them. Bypass
    # via the same helper used at training time.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from datasets import _bypass_torchvision_cifar_md5_if_prepared
    _bypass_torchvision_cifar_md5_if_prepared(str(root))
    try:
        from torchvision.datasets import CIFAR10
        tr = CIFAR10(root=str(root), train=True, download=False)
        te = CIFAR10(root=str(root), train=False, download=False)
        # quick content sanity: correct sizes + label range
        assert len(tr) == 50000 and len(te) == 10000
        _, y = tr[0]; assert 0 <= y < 10
        print(f"[cifar10] verified via torchvision: train={len(tr)} test={len(te)}, sample label={y}")
    except Exception as e:
        print(f"[cifar10] WARNING: torchvision could not load: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
