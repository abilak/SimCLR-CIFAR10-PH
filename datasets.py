"""
datasets.py -- dataset registry for multi-dataset PH-ACL experiments.

Centralizes the per-dataset knobs (#classes, SSL image size, and how to build the
SSL two-views training set + the labeled eval sets) so simclr.py, simclr_lin.py,
eval_robustness.py and eval_mechanism.py stay dataset-agnostic. The `dataset`
config/CLI key selects one; default `cifar10` reproduces the original pipeline
byte-for-byte.

Threat model is unchanged across datasets: attacks operate in [0,1] (ToTensor
only, no normalization), so robustness numbers stay comparable.

Supported: cifar10, cifar100 (both 32x32), stl10 (96x96, SSL on the unlabeled
split). Add a new dataset by extending _REGISTRY and the two factory functions.
"""
from __future__ import annotations

import os

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10


def _bypass_torchvision_cifar_md5_if_prepared(root: str) -> None:
    """
    Skip torchvision's per-file md5 check when a valid cifar-10-batches-py/ has
    already been prepared locally (e.g., by scripts/fetch_cifar10.py, which pulls
    from the HuggingFace uoft-cs/cifar10 mirror when the canonical toronto host is
    unreachable). We regenerate the batches from an equivalent source and torchvision's
    hardcoded md5s only match the original 2009 pickle bytes -- so the *data* is
    correct but the file hashes differ. Monkey-patch _check_integrity to accept the
    prepared layout so torchvision loads without demanding a re-download it can't do.

    Only patches when the extracted dir + all expected batch files exist. Otherwise a
    no-op (normal md5 check still runs).
    """
    for cls, subdir, files in [
        (CIFAR10, "cifar-10-batches-py",
         [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]),
        (CIFAR100, "cifar-100-python", ["train", "test", "meta"]),
    ]:
        base = os.path.join(root, subdir)
        if all(os.path.exists(os.path.join(base, f)) for f in files):
            cls._check_integrity = lambda self: True   # trust batch files
            # _load_meta() also md5-checks the meta file separately; clear it too.
            if isinstance(cls.meta, dict):
                cls.meta = {**cls.meta, "md5": None}


# Resolution is baked into the dataset NAME (e.g. stl10_64) so it flows through
# training, eval, and the checkpoint config identically -- no separate image_size
# knob that could desync train vs eval. `loader` picks the torchvision class.
_REGISTRY = {
    "cifar10":   dict(n_classes=10,  image_size=32, loader="cifar10"),
    "cifar100":  dict(n_classes=100, image_size=32, loader="cifar100"),
    "stl10":     dict(n_classes=10,  image_size=96, loader="stl10"),   # native res
    "stl10_64":  dict(n_classes=10,  image_size=64, loader="stl10"),   # downsized (compute/budget)
    "stl10_32":  dict(n_classes=10,  image_size=32, loader="stl10"),
}


def dataset_info(name: str) -> dict:
    name = str(name).lower()
    if name not in _REGISTRY:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def num_classes(name: str) -> int:
    return dataset_info(name)["n_classes"]


def image_size(name: str) -> int:
    return dataset_info(name)["image_size"]


def _color_distortion(s: float = 0.5):
    """Color jitter + grayscale (SimCLR appendix). Local copy to avoid a circular
    import with simclr.py; identical to simclr.get_color_distortion."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return transforms.Compose([
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])


def ssl_train_transform(name: str, color_strength: float = 0.5):
    """The SimCLR two-views augmentation, sized to the dataset (crop = image_size)."""
    sz = image_size(name)
    return transforms.Compose([
        transforms.RandomResizedCrop(sz),
        transforms.RandomHorizontalFlip(p=0.5),
        _color_distortion(s=color_strength),
        transforms.ToTensor(),  # -> [0,1]; attacks live here
    ])


class TwoView(torch.utils.data.Dataset):
    """
    Wrap ANY (image, label) dataset and return two independently-augmented views,
    interleaved by the collate into the [v1, v2, v1, v2, ...] layout the losses
    expect. The base dataset must yield a PIL image (we set its transform=None and
    apply ours), which CIFAR10/100 and STL10 all do.
    """
    def __init__(self, base, transform):
        self.base = base
        self.base.transform = None  # we apply our own (two views)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]   # PIL image, label
        return torch.stack([self.transform(img), self.transform(img)]), target


def make_ssl_trainset(name: str, root: str, color_strength: float = 0.5, download: bool = True):
    """SSL pretraining set (two views). For STL-10 this is the large unlabeled split
    (train+unlabeled); for CIFAR it is the train split. Crop sized to image_size(name)."""
    _bypass_torchvision_cifar_md5_if_prepared(root)
    loader = dataset_info(name)["loader"]
    tf = ssl_train_transform(name, color_strength)
    if loader == "stl10":
        base = STL10(root=root, split="train+unlabeled", transform=None, download=download)
    elif loader == "cifar100":
        base = CIFAR100(root=root, train=True, transform=None, download=download)
    else:
        base = CIFAR10(root=root, train=True, transform=None, download=download)
    return TwoView(base, tf)


def make_eval_sets(name: str, root: str, download: bool = True):
    """
    Labeled (train, test) sets in [0,1] for the linear probe + attacks. Images are
    Resized to image_size(name) so eval matches the resolution the encoder was
    trained at (critical for the downsized stl10_64/stl10_32 variants -- a 96px
    eval on a 64px-trained encoder would silently tank robustness). For native-res
    datasets the Resize is a no-op.
    """
    _bypass_torchvision_cifar_md5_if_prepared(root)
    info = dataset_info(name)
    loader, sz = info["loader"], info["image_size"]
    tf = transforms.Compose([transforms.Resize(sz), transforms.ToTensor()])
    if loader == "stl10":
        train = STL10(root=root, split="train", transform=tf, download=download)
        test = STL10(root=root, split="test", transform=tf, download=download)
    elif loader == "cifar100":
        train = CIFAR100(root=root, train=True, transform=tf, download=download)
        test = CIFAR100(root=root, train=False, transform=tf, download=download)
    else:
        train = CIFAR10(root=root, train=True, transform=tf, download=download)
        test = CIFAR10(root=root, train=False, transform=tf, download=download)
    return train, test


def label_of(dataset, idx) -> int:
    """Label for a sample, robust across CIFAR (.targets) and STL10 (.labels)."""
    if hasattr(dataset, "targets"):
        return int(dataset.targets[idx])
    if hasattr(dataset, "labels"):
        return int(dataset.labels[idx])
    return int(dataset[idx][1])
