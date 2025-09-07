from __future__ import annotations

"""Data-loading & pre-processing helpers (mini-ImageNet from HuggingFace)."""

import pathlib
from typing import Tuple

import datasets as hfd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

__all__ = ["build_dataloaders", "DataDownloadError"]

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Utilities (local to this file to satisfy the 6-file constraint)
# -----------------------------------------------------------------------------


def _sha256_of_file(path: pathlib.Path, chunk_size: int = 1_048_576) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: pathlib.Path, expected: str) -> None:
    actual = _sha256_of_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class DataDownloadError(RuntimeError):
    """Raised when HuggingFace dataset download / verification fails."""


# -----------------------------------------------------------------------------
# Internal dataset wrapper
# -----------------------------------------------------------------------------


class _HFDataset(Dataset):
    def __init__(self, split, img_size: int):
        self.split = split
        self.transform = T.Compose(
            [
                T.Resize(img_size + 16, antialias=True),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        record = self.split[int(idx)]
        img = record.get("img", record.get("image")).convert("RGB")
        return {"x": self.transform(img)}


# -----------------------------------------------------------------------------
# Public helper
# -----------------------------------------------------------------------------


def build_dataloaders(cfg: dict, seed: int) -> Tuple[DataLoader, DataLoader]:
    """Download (if needed) and build deterministic train/val dataloaders."""

    try:
        ds = hfd.load_dataset(cfg["hf_repo"], cache_dir=str(DATA_DIR))
    except Exception as e:  # pragma: no cover – network issues
        raise DataDownloadError(f"Could not download dataset: {e}") from e

    # Optional SHA-256 verification (offline integrity check)
    sha_file = cfg.get("sha256_file")
    if sha_file and pathlib.Path(sha_file).exists():
        for line in pathlib.Path(sha_file).read_text().splitlines():
            expected, rel = line.strip().split()[:2]
            _verify_sha256(DATA_DIR / rel, expected)

    split = ds["train"].train_test_split(test_size=cfg["val_split"], seed=seed)
    train_ds = _HFDataset(split["train"], img_size=cfg["img_size"])
    val_ds = _HFDataset(split["test"], img_size=cfg["img_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["global_batch"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["global_batch"], shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader
