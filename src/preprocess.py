"""src/preprocess.py
Dataset download & preprocessing utilities (HuggingFace 🤗 Datasets + TorchVision).
This keeps the heavy I/O logic out of *train.py* / *evaluate.py*.
"""
from __future__ import annotations

from typing import Tuple

import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class DataDownloadError(RuntimeError):
    """Raised when the dataset could not be downloaded (no silent fallback)."""


class _HFDataset(Dataset):
    def __init__(self, hf_split, *, img_size: int, train: bool):
        self.ds = hf_split
        if train:
            self.tx = T.Compose(
                [
                    T.Resize(img_size + 16, antialias=True),
                    T.RandomCrop(img_size),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([-1, -1, -1], [2, 2, 2]),
                ]
            )
        else:
            self.tx = T.Compose(
                [
                    T.Resize(img_size + 16, antialias=True),
                    T.CenterCrop(img_size),
                    T.ToTensor(),
                    T.Normalize([-1, -1, -1], [2, 2, 2]),
                ]
            )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[int(idx)]["image"].convert("RGB")
        return {"x": self.tx(img)}


# -----------------------------------------------------------------------------
#  PUBLIC API
# -----------------------------------------------------------------------------

def build_dataloaders(ds_cfg: dict, train_cfg: dict, seed: int):
    """Download (if necessary) & return train-dataloader and validation dataset."""

    try:
        dataset = datasets.load_dataset(ds_cfg["hf_repo"], cache_dir="data")
    except Exception as e:  # noqa: BLE001
        raise DataDownloadError(str(e)) from e

    train_split = dataset["train"]
    val_split = dataset[ds_cfg.get("val_split_name", "validation")]

    train_ds = _HFDataset(train_split, img_size=ds_cfg["img_size"], train=True)
    val_ds = _HFDataset(val_split, img_size=ds_cfg["img_size"], train=False)

    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg["batch"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    return train_dl, val_ds
