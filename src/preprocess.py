"""src/preprocess.py – dataset downloading & DataLoader creation."""
from __future__ import annotations

import shutil
import tarfile
from typing import Tuple

import pathlib
import requests
import torch
import torchvision.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader


HF_URL = "https://huggingface.co/datasets/mlx-vision/imagenet-1k/resolve/main/{split}/%06d.tar"
N_SHARDS = {"train": 1280, "val": 50}
DATA_ROOT = pathlib.Path("data/imagenet_wds")
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _download_split(split: str) -> None:  # pragma: no cover – network IO
    split_dir = DATA_ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for shard_id in range(N_SHARDS[split]):
        fname = split_dir / f"{shard_id:06d}.tar"
        if fname.exists():
            continue
        url = HF_URL.format(split=split) % shard_id
        print(f"[imagenet] downloading {url}")
        r = requests.get(url, stream=True, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download shard {shard_id} of {split} (HTTP {r.status_code})")
        with open(fname, "wb") as f:
            shutil.copyfileobj(r.raw, f)


def build_dataloader(global_batch: int, img_size: int) -> Tuple[DataLoader, DataLoader]:
    """Return train & validation loaders backed by WebDataset streams."""
    for split in ("train", "val"):
        _download_split(split)

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size), antialias=True),
        T.Normalize(0.5, 0.5),  # [0,1] -> [-1,1]
    ])

    def _make_loader(split: str) -> DataLoader:  # inner helper
        pattern = str(DATA_ROOT / split / "{000000..%06d}.tar" % (N_SHARDS[split] - 1))
        dataset = (
            wds.WebDataset(pattern, resampled=True)
            .decode("pil")
            .to_tuple("jpg", "cls")
            .map_tuple(transform, lambda y: torch.tensor(y, dtype=torch.int64))
        )
        world = torch.cuda.device_count() or 1
        return DataLoader(dataset.batched(global_batch // world), batch_size=None, num_workers=8, pin_memory=True)

    return _make_loader("train"), _make_loader("val")
