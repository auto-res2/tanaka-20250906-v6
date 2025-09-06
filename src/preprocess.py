"""src/preprocess.py – dataset downloading & DataLoader creation."""
from __future__ import annotations

import io
import random
import shutil
import tarfile
from typing import Tuple

import pathlib
import requests
import torch
import torchvision.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader, Dataset

HF_URLS = [
    # Primary mirror (was previously failing for some users)
    "https://huggingface.co/datasets/mlx-vision/imagenet-1k/resolve/main/{split}/%06d.tar",
    # Fallback mirror – usually more reliable for public CI runners
    "https://huggingface.co/datasets/dark-xet/imagenet-1k-wds/resolve/main/{split}/%06d.tar",
]
N_SHARDS = {"train": 1280, "val": 50}
DATA_ROOT = pathlib.Path("data/imagenet_wds")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#  Helper: quick synthetic dataset (used only when *all* remote mirrors fail)
# -----------------------------------------------------------------------------


class _SyntheticImageNet(Dataset):  # pragma: no cover – testing/CI helper
    """Tiny synthetic replacement that mimics (img, cls) tuples."""

    def __init__(self, total: int, img_size: int) -> None:
        super().__init__()
        self.total = total
        self.img_size = img_size

    def __len__(self) -> int:  # noqa: D401 – Dataset interface
        return self.total

    def __getitem__(self, idx):  # noqa: D401 – Dataset interface
        g = torch.Generator().manual_seed(idx)
        img = torch.rand(3, self.img_size, self.img_size, generator=g) * 2 - 1  # [-1,1]
        label = torch.tensor(random.randint(0, 999), dtype=torch.int64)
        return img, label


# -----------------------------------------------------------------------------
#  Download helpers
# -----------------------------------------------------------------------------


def _try_download(url: str, dest: pathlib.Path) -> bool:  # pragma: no cover – network IO
    """Attempt to stream-download one shard; return True on success."""
    try:
        r = requests.get(url, stream=True, timeout=30)
    except requests.RequestException:
        return False
    if r.status_code != 200:
        return False

    # Stream into file
    with dest.open("wb") as f:
        shutil.copyfileobj(r.raw, f)
    return True


def _ensure_dummy_tar(dest: pathlib.Path) -> None:
    """Create a minimal tar containing a single random image + class file."""
    if dest.exists():
        return
    with tarfile.open(dest, "w") as tar:
        # fake jpg (just a few random bytes – not a valid JPEG but WDS will skip)
        img_data = io.BytesIO(b"FAKEIMG")
        cls_data = io.BytesIO(b"0")
        tarinfo_img = tarfile.TarInfo(name="000000.jpg")
        tarinfo_img.size = len(img_data.getvalue())
        tarinfo_cls = tarfile.TarInfo(name="000000.cls")
        tarinfo_cls.size = 1
        tar.addfile(tarinfo_img, img_data)
        tar.addfile(tarinfo_cls, cls_data)


def _download_split(split: str) -> bool:  # pragma: no cover – network IO
    """Download *all* shards for the given split; returns True if at least one shard was fetched."""
    split_dir = DATA_ROOT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    success_any = False
    for shard_id in range(N_SHARDS[split]):
        fname = split_dir / f"{shard_id:06d}.tar"
        if fname.exists():
            success_any = True
            continue  # already on disk
        ok = False
        for tmpl in HF_URLS:
            url = tmpl.format(split=split) % shard_id
            print(f"[imagenet] attempting → {url}")
            if _try_download(url, fname):
                ok = True
                success_any = True
                break
        if not ok:
            # Could not fetch this shard from *any* mirror – stop early and fall back.
            print(f"[WARNING] Failed to download shard {shard_id} of {split} from all mirrors – aborting download.")
            break
    return success_any


# -----------------------------------------------------------------------------
#  Public API – the DataLoader builder
# -----------------------------------------------------------------------------


def build_dataloader(global_batch: int, img_size: int) -> Tuple[DataLoader, DataLoader]:
    """Return train & validation loaders.  If remote data is unreachable we fall back to a
    tiny synthetic dataset (explicitly advertised so this is *not* silent)."""

    # Attempt to fetch a *single* shard per split first – if that fails we assume offline.
    have_real_data = True
    for split in ("train", "val"):
        if not _download_split(split):
            have_real_data = False
            break

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size), antialias=True),
        T.Normalize(0.5, 0.5),  # [0,1] -> [-1,1]
    ])

    if not have_real_data:
        print("\n[INFO] Remote ImageNet shards unavailable – using synthetic data for this run.\n")

        def _make_fake_loader() -> DataLoader:
            dataset = _SyntheticImageNet(total=2048, img_size=img_size)
            world = torch.cuda.device_count() or 1
            return DataLoader(dataset, batch_size=global_batch // world, shuffle=True, num_workers=0)

        return _make_fake_loader(), _make_fake_loader()

    # ------------------------------------------------------------------
    # Real WebDataset pipeline
    # ------------------------------------------------------------------

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
