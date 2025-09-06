"""src/preprocess.py – dataset downloading, preprocessing, reproducibility helpers"""
import pathlib, random, yaml
from typing import Tuple

import torch, torchvision
from torch.utils.data import DataLoader
from datasets import load_dataset

# -----------------  paths / config  ----------------------------
ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CFG_FILE = ROOT / "config" / "exp.yaml"
CFG      = yaml.safe_load(CFG_FILE.read_text())
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------  helpers  ----------------------------------
class DataDownloadError(RuntimeError):
    """Raised when required HuggingFace dataset shards are unavailable."""
    pass

_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(CFG["dataset"]["img_size"] + 16, antialias=True),
    torchvision.transforms.CenterCrop(CFG["dataset"]["img_size"]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([-1, -1, -1], [2, 2, 2]),
])

def build_dataloaders(*, batch:int, seed:int, cfg:dict):
    """Download (if necessary) the mini-ImageNet set from HuggingFace and build train/val loaders."""
    try:
        ds = load_dataset(cfg["dataset"]["hf_repo"], split="train", cache_dir=str(DATA_DIR))
    except Exception as e:
        raise DataDownloadError(f"Dataset unavailable – aborting. Details: {e}")

    ds = ds.train_test_split(test_size=cfg["dataset"]["val_split"], seed=seed)
    train_ds, val_ds = ds["train"], ds["test"]

    def _map(example):
        img   = _transform(example["img"].convert("RGB"))
        label = int(example["label"])
        return {"x": img, "y": torch.tensor(label, dtype=torch.long)}

    train_ds.set_transform(_map)
    val_ds.set_transform(_map)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=8, drop_last=True, generator=g, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

# -----------------  reproducibility  ---------------------------

def set_seed(seed:int):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
