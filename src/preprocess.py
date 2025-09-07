import pathlib, random, yaml
from typing import Tuple

import torch, torchvision
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# -----------------  paths / config  ----------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CFG_FILE = ROOT / "config" / "config.yaml"
CFG = yaml.safe_load(CFG_FILE.read_text())
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------  helpers  ----------------------------------
class DataDownloadError(RuntimeError):
    """Raised when required HuggingFace dataset shards are unavailable."""
    pass

_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(CFG["dataset"]["img_size"] + 16, antialias=True),
    torchvision.transforms.CenterCrop(CFG["dataset"]["img_size"]),
    torchvision.transforms.ToTensor(),
    # map [0,1] → [-1,1]
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# -----------------  custom dataset wrapper  --------------------
class HFDataset(Dataset):
    """Thin wrapper converting a HF dataset row → tensor dict usable by PyTorch."""
    def __init__(self, hf_ds):
        self.hf_ds = hf_ds
        # figure out the image column name dynamically
        if "img" in hf_ds.column_names:
            self.img_key = "img"
        elif "image" in hf_ds.column_names:
            self.img_key = "image"
        else:
            raise KeyError("Supported image column not found in dataset (expected 'img' or 'image').")

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        example = self.hf_ds[int(idx)]
        img = _transform(example[self.img_key].convert("RGB"))
        label = int(example["label"])
        return {"x": img, "y": torch.tensor(label, dtype=torch.long)}

# -----------------  public API  --------------------------------

def build_dataloaders(*, batch: int, seed: int, cfg: dict):
    """Download (if necessary) the mini-ImageNet set from HuggingFace and build train/val loaders."""
    try:
        ds = load_dataset(cfg["dataset"]["hf_repo"], split="train", cache_dir=str(DATA_DIR))
    except Exception as e:
        raise DataDownloadError(f"Dataset unavailable – aborting. Details: {e}")

    ds = ds.train_test_split(test_size=cfg["dataset"]["val_split"], seed=seed)
    train_ds, val_ds = ds["train"], ds["test"]

    train_loader = DataLoader(HFDataset(train_ds), batch_size=batch, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    val_loader = DataLoader(HFDataset(val_ds), batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

# -----------------  reproducibility  ---------------------------

def set_seed(seed: int):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Only call CUDA seeding helpers if a CUDA device is actually present to avoid
    # runtime errors in CPU-only environments.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
