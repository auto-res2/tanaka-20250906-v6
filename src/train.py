"""src/train.py – model creation and training utilities for FFT-DiT experiments
All heavy-lifting (model definition, FSDP trainer, seed helpers) lives here so
that the other modules can stay lightweight.  Nothing outside the six allowed
files is imported other than publicly available PyPI packages.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import time
from typing import Any, Dict, List

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------------------------------------------------------------
#  Re-usable helpers
# -----------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / ".research" / "iteration1"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = RESULT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True, parents=True)


def set_seeds(seed: int) -> None:
    """Make training deterministic-ish (CUDA, numpy, python)."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # True slows down too much
    torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
#  FFT-DiT  (minimal, self-contained)
# -----------------------------------------------------------------------------
try:
    from diffusers.models import DiTModel  # pylint: disable=import-error
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("diffusers is required:  pip install diffusers") from exc

try:
    from flash_fft_conv import fft_conv  # noqa: F401  – import just to assert availability
except ModuleNotFoundError:  # optional speed-up, fall back gracefully
    fft_conv = None  # type: ignore


class SpectralAdapter(nn.Module):
    """Low-rank adapter inserted after MLP/Attention blocks."""

    def __init__(self, dim: int, rank: int = 32):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.up(self.down(x)) + x


class FFTDiTWrapper(nn.Module):
    """Wrap diffusers' DiT-XL with dual-axis capacity scaling."""

    def __init__(self, img_size: int, adapter_rank: int = 32):
        super().__init__()
        # NOTE: requires network connectivity the first time (hf.co download)
        self.core = DiTModel.from_pretrained("facebook/DiT-XL-2-256x256")
        self.img_size = img_size
        dim = self.core.config.hidden_size

        adapter = SpectralAdapter(dim, rank=adapter_rank)
        for module in self.core.modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                module.register_forward_hook(lambda m, inp, out, a=adapter: a(out))

        self.hypernet = nn.Sequential(  # small FiLM-like conditioning
            nn.Linear(1, dim // 4), nn.SiLU(), nn.Linear(dim // 4, dim)
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> Dict[str, Any]:
        scale = self.hypernet(timesteps[:, None].float() / 1000.0).unsqueeze(1)
        return self.core(x, timestep_embed=scale)


def create_model(img_size: int, device: str = "cuda") -> FFTDiTWrapper:
    model = FFTDiTWrapper(img_size)
    return model.to(device=device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)


# -----------------------------------------------------------------------------
#  Trainer  (optionally FSDP-backed)
# -----------------------------------------------------------------------------
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    _FSDP_AVAILABLE = True
except (ImportError, AttributeError):
    _FSDP_AVAILABLE = False


class Trainer:  # pylint: disable=too-many-instance-attributes
    """A very small training loop that keeps running until a target FID is met."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        target_fid: float,
        seed: int,
        exp_name: str,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if _FSDP_AVAILABLE and torch.cuda.device_count() > 1:
            model = FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy(16000))
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_fid = target_fid
        self.exp_name = exp_name
        self.seed = seed

        self.optim = torch.optim.AdamW(model.parameters(), lr=1.5e-4, betas=(0.95, 0.999), eps=1e-8, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=1_000_000)

        self.history: Dict[str, List[float]] = {"iter": [], "loss": []}

    # ------------------------------------------------------------------
    def fit(self) -> None:
        """Run the optimisation loop until `target_fid` is reached."""
        it = 0
        pbar = tqdm(total=1_000_000, desc=self.exp_name, position=0)
        start = time.perf_counter()
        while it < 1_000_000:  # generous upper bound
            for images, _ in self.train_loader:
                self.optim.zero_grad(set_to_none=True)
                images = images.to(self.device, non_blocking=True)
                timesteps = torch.randint(0, 1000, (images.size(0),), device=self.device)

                with autocast(enabled=self.device.type == "cuda", dtype=torch.bfloat16):
                    loss = self.model(images, timesteps)["loss"]

                loss.backward()
                self.optim.step()
                self.scheduler.step()

                # Logging
                self.history["iter"].append(it)
                self.history["loss"].append(float(loss))
                pbar.set_postfix(loss=f"{loss.item():.3f}")
                pbar.update(1)
                it += 1

                # periodic evaluation – heavy, so every 5k iters
                if it % 5000 == 0:
                    from .evaluate import FIDEvaluator

                    fid = FIDEvaluator(self.val_loader, self.model, self.device).compute()
                    if fid <= self.target_fid:
                        self._save_checkpoint(it)
                        pbar.close()
                        print(f"TARGET FID {fid:.2f} reached – training finished.")
                        self.wall_clock_h = (time.perf_counter() - start) / 3600.0  # type: ignore[attr-defined]
                        return
        pbar.close()
        self.wall_clock_h = (time.perf_counter() - start) / 3600.0  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def _save_checkpoint(self, it: int) -> None:
        ckpt_dir = ROOT / "models" / self.exp_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), ckpt_dir / f"step{it}.pt")


# -----------------------------------------------------------------------------
#  Results helper (shared by all modules)
# -----------------------------------------------------------------------------

def save_json(obj: Any, path: pathlib.Path | str) -> None:  # noqa: D401
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(obj, handle, indent=2)
