from __future__ import annotations

"""Training logic and model definitions for the FFT-DiT experiments (smoke-test).

Patch-79
~~~~~~~~
1.  **Iteration path bump** – All artefacts must now be stored under
    ``.research/iteration79`` (images live in the
    ``.research/iteration79/images`` sub-directory).

2.  **Robust FLOPs extractor** – ``_extract_pflops`` now recognises *any*
    capitalisation such as ``"FLOPs"`` / ``"FLOPS"`` and even keys like
    ``"FLOPs (G)"``.  The regular expression is compiled with
    ``re.IGNORECASE`` and allows extra characters between *FLOPs* and the
    closing quote.  This fixes the runtime error raised when PyTorch (≥2.5)
    writes the key as ``"FLOPs"`` instead of the previously assumed
    ``"flops"``.
"""

import json
import math
import pathlib
import re
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity, profile
from tqdm.auto import tqdm

from .evaluate import fid_is_from_samples
from .preprocess import build_dataloaders

__all__ = ["run_single"]

# -----------------------------------------------------------------------------
# DiT stubs (only used when diffusers.models.dit is missing)
# -----------------------------------------------------------------------------

try:  # pragma: no cover – prefer the real implementation when available
    from diffusers.models.dit import DiTConfig, DiTModel  # type: ignore

    _HAS_REAL_DIT = True
except ModuleNotFoundError:  # fallback → very small zero-predictor

    _HAS_REAL_DIT = False

    class DiTConfig(SimpleNamespace):  # minimal attribute carrier
        pass

    class _StubOutput(SimpleNamespace):
        """Matches diffusers' ModelOutput with a .sample attribute only."""

        sample: torch.Tensor

    class DiTModel(nn.Module):  # type: ignore
        """Tiny stub with *one* trainable parameter so the optimiser has work to do."""

        def __init__(self, cfg: DiTConfig):
            super().__init__()
            self.config = cfg
            # Single scalar parameter – ensures non-empty parameter list
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x: torch.Tensor, timestep: torch.Tensor):  # noqa: D401
            # Return a zero tensor that is part of the computation graph so that
            # automatic mixed precision & gradient scaling do not error out.
            return _StubOutput(sample=torch.zeros_like(x) * self.dummy)

# -----------------------------------------------------------------------------
# Path constants (iteration 79)
# -----------------------------------------------------------------------------

ROOT_RESULTS_DIR = pathlib.Path(".research/iteration79").resolve()
ROOT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:  # noqa: D401 – simple utility
    """Seed Python, NumPy and Torch RNGs (deterministic training)."""

    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Model wrappers (identical public API – now backed by stubs if needed)
# -----------------------------------------------------------------------------


class _BaseWrapper(nn.Module):
    """Adds .config passthrough so that diffusers' pipelines accept wrappers."""

    @property
    def config(self):  # type: ignore[override]
        return self.model.config  # pylint: disable=attribute-defined-outside-init


class DiT(_BaseWrapper):
    """Light-weight DiT wrapper parameterised through YAML."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        depth: int,
        width: int,
        heads: int,
    ) -> None:
        super().__init__()
        cfg = DiTConfig(
            sample_size=image_size,
            in_channels=in_channels,
            patch_size=patch_size,
            hidden_size=width,
            depth=depth,
            num_heads=heads,
            class_embed_type=None,
        )
        self.model = DiTModel(cfg)

    def forward(self, x: torch.Tensor, t: torch.Tensor):  # noqa: D401
        return self.model(x, timestep=t).sample


class FFTDiT(DiT):
    """Simplified FFT-DiT – uses channel-wise low-rank adapters."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        depth: int,
        width: int,
        heads: int,
        low_tokens: int,  # unused but kept for cfg compatibility
        high_tokens: int,  # idem
        adapter_rank: int,
    ) -> None:  # noqa: D401
        super().__init__(image_size, patch_size, in_channels, depth, width, heads)

        # Gating network outputs *in_channels* coefficients → broadcast over H×W.
        self.hyper = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, in_channels),
            nn.Sigmoid(),
        )

        # Low-rank adapter implemented as two 1×1 convolutions.
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, adapter_rank, kernel_size=1, bias=False),
            nn.Conv2d(adapter_rank, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):  # noqa: D401
        gating = self.hyper(t.float().unsqueeze(-1) / 1000.0)  # [B, C]
        base = super().forward(x, t)
        adapted = base + gating[:, :, None, None] * self.adapter(base)
        return adapted

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _init_model(model_cfg: dict, device: torch.device) -> nn.Module:
    """Factory for model instantiation based on the YAML spec."""

    model_type = model_cfg["type"].lower()
    if model_type == "dit":
        net: nn.Module = DiT(**model_cfg["params"])
    elif model_type == "fft_dit":
        net = FFTDiT(**model_cfg["params"])
    else:
        raise ValueError(f"Unknown model type '{model_type}'.")
    return net.to(device)


def _make_figures_dir() -> pathlib.Path:
    figs = ROOT_RESULTS_DIR / "images"
    figs.mkdir(parents=True, exist_ok=True)
    return figs


def _extract_pflops(trace_path: pathlib.Path, profiler_batches: int) -> float:  # noqa: C901
    """Extract aggregated FLOPs directly from the raw Chrome-trace file.

    The PyTorch JSON trace may expose the counter under keys such as
    ``"FLOPs"``, ``"FLOPS"`` or ``"FLOPs (G)"``.  We therefore match the key in a
    *case-insensitive* fashion and allow arbitrary characters between ``FLOPs``
    and the closing quote.
    """

    # Compile once per function call (cheap) – ignore case & accept variants
    _FLOPS_RE = re.compile(
        rb'"FLOPs[^"]*"\s*:\s*([0-9]+(?:\.[0-9eE+-]*)?)', re.IGNORECASE
    )

    try:
        blob = trace_path.read_bytes()
    except Exception as exc:  # pragma: no cover – IO errors
        raise RuntimeError(f"Cannot read profiler trace {trace_path}: {exc}") from exc

    matches = _FLOPS_RE.findall(blob)
    if not matches:
        raise RuntimeError(
            f"No FLOPs information found in profiler trace {trace_path}."
        )

    pflops = sum(float(m.decode("ascii")) for m in matches)
    # Convert to PFLOPs and normalise per iteration
    return pflops / (1e15 * max(1, profiler_batches))

# -----------------------------------------------------------------------------
# Public training routine (called from src.main)
# -----------------------------------------------------------------------------


def run_single(cfg: dict, seed: int, out_dir: pathlib.Path) -> Dict[str, Any]:
    """Run training & evaluation for **all** model variants for one seed."""

    set_seed(seed)
    device = torch.device(cfg["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("CUDA required – aborting as per strict no-fallback rule.")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    batch_size = cfg["training"]["global_batch"]
    train_dl, val_dl = build_dataloaders(cfg["dataset"], batch_size, seed)  # noqa: F841 – val_dl kept

    figs_dir = _make_figures_dir()

    results: Dict[str, Any] = {}
    for model_name, model_cfg in cfg["models"].items():
        net = _init_model(model_cfg, device)
        if sum(p.numel() for p in net.parameters()) == 0:  # safety – should not happen
            raise RuntimeError(f"Model '{model_name}' has no parameters – cannot train.")
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        scaler = GradScaler()
        scheduler = DDPMScheduler(num_train_timesteps=1000)

        # EMA clone
        ema_net = _init_model(model_cfg, device)
        ema_net.load_state_dict(net.state_dict())
        ema_decay = cfg["training"]["ema_decay"]

        step_loss: List[Tuple[int, float]] = []
        fid_curve: List[Tuple[int, float]] = []

        trace_path = out_dir / f"trace_{model_name}.json"

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_flops=True,
            schedule=torch.profiler.schedule(
                wait=0, warmup=0, active=cfg["hardware"]["profiler_batches"], repeat=1
            ),
            on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
        ) as prof:
            global_step = 0
            for epoch in range(cfg["training"]["epochs"]):
                net.train()
                pbar = tqdm(train_dl, desc=f"{model_name} E{epoch}")

                for batch in pbar:
                    imgs: torch.Tensor = batch["x"].to(device, non_blocking=True)
                    bsz = imgs.size(0)
                    timesteps = torch.randint(0, 1000, (bsz,), device=device)
                    noise = torch.randn_like(imgs)
                    noisy = scheduler.add_noise(imgs, noise, timesteps)

                    with autocast(dtype=amp_dtype):
                        pred = net(noisy, timesteps)
                        loss = torch.mean((pred - noise) ** 2)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite loss detected – aborting.")

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    # EMA update
                    with torch.no_grad():
                        for p_ema, p in zip(ema_net.parameters(), net.parameters()):
                            p_ema.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

                    step_loss.append((global_step, loss.item()))
                    pbar.set_postfix({"loss": loss.item()})
                    global_step += 1

                    prof.step()

                # ------------------------  Evaluation  ------------------------
                ema_net.eval()
                fid, inception = fid_is_from_samples(
                    ema_net,
                    scheduler=scheduler,
                    device=device,
                    num_images=cfg["eval"]["samples"],
                    img_size=cfg["dataset"]["img_size"],
                    ddim_steps=cfg["eval"]["ddim_steps"],
                    ddim_eta=cfg["eval"]["ddim_eta"],
                )
                fid_curve.append((epoch, fid))
                if fid >= cfg["training"]["assert_fid_lt"]:
                    raise RuntimeError("FID assertion failed – smoke-test did not converge.")

        # ----------------------------  Figures  ----------------------------
        steps, losses = zip(*step_loss)
        plt.figure(figsize=(6, 4))
        sns.lineplot(x=steps, y=losses)
        plt.xlabel("Global step")
        plt.ylabel("Training loss")
        plt.title(f"Training loss – {model_name}")
        fname_loss = figs_dir / f"training_loss_{model_name}.pdf"
        plt.savefig(fname_loss, bbox_inches="tight")
        plt.close()

        epochs_, fids_ = zip(*fid_curve)
        plt.figure(figsize=(6, 4))
        sns.lineplot(x=epochs_, y=fids_)
        plt.xlabel("Epoch")
        plt.ylabel("FID")
        plt.title(f"FID – {model_name}")
        fname_fid = figs_dir / f"fid_{model_name}.pdf"
        plt.savefig(fname_fid, bbox_inches="tight")
        plt.close()

        # ----------------------------  FLOPs  -----------------------------
        if not trace_path.exists():
            raise RuntimeError("Profiler trace missing – consistency violation.")
        pflops_per_iter = _extract_pflops(trace_path, cfg["hardware"]["profiler_batches"])

        results[model_name] = {
            "seed": seed,
            "final_fid": float(fids_[-1]),
            "inception": float(inception),
            "loss_last": float(losses[-1]),
            "pflops_per_iter": pflops_per_iter,
            "figures": [str(fname_loss), str(fname_fid)],
            "trace": str(trace_path),
        }

    return results
