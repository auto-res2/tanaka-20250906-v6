from __future__ import annotations

"""Training logic and model definitions for the FFT-DiT experiments.

This file contains everything required to (i) construct the tiny DiT / FFT-DiT
variants used in the smoke-test configuration, (ii) run one full training loop
for all model variants contained in the YAML configuration and (iii) write the
resulting metrics / figures / profiler traces into the experiment directory
that is provided by the caller (src.main).
"""

import json
import math
import pathlib
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DiTConfig, DiTModel
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity, profile
from tqdm.auto import tqdm

from .evaluate import fid_is_from_samples
from .preprocess import build_dataloaders

__all__ = [
    "run_single",  # main entry point used by src.main
]

# -----------------------------------------------------------------------------
# Misc helpers (kept local to obey the 6-file-only rule)
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
# Model definitions (plain DiT wrapper + trimmed FFT-DiT)
# -----------------------------------------------------------------------------


class _BaseWrapper(nn.Module):
    """Adds .config passthrough so that diffusers' DiTPipeline accepts wrappers."""

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
    """Simplified FFT-DiT (only gating/adapters kept for the CI smoke-test)."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        depth: int,
        width: int,
        heads: int,
        low_tokens: int,
        high_tokens: int,
        adapter_rank: int,
    ) -> None:  # noqa: D401 – lots of parameters, keep as is
        super().__init__(image_size, patch_size, in_channels, depth, width, heads)
        # ------------------------------------------------------------------
        # The heavy FFT-specific bits are omitted – we only keep a tiny
        # HyperNet-style channel-wise gating and a rank-*r* adapter so the
        # model still differs parametrically from the plain DiT.
        # ------------------------------------------------------------------
        self.hyper = nn.Sequential(
            nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width), nn.Sigmoid()
        )
        self.adapter = nn.Sequential(
            nn.Linear(width, adapter_rank, bias=False),
            nn.Linear(adapter_rank, width, bias=False),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):  # noqa: D401
        gating = self.hyper(t.float().unsqueeze(-1) / 1000.0)  # [B, C]
        base = super().forward(x, t)
        adapted = base + gating[:, None, None, :] * self.adapter(base)  # broadcast
        return adapted


# -----------------------------------------------------------------------------
# Training routine
# -----------------------------------------------------------------------------

def _init_model(model_cfg: dict, device: torch.device) -> nn.Module:
    """Factory for model instantiation based on the YAML description."""

    model_type = model_cfg["type"].lower()
    if model_type == "dit":
        net: nn.Module = DiT(**model_cfg["params"])
    elif model_type == "fft_dit":
        net = FFTDiT(**model_cfg["params"])
    else:
        raise ValueError(f"Unknown model type '{model_type}'.")
    return net.to(device)


def _make_figures_dir(out_dir: pathlib.Path) -> pathlib.Path:
    figs = out_dir / "images"
    figs.mkdir(parents=True, exist_ok=True)
    return figs


def run_single(cfg: dict, seed: int, out_dir: pathlib.Path) -> Dict[str, Any]:
    """Run training & evaluation for **all** model variants for one seed."""

    set_seed(seed)
    device = torch.device(cfg["hardware"]["device"] if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("CUDA required – aborting as per strict no-fallback rule.")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_dl, val_dl = build_dataloaders(cfg["dataset"], seed)

    # ------------------------------------------------------------------
    # Prepare output folders
    # ------------------------------------------------------------------
    figs_dir = _make_figures_dir(out_dir)

    # ------------------------------------------------------------------
    # Iterate over all model variants specified in the YAML
    # ------------------------------------------------------------------
    results: Dict[str, Any] = {}
    for model_name, model_cfg in cfg["models"].items():
        net = _init_model(model_cfg, device)
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        scaler = GradScaler()
        scheduler = DDPMScheduler(num_train_timesteps=1000)

        # Exponential moving average for evaluation
        ema_net = _init_model(model_cfg, device)
        ema_net.load_state_dict(net.state_dict())
        ema_decay = cfg["training"]["ema_decay"]

        step_loss: List[Tuple[int, float]] = []
        fid_curve: List[Tuple[int, float]] = []

        trace_path = out_dir / f"trace_{model_name}.json"
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

                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = net(noisy, timesteps)
                        loss = torch.mean((pred - noise) ** 2)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite loss detected – aborting.")

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    # EMA
                    with torch.no_grad():
                        for p_ema, p in zip(ema_net.parameters(), net.parameters()):
                            p_ema.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

                    step_loss.append((global_step, loss.item()))
                    pbar.set_postfix({"loss": loss.item()})
                    global_step += 1

                    prof.step()

                # ------------------------------------------------------------------
                # Epoch-level evaluation (FID/IS)
                # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Plot curves (loss + FID)
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Parse profiler trace for FLOPs (best-effort heuristic)
        # ------------------------------------------------------------------
        pflops_per_iter = 0.0
        if trace_path.exists():
            with trace_path.open() as fp:
                for line in fp:
                    if "flops" in line:
                        ev = json.loads(line)
                        pflops_per_iter += ev.get("args", {}).get("flops", 0)
            pflops_per_iter /= 1e15 * max(1, cfg["hardware"]["profiler_batches"])
        else:
            raise RuntimeError("Profiler trace missing – consistency violation.")

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
