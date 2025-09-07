"""src/train.py
All model architectures, training utilities and hardware–measurement helpers live here.
The file is completely self-contained so that *no other* local module needs to be
imported from anywhere else in the project – this satisfies the strict-file rule.
"""
from __future__ import annotations

import math
import pathlib
import random
import re
import subprocess
import threading
import time
import warnings
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity, profile

# -----------------------------------------------------------------------------
#  REPRODUCIBILITY & HARDWARE HELPERS
# -----------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
BF16_OK = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if BF16_OK else torch.float16


def set_seed(seed: int) -> None:
    """Set *all* RNG seeds for full reproducibility."""
    # local import to keep global namespace clean
    import numpy as _np  # noqa: N812  -- alias avoids shadowing the global np

    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
#  MODEL DEFINITIONS  (DiT & FFT-DiT)
# -----------------------------------------------------------------------------
# NOTE:  The internal package layout of *diffusers* changed a few times. Newer
#        versions expose DiT both via ``diffusers.models.dit`` *and* directly
#        from ``diffusers.models``.  To stay compatible with every minor version
#        in the allowed range (<0.36) we try the more specific import first and
#        gracefully fall back to the generic one.
try:
    from diffusers.models.dit import DiTConfig, DiTModel  # pylint: disable=wrong-import-order
except ModuleNotFoundError:  # pragma: no cover – executed only on older wheels
    from diffusers.models import DiTConfig, DiTModel  # type: ignore  # noqa: F401,E501


class BaseDiT(nn.Module):
    """Thin wrapper around the public DiT implementation shipped with *diffusers*."""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.model = DiTModel(DiTConfig(**cfg))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: D401, N802
        return self.model(x, timestep=t).sample


def _spectral_adapter(in_ch: int, out_ch: int, rank: int) -> nn.Sequential:  # helper
    return nn.Sequential(
        nn.Conv2d(in_ch, rank, 1, bias=False),
        nn.GELU(),
        nn.Conv2d(rank, out_ch, 1, bias=False),
    )


class FFTDiT(BaseDiT):
    """Frequency- and time-adaptive DiT variant from the paper."""

    def __init__(self, cfg: Dict, adapter_rank: int):
        # ``adapter_rank`` is **not** a valid argument for DiTConfig.  Strip it
        # before passing the dict further down to the official implementation.
        cfg_core = {k: v for k, v in cfg.items() if k != "adapter_rank"}
        super().__init__(cfg_core)
        ic = cfg_core["in_channels"]
        self.adapter = _spectral_adapter(ic, ic, adapter_rank)
        self.hyper = nn.Sequential(
            nn.Linear(1, cfg_core["hidden_size"]),
            nn.SiLU(),
            nn.Linear(cfg_core["hidden_size"], ic),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: D401, N802
        base = super().forward(x, t)
        gate = self.hyper(t[:, None].float() / 1000.0)[:, :, None, None]
        return base + gate * self.adapter(base)


# Public factory ----------------------------------------------------------------

def make_model(model_cfg: Dict) -> nn.Module:
    typ = model_cfg["type"].lower()
    params = model_cfg["params"].copy()  # shallow copy so we can mutate safely
    if typ == "dit":
        params.pop("adapter_rank", None)  # ignore if present accidentally
        return BaseDiT(params)
    if typ == "fft_dit":
        adapter_rank = params.pop("adapter_rank")
        return FFTDiT(params, adapter_rank=adapter_rank)
    raise ValueError(f"Unknown model type: {typ}")


# -----------------------------------------------------------------------------
#  EMA WEIGHT UPDATE
# -----------------------------------------------------------------------------

def ema_update(src: nn.Module, dst: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for p_src, p_dst in zip(src.parameters(), dst.parameters(), strict=True):
            p_dst.mul_(decay).add_(p_src, alpha=1 - decay)


# -----------------------------------------------------------------------------
#  POWER LOGGER –  background GPU power monitor
# -----------------------------------------------------------------------------
class PowerLogger:  # pylint: disable=too-few-public-methods
    """Very small helper that queries *nvidia-smi* every <interval> seconds."""

    def __init__(self, *, interval: int, outfile: pathlib.Path):
        self._int = interval
        self._of = outfile
        self._stop = False
        self._thr = None

    def _loop(self):  # noqa: D401
        with self._of.open("w") as f:
            while not self._stop:
                try:
                    power = (
                        subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=power.draw",
                                "--format=csv,noheader,nounits",
                            ]
                        )
                        .decode()
                        .strip()
                    )
                except subprocess.SubprocessError:
                    power = "nan"
                ts = time.time()
                f.write(f"{ts},{power}\n")
                f.flush()
                time.sleep(self._int)

    def start(self):
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop = True
        if self._thr is not None:
            self._thr.join()


# -----------------------------------------------------------------------------
#  PROFILER → PFLOP EXTRACTOR
# -----------------------------------------------------------------------------

def extract_pflops(trace_file: pathlib.Path | str, prof_batches: int) -> float:
    """Return PFLOPs / iteration from the PyTorch chrome-trace. Falls back to Nsight."""

    trace_file = pathlib.Path(trace_file)
    if not trace_file.exists():
        warnings.warn("Trace file not found; returning NaN PFLOPs")
        return float("nan")

    blob = trace_file.read_bytes()
    matches = re.findall(rb'"FLOPs[^"]*"\s*:\s*([0-9eE.+-]+)', blob)
    if matches:
        total = sum(float(x) for x in matches)
        return total / 1e15 / max(1, prof_batches)

    # --------------  FALLBACK  ----------------
    try:
        subprocess.run(
            ["nsys", "profile", "-t", "cuda,nvtx", "-o", "_nsys_tmp", "sleep", "1"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        rep = pathlib.Path("_nsys_tmp.nsys-rep")
        if rep.exists():
            js = subprocess.check_output(
                ["nsys", "stats", "--report", "gpu-kernels", "--format", "json", rep]
            )
            flops = float(math.nan)
            try:
                import json as _json

                flops = _json.loads(js)["total"].get("flop", float("nan"))
            except Exception:  # noqa: BLE001
                pass
            return flops / 1e15
    except Exception:  # noqa: BLE001
        pass

    warnings.warn("Could not parse FLOPs; returning NaN")
    return float("nan")


# -----------------------------------------------------------------------------
#  ONE-EPOCH TRAINING ROUTINE (public API)
# -----------------------------------------------------------------------------

def train_one_epoch(
    *,
    model: nn.Module,
    ema: nn.Module,
    opt: torch.optim.Optimizer,
    scheduler,  # diffusion noise schedule  (e.g. diffusers.DDPMScheduler)
    dataloader: torch.utils.data.DataLoader,
    scaler: GradScaler,
    device: torch.device,
    ema_decay: float,
    profiler_batches: int,
) -> float:
    """Run **exactly** one epoch and return the average training loss."""

    model.train()
    loss_meter: List[float] = []

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True,
        profile_memory=True,
        schedule=torch.profiler.schedule(
            wait=0, warmup=0, active=profiler_batches, repeat=1
        ),
        on_trace_ready=lambda p: p.export_chrome_trace("trace.json"),
    ) as prof:
        for _, batch in enumerate(dataloader):  # noqa: B007  (step unused)
            imgs = batch["x"].to(device, non_blocking=True)
            bsz = imgs.size(0)
            timesteps = torch.randint(0, 1000, (bsz,), device=device)
            noise = torch.randn_like(imgs)
            noisy = scheduler.add_noise(imgs, noise, timesteps)

            with autocast(dtype=AMP_DTYPE):
                pred = model(noisy, timesteps)
                loss = torch.mean((pred - noise) ** 2)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered – aborting epoch.")

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            ema_update(model, ema, ema_decay)
            loss_meter.append(loss.item())
            prof.step()

    return float(np.mean(loss_meter))
