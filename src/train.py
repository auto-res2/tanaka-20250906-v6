"""src/train.py – model definitions, training loop, per-run execution (fixed paths & config)"""
import json, pathlib, random, shutil, subprocess, sys, time, os
from typing import Dict, Any, List

import torch, yaml, numpy as np
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# ---------------------------------------------------------------------------
#  Repository paths (identical logic in all src/* modules for self-containment)
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent      # repo root (one level above src)
DATA_DIR = ROOT / "data"
#  Mandatory research output dirs (see task description)
RESEARCH_DIR = ROOT / ".research" / "iteration21"
IMG_DIR = RESEARCH_DIR / "images"
for p in (DATA_DIR, RESEARCH_DIR, IMG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Keep legacy aliases so the rest of the codebase remains unchanged ------------
RES_DIR = RESEARCH_DIR   # JSON, traces, etc.
FIG_DIR = IMG_DIR        # figures / images

# ---------------------------------------------------------------------------
#  Configuration loader (single source of truth)
# ---------------------------------------------------------------------------
CFG_FILE = ROOT / "config" / "config.yaml"   # fixed name (was exp.yaml)
if not CFG_FILE.exists():
    raise FileNotFoundError("Configuration file not found – create it under config/config.yaml before running.")
CFG = yaml.safe_load(CFG_FILE.read_text())

# ---------------------------------------------------------------------------
#  Model components – FFT-DiT-S and baseline DiT-S
# ---------------------------------------------------------------------------
DIT_REPO = ROOT / "third_party" / "DiT"
if not DIT_REPO.exists():
    print("[SETUP] Cloning DiT repository …")
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/DiT", str(DIT_REPO)])
    shutil.rmtree(DIT_REPO / "experiments", ignore_errors=True)
    (DIT_REPO / ".git").rename(DIT_REPO / "_git")  # avoid nested-repo issues

sys.path.insert(0, str(DIT_REPO))
from models import DiT as _DiT       # noqa: E402 (external import after path manipulation)
from models import DiTBlock          # noqa: E402


class SpectralAdapter(nn.Module):
    """Low-rank spectral adapter (LoRA-style)."""
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)

    def forward(self, x):
        return x + (x @ self.A) @ self.B


class HyperNet(nn.Module):
    """Mini hyper-network that predicts FiLM-style scaling vectors from timestep t."""
    def __init__(self, hidden: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t):
        return self.net(t.unsqueeze(-1))  # → (B, dim)


class FFTDiT_S(nn.Module):
    """Frequency- & Friction-adaptive DiT-S (≈120 M parameters)."""
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["width"]
        depth = cfg["depth"]
        heads = cfg["heads"]
        self.adapter_rank = cfg["adapter_rank"]
        self.img_size = cfg["img_size"]

        # Stem – 4×4 patch embed (like ViT/DiT)
        self.patch_embed = nn.Conv2d(3, d, kernel_size=4, stride=4)
        self.pos = nn.Parameter(torch.randn(1, (self.img_size // 4) ** 2, d) * 0.02)

        # Hyper-network for FiLM gating
        self.hyper = HyperNet(hidden=d, dim=d)

        # Transformer backbone (reuse official DiTBlock)
        self.blocks = nn.ModuleList([DiTBlock(d, heads) for _ in range(depth)])
        self.adapter = SpectralAdapter(d, self.adapter_rank)  # shared across layers

        # Output – predict noise ε per patch (48 dims per patch → 3×4×4)
        self.ln_out = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 3 * 4 * 4)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """x ∈ [-1,1]  (B,3,H,W) ;  t ∈ [0,1]  (B,)"""
        B = x.size(0)
        patch_H = self.img_size // 4
        x = self.patch_embed(x)                      # (B, d, H/4, W/4)
        x = x.flatten(2).transpose(1, 2) + self.pos  # (B, N, d)
        gamma = self.hyper(t)                        # (B, d)
        for blk in self.blocks:
            x = blk(x)
            x = x * gamma.unsqueeze(1)               # FiLM gating
            x = self.adapter(x)                      # spectral adapter
        x = self.ln_out(x)
        x = self.proj(x)                             # (B, N, 48)

        # Re-fold tokens → feature map (B,48,patch_H,patch_H)
        x = x.view(B, patch_H, patch_H, 48).permute(0, 3, 1, 2).contiguous()

        # Single pixel-shuffle to original resolution (4×) ⇒ (B,3,H,W)
        x = torch.nn.functional.pixel_shuffle(x, 4)
        return x


class DiT_S(nn.Module):
    """Wrapper around the official DiT implementation adjusted for CIFAR/mini-ImageNet size."""
    def __init__(self, cfg: dict):
        super().__init__()
        self.net = _DiT(
            image_size=CFG["dataset"]["img_size"],
            patch_size=4,
            in_channels=3,
            num_classes=1000,
            hidden_size=cfg["width"],
            depth=cfg["depth"],
            num_heads=cfg["heads"],
            mlp_ratio=4.0,
        )

    def forward(self, x, t):
        return self.net(x, t)


# ---------------------------------------------------------------------------
#  Training utilities
# ---------------------------------------------------------------------------
from diffusers import DDPMScheduler
from .evaluate import evaluate_fid          # circular-safe (evaluate does not import train)
from .preprocess import build_dataloaders, set_seed


class DiffusionTrainer:
    """Encapsulates optimiser, AMP, DDPM scheduler and the training loop."""
    def __init__(self, model: nn.Module, cfg: dict, seed: int):
        self.model = model.cuda()
        self.cfg = cfg
        self.seed = seed

        self.scaler = GradScaler()
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(0.9, 0.999),
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

    # ---------------------  main loop  -------------------------
    def train(self, train_loader, val_loader, exp_key: str):
        profile_batches = CFG.get("profile_batches", 100)
        prof = profile(
            activities=[ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            schedule=torch.profiler.schedule(wait=0, warmup=2, active=profile_batches, repeat=1),
            on_trace_ready=tensorboard_trace_handler(str(RES_DIR / exp_key)),
        )

        best_fid = float("inf")
        start = time.time()
        max_epochs = CFG.get("max_epochs", 50)
        early_stop_fid = CFG.get("early_stop_fid", 11.0)

        for epoch in range(max_epochs):
            self.model.train()
            for _, batch in enumerate(train_loader):
                imgs = batch["x"].cuda(non_blocking=True)
                t = torch.randint(0, 1000, (imgs.size(0),), device=imgs.device)
                noise = torch.randn_like(imgs)
                noisy = self.scheduler.add_noise(imgs, noise, t)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = self.model(noisy, t.float() / 1000.0)
                    loss = F.mse_loss(pred, noise)

                self.optim.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                prof.step()

            # -------------  validation FID every 2 epochs -------------
            if (epoch + 1) % 2 == 0:
                fid = evaluate_fid(self.model, val_loader, self.scheduler, CFG)
                print(f"[VAL] epoch={epoch + 1}   FID={fid:.2f}")
                best_fid = min(best_fid, fid)
                if fid <= early_stop_fid:
                    print("[EARLY STOP] Target FID reached – stopping training.")
                    break

        wall_hours = (time.time() - start) / 3600.0
        return best_fid, wall_hours


# ---------------------------------------------------------------------------
#  One complete run (model × seed)
# ---------------------------------------------------------------------------

def _instantiate_model(model_key: str):
    if model_key == "fft_dit":
        cfg = {**CFG["models"]["fft_dit"], "img_size": CFG["dataset"]["img_size"]}
        return FFTDiT_S(cfg), cfg
    elif model_key == "dit":
        cfg = CFG["models"]["dit"]
        return DiT_S(cfg), cfg
    else:
        raise NotImplementedError(model_key)


def run_single(model_key: str, seed: int) -> Dict[str, Any]:
    print(f"\n==========  {model_key.upper()}  |  seed={seed}  ==========")

    # reproducibility
    set_seed(seed)

    # data
    train_loader, val_loader = build_dataloaders(batch=CFG["dataset"]["batch"], seed=seed, cfg=CFG)

    # model & trainer
    model, model_cfg = _instantiate_model(model_key)
    trainer = DiffusionTrainer(model, model_cfg, seed)

    fid, wall = trainer.train(train_loader, val_loader, f"{model_key}_s{seed}")

    # ---------------  persist JSON log  ---------------
    res = {
        "model": model_key,
        "seed": seed,
        "best_fid": fid,
        "wall_clock_h": wall,
        "params": sum(p.numel() for p in model.parameters()),
    }
    jpath = RES_DIR / f"{model_key}_s{seed}.json"
    jpath.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res
