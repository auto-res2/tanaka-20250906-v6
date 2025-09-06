"""src/train.py – model definitions, training loop, per-run execution
(updated to iteration33 paths + resilient AMP import)"""
import json, pathlib, random, shutil, subprocess, sys, time, os, contextlib
from typing import Dict, Any, List

import torch, yaml, numpy as np
from torch import nn
# --------------------------  AMP helpers  ---------------------------------
# GradScaler recently moved to torch.amp but remains in torch.cuda.amp for
# older PyTorch releases.  We import from the new location first and
# transparently fall-back to the classic path to maximise compatibility.
try:
    from torch.amp import GradScaler            # PyTorch ≥2.1
except (ImportError, AttributeError):           # older versions
    from torch.cuda.amp import GradScaler       # type: ignore

from torch import autocast                       # torch.autocast("cuda", …)
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  Repository paths (NOTE: iteration **33** as mandated)
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent      # repo root
DATA_DIR = ROOT / "data"
RESEARCH_DIR = ROOT / ".research" / "iteration33"
IMG_DIR = RESEARCH_DIR / "images"
for p in (DATA_DIR, RESEARCH_DIR, IMG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# legacy aliases ------------------------------------------------------------
RES_DIR = RESEARCH_DIR   # JSON, traces, etc.
FIG_DIR = IMG_DIR        # figures / images

# ---------------------------------------------------------------------------
#  Configuration loader
# ---------------------------------------------------------------------------
CFG_FILE = ROOT / "config" / "config.yaml"
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
from models import DiT as _DiT          # noqa: E402
from models import DiTBlock             # noqa: E402

# ---------------------------------------------------------------------------
#  Small utility – graceful profiler disable ---------------------------------
# ---------------------------------------------------------------------------
class _NoOpProfiler(contextlib.AbstractContextManager):
    """Stand-in for torch.profiler.profile when profiling is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
    def step(self):
        pass

# ---------------------------------------------------------------------------
#  Layers / modules ----------------------------------------------------------
# ---------------------------------------------------------------------------
class SpectralAdapter(nn.Module):
    """Low-rank spectral adapter (LoRA-style)."""
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)

    def forward(self, x):
        return x + (x @ self.A) @ self.B


class HyperNet(nn.Module):
    """Mini hyper-network predicting FiLM scaling vectors from timestep t."""
    def __init__(self, hidden: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t):
        return self.net(t.unsqueeze(-1))


class FFTDiT_S(nn.Module):
    """Frequency- & Friction-adaptive DiT-S (≈120 M parameters)."""
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["width"]
        depth = cfg["depth"]
        heads = cfg["heads"]
        self.adapter_rank = cfg["adapter_rank"]
        self.img_size = cfg["img_size"]

        # Stem – 4×4 patch embed
        self.patch_embed = nn.Conv2d(3, d, kernel_size=4, stride=4)
        self.pos = nn.Parameter(torch.randn(1, (self.img_size // 4) ** 2, d) * 0.02)

        # FiLM hyper-network
        self.hyper = HyperNet(hidden=d, dim=d)

        # Backbone + shared adapter
        self.blocks = nn.ModuleList([DiTBlock(d, heads) for _ in range(depth)])
        self.adapter = SpectralAdapter(d, self.adapter_rank)

        # Output head – predict ε
        self.ln_out = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 3 * 4 * 4)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """x ∈ [-1,1]  (B,3,H,W) ;  t ∈ [0,1]  (B,)"""
        B = x.size(0)
        patch_H = self.img_size // 4
        d = self.pos.size(-1)

        # Stem
        x = self.patch_embed(x)                      # (B,d,H/4,W/4)
        x = x.flatten(2).transpose(1, 2) + self.pos  # (B,N,d)

        # FiLM
        gamma = self.hyper(t)                        # (B,d)
        c = torch.zeros(B, d, device=x.device, dtype=x.dtype)  # dummy class token context

        for blk in self.blocks:
            x = blk(x, c)
            x = x * gamma.unsqueeze(1)               # FiLM gating
            x = self.adapter(x)                      # spectral adapter

        # Head
        x = self.ln_out(x)
        x = self.proj(x)                             # (B,N,48)
        x = x.view(B, patch_H, patch_H, 48).permute(0, 3, 1, 2).contiguous()
        x = torch.nn.functional.pixel_shuffle(x, 4)  # (B,3,H,W)
        return x


class DiT_S(nn.Module):
    """Wrapper around official DiT implementation adjusted for mini-ImageNet size."""
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

        # micro-batching to tame GPU memory ----------------------------------
        # If "micro_batch" not in YAML, fall back to 16 which is safe on T4 / 16 GB
        self.micro_batch = int(CFG.get("micro_batch", 16))

        # decide if we profile ------------------------------------------------
        self.profile_batches = CFG.get("profile_batches", 0)
        if self.profile_batches > 0:
            from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
            trace_dir = RES_DIR / f"trace_seed{seed}"
            trace_dir.mkdir(parents=True, exist_ok=True)
            self.prof = profile(
                activities=[ProfilerActivity.CPU],  # CPU-only to avoid GPU OOM
                record_shapes=False,
                schedule=torch.profiler.schedule(wait=0, warmup=2, active=self.profile_batches, repeat=1),
                on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
            )
        else:
            self.prof = _NoOpProfiler()

    # ---------------------  main loop  -------------------------
    def train(self, train_loader, val_loader, exp_key: str):
        best_fid = float("inf")
        start = time.time()
        max_epochs = CFG.get("max_epochs", 50)
        early_stop_fid = CFG.get("early_stop_fid", 11.0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        with self.prof:
            for epoch in range(max_epochs):
                self.model.train()
                for batch in train_loader:
                    imgs_cpu = batch["x"]  # still on host (pinned) memory
                    total_B = imgs_cpu.size(0)
                    self.optim.zero_grad(set_to_none=True)
                    accum_steps = (total_B + self.micro_batch - 1) // self.micro_batch

                    for i0 in range(0, total_B, self.micro_batch):
                        imgs = imgs_cpu[i0:i0 + self.micro_batch].cuda(non_blocking=True)
                        cur_B = imgs.size(0)
                        t = torch.randint(0, 1000, (cur_B,), device=imgs.device)
                        noise = torch.randn_like(imgs)
                        noisy = self.scheduler.add_noise(imgs, noise, t)

                        with autocast("cuda", dtype=amp_dtype):
                            pred = self.model(noisy, t.float() / 1000.0)
                            loss = F.mse_loss(pred, noise) / accum_steps  # normalise by grad-accum steps

                        self.scaler.scale(loss).backward()

                    # Optimiser step after full logical batch ----------------
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.prof.step()

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
        cfg = {**CFG["models"]["dit"], "img_size": CFG["dataset"]["img_size"]}
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
    # Immediate stdout for CI log parsing / verification
    print(json.dumps(res, indent=2))
    return res