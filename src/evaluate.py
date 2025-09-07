import json, pathlib, matplotlib
from typing import Dict, List, Any

# Use a non-interactive backend to avoid display issues in headless CI
matplotlib.use("Agg")

import torch, yaml
from torch import autocast
from torch_fidelity import calculate_metrics
import matplotlib.pyplot as plt
import seaborn as sns

# ----------  paths & config  ------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
RESEARCH_DIR = ROOT / ".research" / "iteration34"  # updated to iteration34
IMG_DIR = RESEARCH_DIR / "images"                    # ensured below
for p in (RESEARCH_DIR, IMG_DIR):
    p.mkdir(parents=True, exist_ok=True)

CFG_FILE = ROOT / "config" / "config.yaml"
CFG = yaml.safe_load(CFG_FILE.read_text())

# ----------  FID evaluation  ------------------------------------------------

def evaluate_fid(model, val_loader, scheduler, cfg: Dict[str, Any]):
    """Generate 10 k images with a 20-step DDPM sampling loop and compute FID."""
    model.eval()
    imgs = []
    total_needed = 10000
    batch_size = val_loader.batch_size or cfg["dataset"]["batch"]
    steps = total_needed // batch_size + 1  # ensure > total_needed

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for _ in range(steps):
            z = torch.randn(batch_size, 3, cfg["dataset"]["img_size"], cfg["dataset"]["img_size"], device="cuda")
            for i in range(20)[::-1]:
                t_val = i * 50
                t = torch.full((z.size(0),), t_val, device=z.device)
                with autocast("cuda", dtype=amp_dtype):
                    eps = model(z, t.float() / 1000.0)
                alpha = scheduler.alphas_cumprod[t_val].to(z.device)
                alpha = alpha.view(1, 1, 1, 1)
                z = (z - (1 - alpha).sqrt() * eps) / alpha.sqrt()
            imgs.append(z.detach().cpu())

    imgs = torch.cat(imgs)[:total_needed]
    # torch-fidelity expects images in [0,1]; our samples are in [-1,1]
    imgs = (imgs + 1) / 2.0
    metrics = calculate_metrics(
        input1=imgs,
        input2="cifar10-train",  # reference statistics shipped with torch-fidelity
        metrics=["fid"],
        kid=False,
        feature_layer=2048,
        is_score=False,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])

# ----------  Plotting helper  ----------------------------------------------

def plot_fid(results: List[Dict[str, Any]]):
    sns.set_theme()
    plt.figure(figsize=(5, 3))
    xs = [f"{r['model']}-{r['seed']}" for r in results]
    ys = [r['best_fid'] for r in results]
    bars = plt.bar(xs, ys)
    for bar, y in zip(bars, ys):
        plt.text(bar.get_x() + bar.get_width() / 2, y + 0.2, f"{y:.1f}", ha='center', va='bottom', fontsize=8)
    plt.ylabel("FID ↓")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    name = "fid_exp1.pdf"
    plt.savefig(IMG_DIR / name, bbox_inches="tight")
    print(f"[FIG] saved {IMG_DIR / name}")
