"""src/evaluate.py – evaluation utilities and plotting"""
import json, pathlib
from typing import Dict, List, Any

import torch, yaml
from torch.cuda.amp import autocast
from torch_fidelity import calculate_metrics
import matplotlib.pyplot as plt
import seaborn as sns

# ----------  paths & config  ------------------------------------------------
ROOT      = pathlib.Path(__file__).resolve().parent.parent
RES_DIR   = ROOT / "results"
FIG_DIR   = ROOT / "figures"
CFG_FILE  = ROOT / "config" / "exp.yaml"
CFG       = yaml.safe_load(CFG_FILE.read_text())
for p in (RES_DIR, FIG_DIR):
    p.mkdir(exist_ok=True, parents=True)

# ----------  FID evaluation  ------------------------------------------------

def evaluate_fid(model, val_loader, scheduler, cfg:Dict[str,Any]):
    """Generate 10k images with DDIM-style loop (20 steps) and compute FID."""
    model.eval()
    imgs = []
    total_needed = 10000
    batch_size   = val_loader.batch_size
    steps        = total_needed // batch_size

    with torch.no_grad():
        for _ in range(steps):
            z = torch.randn(batch_size, 3, cfg["dataset"]["img_size"], cfg["dataset"]["img_size"], device="cuda")
            for i in range(20)[::-1]:
                t = torch.full((z.size(0),), i*50, device=z.device)
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    eps = model(z, t.float()/1000.0)
                alpha = scheduler.alphas_cumprod[t.long()][:, None, None, None]
                z = (z - (1-alpha).sqrt()*eps) / alpha.sqrt()
            imgs.append(z.detach().cpu())

    imgs = torch.cat(imgs)[:total_needed]
    metrics = calculate_metrics(
        input1=imgs,
        input2="cifar10-train",   # reference statistics shipped with torch-fidelity
        metrics=["fid"],
        kid=False,
        feature_layer=2048,
        is_score=False,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])

# ----------  Plotting helper  ----------------------------------------------

def plot_fid(results:List[Dict[str,Any]]):
    sns.set_theme()
    plt.figure(figsize=(5,3))
    xs = [f"{r['model']}-{r['seed']}" for r in results]
    ys = [r['best_fid'] for r in results]
    bars = plt.bar(xs, ys)
    for bar, y in zip(bars, ys):
        plt.text(bar.get_x()+bar.get_width()/2, y+0.2, f"{y:.1f}", ha='center', va='bottom', fontsize=8)
    plt.ylabel("FID ↓")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    name = "fid_exp1.pdf"
    plt.savefig(FIG_DIR/name, bbox_inches="tight")
    print(f"[FIG] saved {FIG_DIR/name}")
