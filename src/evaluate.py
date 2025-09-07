"""src/evaluate.py
Evaluation utilities: DDIM sampling, FID/IS computation and simple plotting.
Only *standard* Python packages and the dependencies declared in *pyproject.toml*
are used so that the code works in any environment once dependencies are
installed.
"""
from __future__ import annotations

import shutil
import tempfile
import pathlib
from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from diffusers import DDIMScheduler, DiTPipeline
from torch_fidelity import calculate_metrics

# -----------------------------------------------------------------------------


def _tensor_to_pil(t: torch.Tensor):  # helper
    import torchvision.transforms.functional as F

    return F.to_pil_image((t.clamp(-1, 1) + 1) / 2)


# -----------------------------------------------------------------------------
#  PUBLIC API
# -----------------------------------------------------------------------------

def sample_and_compute_fid(
    *,
    model,
    scheduler,
    device,
    ref_ds,
    num_imgs: int,
    ddim_steps: int,
) -> Tuple[float, float]:
    """Generate *num_imgs* samples with DDIM and return (FID, IS)."""

    pipe = DiTPipeline(
        transformer=model,
        scheduler=DDIMScheduler.from_config(scheduler.config),
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    batch_size = 50  # somewhat memory-friendly default

    gen_dir = pathlib.Path(tempfile.mkdtemp())
    ref_dir = pathlib.Path(tempfile.mkdtemp())

    # ---- prepare reference set  (5k randomly chosen validation images) ----
    all_ids = torch.randperm(len(ref_ds))[:5000]
    for i, idx in enumerate(all_ids):
        img = ref_ds[int(idx)]["x"]
        _tensor_to_pil(img).save(ref_dir / f"{i}.png")

    # ---- generate synthetic images ----------------------------------------
    generated = []
    while len(generated) < num_imgs:
        with torch.no_grad():
            out = pipe(
                batch_size=min(batch_size, num_imgs - len(generated)),
                num_inference_steps=ddim_steps,
                guidance_scale=None,
            ).images
            generated.extend(out)
    for i, img in enumerate(generated):
        img.save(gen_dir / f"{i}.png")

    # ---- compute FID & IS --------------------------------------------------
    metrics = calculate_metrics(
        input1=str(gen_dir),
        input2=str(ref_dir),
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
    )

    # cleanup temporary folders to avoid cluttering /tmp
    shutil.rmtree(gen_dir, ignore_errors=True)
    shutil.rmtree(ref_dir, ignore_errors=True)

    return float(metrics["frechet_inception_distance"]), float(metrics["inception_score_mean"])


# -----------------------------------------------------------------------------
#  SIMPLE FIGURE
# -----------------------------------------------------------------------------

def plot_fid_curve(fid_values: List[float], save_path: pathlib.Path) -> None:
    plt.figure(figsize=(5, 3))
    sns.lineplot(x=list(range(len(fid_values))), y=fid_values)
    for i, v in enumerate(fid_values):
        plt.text(i, v, f"{v:.1f}")
    plt.xlabel("Epoch")
    plt.ylabel("FID")
    plt.title("FID vs epoch")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
