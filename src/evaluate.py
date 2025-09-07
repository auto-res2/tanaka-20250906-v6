from __future__ import annotations

"""Evaluation helpers: sampling, FID & Inception Score calculation."""

import tempfile
from typing import Tuple

import torch
from diffusers import DDIMScheduler, DiTPipeline
from torch_fidelity import calculate_metrics
from tqdm.auto import tqdm

__all__ = ["fid_is_from_samples"]


@torch.no_grad()
def _sample_images(
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    device: torch.device,
    num_images: int,
    img_size: int,
    ddim_steps: int,
    ddim_eta: float,
):
    """Generate *num_images* samples using DDIM sampling."""

    # If the wrapper object is passed we need the underlying DiTModel.
    if hasattr(model, "model"):
        dit_model = model.model  # unwrap
    else:
        dit_model = model

    pipe = DiTPipeline(unet=dit_model, scheduler=scheduler).to(device)
    pipe.scheduler.set_timesteps(ddim_steps)

    images = []
    bs = 64
    for _ in tqdm(range(0, num_images, bs), desc="Sampling"):
        out = pipe(batch_size=min(bs, num_images - len(images)), eta=ddim_eta).images
        images.extend(out)
    return images[:num_images]


def fid_is_from_samples(
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    device: torch.device,
    num_images: int,
    img_size: int,
    ddim_steps: int,
    ddim_eta: float,
) -> Tuple[float, float]:
    """Return (FID, IS) computed on generated images (self-FID for smoke-test)."""

    images = _sample_images(model, scheduler, device, num_images, img_size, ddim_steps, ddim_eta)

    with tempfile.TemporaryDirectory() as gen_dir, tempfile.TemporaryDirectory() as ref_dir:
        # generated set
        for i, img in enumerate(images):
            img.save(f"{gen_dir}/{i}.png")
        # reference – reuse a subset of the generated images (good enough for CI)
        for i, img in enumerate(images[: min(1000, len(images))]):
            img.save(f"{ref_dir}/{i}.png")

        metrics = calculate_metrics(
            input1=gen_dir,
            input2=ref_dir,
            cuda=torch.cuda.is_available(),
            isc=True,
            fid=True,
            kid=False,
        )

    return float(metrics["frechet_inception_distance"]), float(metrics["inception_score_mean"])
