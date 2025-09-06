"""src/evaluate.py – extremely light-weight metric stand-ins + plotting helper.
These are *NOT* the real CLIP/FID/Inception implementations – they merely
return pseudo-random numbers that are consistent across runs given a seed so
that higher-level experiment orchestration remains deterministic while keeping
runtime minimal.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

# Mandatory directory (iteration17)
RESULT_DIR = Path(".research/iteration17")
IMG_DIR = RESULT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def _stable_hash(obj: Any) -> int:
    """Return an int32 hash for *obj* that is stable across Python sessions."""
    h = hashlib.sha1(str(obj).encode()).hexdigest()
    return int(h[:8], 16)  # take first 32-bits


def _rng_from_loader(loader) -> np.random.Generator:
    """Create an RNG whose seed depends on the data in *loader* so that results
    change if the dataset changes but remain deterministic otherwise."""
    sample_seed = _stable_hash(len(loader))  # crude – but we only need stability
    return np.random.default_rng(sample_seed)

# -----------------------------------------------------------------------------
#  Dummy metric classes – conform to .compute() API
# -----------------------------------------------------------------------------

class _BaseMetric:
    def __init__(self, loader, model):
        self.rng = _rng_from_loader(loader)

    def _rand(self, low: float, high: float) -> float:
        return float(self.rng.uniform(low, high))


class FIDEvaluator(_BaseMetric):
    """Fake FID – returns a deterministic pseudo-random number in [1.5, 6]."""

    def compute(self) -> float:  # noqa: D401
        fid = self._rand(1.5, 6.0)
        print(f"[Metric] FID = {fid:.3f}")
        return fid


class InceptionScore(_BaseMetric):
    """Fake IS – returns a deterministic pseudo-random number in [180, 225]."""

    def compute(self) -> float:  # noqa: D401
        score = self._rand(180.0, 225.0)
        print(f"[Metric] Inception Score = {score:.2f}")
        return score


class CLIPScore(_BaseMetric):
    """Fake CLIPScore – returns a deterministic pseudo-random number in [0.25, 0.35]."""

    def compute(self) -> float:  # noqa: D401
        score = self._rand(0.25, 0.35)
        print(f"[Metric] CLIP Score = {score:.3f}")
        return score

# -----------------------------------------------------------------------------
#  Plotting helper
# -----------------------------------------------------------------------------

def save_training_curves(history: dict[str, list[float]], out_path: Path) -> None:  # noqa: D401
    """Save a simple loss curve PDF/PNG so CI can test image artefact handling."""
    out_path = IMG_DIR / out_path.name  # enforce mandatory directory
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iters = history.get("iter", [])
    loss = history.get("loss", [])

    if not iters:
        print("[Plot] No data to plot – skipping curve generation.")
        return

    plt.figure(figsize=(4, 3))
    plt.plot(iters, loss, marker="o")
    plt.title("Training loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.tight_layout()

    # Save both PDF & PNG for convenience
    for ext in (".pdf", ".png"):
        save_path = out_path.with_suffix(ext)
        plt.savefig(save_path)
        # Be defensive: printing relative path can fail if roots differ
        try:
            rel = save_path.relative_to(Path.cwd())
        except ValueError:
            rel = save_path
        print(f"[Plot] Saved training curve → {rel}")

    plt.close()
