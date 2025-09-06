"""src/evaluate.py – evaluation & visualisation utilities."""
from __future__ import annotations

import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset

# heavy imports are lazy to avoid unnecessary runtime cost
FID_METRICS_AVAILABLE = True
try:
    from torch_fidelity import calculate_metrics  # pylint: disable=import-error
except ModuleNotFoundError:
    FID_METRICS_AVAILABLE = False

try:
    import open_clip  # pylint: disable=import-error
except ModuleNotFoundError:
    open_clip = None  # type: ignore


# -----------------------------------------------------------------------------
#  Utility – wrap a tensor batch into a ``torch.utils.data.Dataset`` so that
#  ``torch-fidelity`` (which expects a Dataset, path, or generator) accepts it.
# -----------------------------------------------------------------------------


class _TensorDataset(Dataset):
    """Minimal Dataset wrapper around a 4-D image tensor (N,C,H,W)."""

    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.tensor = tensor

    def __len__(self) -> int:  # noqa: D401 – Dataset protocol
        return self.tensor.shape[0]

    def __getitem__(self, idx):  # noqa: D401 – Dataset protocol
        return self.tensor[idx]


# -----------------------------------------------------------------------------
#  Metric wrappers – they all expose a `.compute()` method matching the original
# -----------------------------------------------------------------------------


class FIDEvaluator:  # pylint: disable=too-few-public-methods
    def __init__(self, dataloader: DataLoader, model: torch.nn.Module, device: str | torch.device = "cuda") -> None:
        self.dataloader = dataloader
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def compute(self) -> float:  # noqa: D401
        if not FID_METRICS_AVAILABLE:
            raise RuntimeError("`torch-fidelity` not installed – install to compute FID.")

        # gather 50 batches (~50*batch images) → quick but noisy estimate
        imgs: List[torch.Tensor] = []
        with torch.no_grad():
            for idx, (x, _) in enumerate(self.dataloader):
                imgs.append(x.cpu())
                if idx == 49:
                    break
        imgs_tensor = torch.cat(imgs, dim=0)

        # Convert tensors to Dataset objects acceptable by torch-fidelity
        dataset_real = _TensorDataset(imgs_tensor)
        dataset_fake = _TensorDataset(self._generate(imgs_tensor.shape[0]))
        try:
            metrics: Dict[str, float] = calculate_metrics(
                input1=dataset_real,
                input2=dataset_fake,
                fid=True,
                isc=False,
                kid=False,
                prc=False,
                verbose=False,
            )
            return float(metrics["frechet_inception_distance"])
        except ValueError as exc:  # Fallback – return a dummy large value so training continues
            print("[WARNING] torch-fidelity failed to compute FID – using placeholder value.\n", str(exc))
            return 999.0

    def _generate(self, n: int) -> torch.Tensor:
        """Generate *n* synthetic images with the diffusion model (very rough)."""
        self.model.eval()
        noise = torch.randn(n, 3, 256, 256, device=self.device)
        with torch.no_grad():
            for t in range(999, 0, -50):  # extremely coarse DDPM schedule
                out = self.model(noise, torch.full((n,), t, device=self.device))["sample"]
                noise = out.detach()
        return noise.cpu().clamp(-1, 1)


class InceptionScore:  # pragma: no cover – light wrapper around torch-fidelity
    def __init__(self, dataloader: DataLoader, model: torch.nn.Module):
        self.dataloader = dataloader
        self.model = model

    def compute(self) -> float:
        if not FID_METRICS_AVAILABLE:
            raise RuntimeError("`torch-fidelity` not installed – install to compute IS.")
        images = next(iter(self.dataloader))[0][:500].cpu()
        dataset = _TensorDataset(images)
        try:
            # torch-fidelity requires keyword arguments; positional triggers a TypeError
            metrics = calculate_metrics(input1=dataset, isc=True, fid=False, verbose=False)
            val = metrics["inception_score_mean"]
            return float(val)
        except ValueError as exc:
            print("[WARNING] torch-fidelity failed to compute IS – using placeholder value.\n", str(exc))
            return 0.0


class CLIPScore:  # very approximate – uses open_clip textual encoder if available
    def __init__(self, dataloader: DataLoader, model: torch.nn.Module, device: str | torch.device = "cuda") -> None:
        self.dataloader = dataloader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model

    def compute(self) -> float:
        if open_clip is None:
            raise RuntimeError("open_clip_torch not installed – cannot compute CLIPScore")
        clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        clip_model = clip_model.to(self.device)
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        images, _ = next(iter(self.dataloader))
        images = preprocess(images).to(self.device)
        with torch.no_grad():
            img_feat = clip_model.encode_image(images).float()
            txt_feat = clip_model.encode_text(tokenizer(["a photo"] * images.size(0))).float()
        score = (img_feat * txt_feat).sum(-1).mean() * 2.5  # scaling heuristic
        return float(score)


# -----------------------------------------------------------------------------
#  Visualisation helper
# -----------------------------------------------------------------------------

def save_training_curves(history: Dict[str, List[float]], fname: pathlib.Path) -> None:  # noqa: D401
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(6, 4))
    plt.plot(history["iter"], history["loss"], label="loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
