"""src/main.py – orchestrates the complete FFT-DiT experiment suite.
Run with  :  python -m src.main
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any, Dict, List

import yaml

from .evaluate import CLIPScore, FIDEvaluator, InceptionScore, save_training_curves
from .preprocess import build_dataloader
from .train import RESULT_DIR, Trainer, create_model, save_json, set_seeds

# -----------------------------------------------------------------------------
#  Config handling (YAML on disk so the user can edit between runs)
# -----------------------------------------------------------------------------
CFG_PATH = pathlib.Path("config")
CFG_PATH.mkdir(exist_ok=True)
CFG_FILE = CFG_PATH / "config.yaml"
DEFAULT_CFG = {
    "exp": "exp1",  # only exp1 implemented in this refactor
    "seed": 13,
    "num_seeds": 3,
    "device": "cuda",
    "global_batch": 2048,
    "img_size": 256,
}
if not CFG_FILE.exists():
    with CFG_FILE.open("w") as handle:
        yaml.safe_dump(DEFAULT_CFG, handle)

with CFG_FILE.open("r") as handle:
    CFG = yaml.safe_load(handle)


# -----------------------------------------------------------------------------
#  Experiment specifics (mirrors original single-file script)
# -----------------------------------------------------------------------------
MODELS = {
    "fft_dit": {"params": 480_000_000, "target_fid": 4.0},
    "dit_xl": {"params": 675_000_000, "target_fid": 4.0},
    "u_vit": {"params": 620_000_000, "target_fid": 4.0},
    "wavelet": {"params": 510_000_000, "target_fid": 4.0},
}

DATASET_KEY = "imagenet"  # only ImageNet handled in this trimmed refactor


# -----------------------------------------------------------------------------
#  Main entry – loops over model×seed grid and aggregates JSON results
# -----------------------------------------------------------------------------

def _single_run(model_key: str, seed: int) -> Dict[str, Any]:  # noqa: D401
    set_seeds(seed)

    train_loader, val_loader = build_dataloader(global_batch=CFG["global_batch"], img_size=CFG["img_size"])

    model = create_model(img_size=CFG["img_size"], device=CFG["device"])

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        target_fid=MODELS[model_key]["target_fid"],
        seed=seed,
        exp_name=f"exp1_{DATASET_KEY}_{model_key}_s{seed}",
        device=CFG["device"],
    )

    trainer.fit()

    fid = FIDEvaluator(val_loader, model).compute()
    inc = InceptionScore(val_loader, model).compute()
    clip = CLIPScore(val_loader, model).compute()

    res = dict(
        experiment="exp1",
        dataset=DATASET_KEY,
        model=model_key,
        seed=seed,
        wall_clock_h=getattr(trainer, "wall_clock_h", None),
        fid=fid,
        inception=inc,
        clip_score=clip,
        params=MODELS[model_key]["params"],
    )

    # save individual result file – path changed to iteration5 automatically via RESULT_DIR import
    ts = int(time.time())
    out_path = RESULT_DIR / f"exp1_{DATASET_KEY}_{model_key}_s{seed}_{ts}.json"
    save_json(res, out_path)

    # Immediately print JSON contents for verification as per mandatory rule
    print("\n--- Individual Result JSON ---")
    print(json.dumps(res, indent=2))
    print("------------------------------\n")

    # training loss curve
    curve_path = RESULT_DIR / "images" / f"training_loss_{model_key}_{DATASET_KEY}.pdf"
    save_training_curves(trainer.history, curve_path)

    return res


def main() -> None:
    print("=================  FFT-DiT EXPERIMENT SUITE  =================")
    print(json.dumps(CFG, indent=2))

    results: List[Dict[str, Any]] = []
    seeds = [13, 29, 87][: CFG["num_seeds"]]
    for model_key in MODELS:
        for seed in seeds:
            print(f"\n=====  {model_key.upper()}  |  seed={seed}  =====")
            res = _single_run(model_key, seed)
            results.append(res)

    # consolidated results file (timestamped)
    all_path = RESULT_DIR / f"exp1_results_{int(time.time())}.json"
    save_json(results, all_path)

    print("\n===========  RESULTS (JSON)  ===========")
    print(json.dumps(results, indent=2))
    print("========================================")
    print(f"Saved consolidated results  ->  {all_path.relative_to(pathlib.Path.cwd())}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
