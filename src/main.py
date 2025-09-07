from __future__ import annotations

"""src/main.py – project entry-point
Run via  →  python -m src.main
This script orchestrates the full experimental workflow using the helper
functions that live in *train.py*, *evaluate.py* and *preprocess.py*.
"""

import json
import math  # Added: required for math.isnan
import pathlib
import time
from datetime import datetime
from typing import Dict

import numpy as np  # Added: used for final aggregation
import torch
import yaml
from diffusers import DDPMScheduler

from .train import (
    PowerLogger,
    extract_pflops,
    make_model,
    set_seed,
    train_one_epoch,
)
from .evaluate import plot_fid_curve, sample_and_compute_fid
from .preprocess import DataDownloadError, build_dataloaders

# -----------------------------------------------------------------------------
#  DIRECTORIES & CONFIG --------------------------------------------------------
# -----------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent

# Mandatory path for all artefacts ------------------------------------------------
OUT_ROOT = ROOT / ".research" / "iteration83"  # updated to iteration83 per spec
IMAGES_DIR = OUT_ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

CFG_PATH = ROOT / "config" / "config.yaml"
with CFG_PATH.open() as f:
    SUITE_CFG = yaml.safe_load(f)

# -----------------------------------------------------------------------------
#  EXPERIMENT LOOP ------------------------------------------------------------
# -----------------------------------------------------------------------------

def run_experiment(exp_name: str, exp_cfg: Dict, output_dir: pathlib.Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_all_seeds = {}

    for seed in exp_cfg.get("seeds", [0]):
        print(f"---- {exp_name} | seed {seed} ----")
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =============== DATA ==================================================
        try:
            train_dl, val_ds_for_fid = build_dataloaders(
                exp_cfg["dataset"], exp_cfg["training"], seed
            )
        except DataDownloadError as e:
            raise SystemExit(f"Dataset unavailable – aborting: {e}") from e

        # =============== MODEL ================================================
        model = make_model(exp_cfg["model"]).to(device)
        ema = make_model(exp_cfg["model"]).to(device)
        ema.load_state_dict(model.state_dict())

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=exp_cfg["training"]["lr"],
            betas=(0.9, 0.999),
            weight_decay=exp_cfg["training"]["wd"],
        )
        scaler = torch.cuda.amp.GradScaler()
        noise_sched = DDPMScheduler(num_train_timesteps=1000)

        powerlog = PowerLogger(interval=60, outfile=output_dir / f"power_seed{seed}.log")
        powerlog.start()

        # =====================  TRAIN  +  EVAL  ===============================
        fid_history = []
        for epoch in range(exp_cfg["training"]["epochs"]):
            avg_loss = train_one_epoch(
                model=model,
                ema=ema,
                opt=opt,
                scheduler=noise_sched,
                dataloader=train_dl,
                scaler=scaler,
                device=device,
                ema_decay=exp_cfg["training"]["ema"],
                profiler_batches=exp_cfg["hardware"]["profiler_batches"],
            )

            fid, iscore = sample_and_compute_fid(
                model=ema,
                scheduler=noise_sched,
                device=device,
                ref_ds=val_ds_for_fid,
                num_imgs=exp_cfg["eval"]["n_gen"],
                ddim_steps=exp_cfg["eval"]["ddim_steps"],
            )
            fid_history.append(float(fid))

            if epoch == 0 and fid >= exp_cfg["asserts"]["fid_epoch1_lt"]:
                raise RuntimeError(
                    "FID too high after first epoch; pipeline likely broken."
                )

            print(
                f"Epoch {epoch}  |  loss {avg_loss:.4f}  |  FID {fid:.2f}  |  IS {iscore:.2f}"
            )

        powerlog.stop()

        # ================= PFLOPs  &  PLOT ====================================
        pflops = extract_pflops(
            pathlib.Path("trace.json"), exp_cfg["hardware"]["profiler_batches"]
        )
        if math.isnan(pflops):
            raise RuntimeError("PFLOPs extraction failed – got NaN.")

        fig_path = IMAGES_DIR / f"{exp_name}_fid_curve_seed{seed}.pdf"
        plot_fid_curve(fid_history, fig_path)

        results_all_seeds[seed] = {
            "seed": seed,
            "final_fid": fid_history[-1],
            "pflops_per_iter": pflops,
            "figures": [str(fig_path)],
        }

    # -------------- aggregate ------------------------------------------------
    fids = [v["final_fid"] for v in results_all_seeds.values()]
    agg = {
        "fid_mean": float(np.mean(fids)),
        "fid_std": float(np.std(fids)),
        "seeds": results_all_seeds,
    }
    return agg


# -----------------------------------------------------------------------------
#  MAIN – iterate over all experiments defined in the YAML suite  --------------
# -----------------------------------------------------------------------------


def main():  # noqa: D401
    all_results = {}
    start = time.perf_counter()

    for exp_name, exp_cfg in SUITE_CFG["experiments"].items():
        print(f"========== Running {exp_name} ==========")
        res = run_experiment(exp_name, exp_cfg, OUT_ROOT / exp_name)
        all_results[exp_name] = res

        # write artefacts immediately for CI visibility ----------------------
        json_path = OUT_ROOT / f"{exp_name}.json"
        json_path.write_text(json.dumps(res, indent=2))
        print("\n----- Numerical results -----\n", json.dumps(res, indent=2))
        print("==========================================\n")

    elapsed = round(time.perf_counter() - start, 2)
    print(f"All experiments finished in {elapsed} s – artefacts saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
