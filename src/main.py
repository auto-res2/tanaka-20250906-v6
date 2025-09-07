from __future__ import annotations

"""Entry-point.  Orchestrates the complete experiment suite for all seeds."""

import json
import pathlib
import time
from datetime import datetime
from typing import Dict

import yaml

from .train import ROOT_RESULTS_DIR, run_single

# -----------------------------------------------------------------------------
# Environment logging (kept here to avoid an extra file)
# -----------------------------------------------------------------------------


def _log_environment(out_path: pathlib.Path) -> None:
    """Write minimal reproducibility snapshot (git hash, torch/cuDNN versions…)."""

    import subprocess
    import sys

    info: Dict[str, str] = {}
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            if (pathlib.Path.cwd() / ".git").exists()
            else "n/a"
        )
    except Exception:  # pragma: no cover – git not available
        git_hash = "n/a"

    info["git_commit"] = git_hash
    info["python"] = sys.version
    info["datetime_utc"] = datetime.utcnow().isoformat() + "Z"
    out_path.write_text(json.dumps(info, indent=2))


# -----------------------------------------------------------------------------
# Config helper
# -----------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_cfg() -> dict:
    cfg_path = ROOT / "config" / "config.yaml"
    return yaml.safe_load(cfg_path.read_text())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:  # noqa: D401
    cfg = _load_cfg()

    run_id = cfg["run_id"] + "_" + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # ------------------------------------------------------------------
    # All outputs – JSON, traces, figs – must live under .research/iteration79/
    # ------------------------------------------------------------------
    out_dir = ROOT_RESULTS_DIR  # fixed path (iteration 79)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log_environment(out_dir / "env.json")

    start = time.perf_counter()
    all_results: Dict[str, dict] = {}

    for seed in cfg["seed_list"]:
        res = run_single(cfg, seed, out_dir)
        all_results[str(seed)] = res

        # ------------------------------------------------------------------
        # Persist per-seed JSON & echo to stdout for CI verification
        # ------------------------------------------------------------------
        result_path = out_dir / f"seed_{seed}.json"
        result_path.write_text(json.dumps(res, indent=2))

        print("\n===================== Experiment description =====================")
        print(f"Run-ID: {run_id}    Seed: {seed}")
        print("Dataset:", cfg["dataset"]["name"], "| Model variants:", ", ".join(res.keys()))
        print("===================== Numerical results ========================")
        print(json.dumps(res, indent=2))
        print("===================== Figures ==================================")
        for m in res.values():
            for fig in m["figures"]:
                print("Figure saved:", fig)

    # Aggregate results (all seeds)
    (out_dir / "aggregate.json").write_text(json.dumps(all_results, indent=2))
    elapsed = round(time.perf_counter() - start, 2)
    print("================================================================")
    print("All experiments finished in", elapsed, "sec")


if __name__ == "__main__":  # pragma: no cover
    main()
