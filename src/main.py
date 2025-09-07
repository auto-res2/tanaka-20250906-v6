from __future__ import annotations

"""src/main.py – light-weight entry-point used by the unit-tests.

IMPORTANT – the rubric now mandates that *all* artefacts for this iteration
are stored under ``.research/iteration39/`` (with images, if any, further nested
under ``.research/iteration39/images``).  This file therefore mirrors the
previous implementation but writes JSON outputs to the new location so the test
harness can discover them.
"""

import json
import pathlib
import time
from datetime import datetime

import yaml

from . import train  # local import – uses the stub provided in src/train.py

# -----------------------------------------------------------------------------
# Configuration & global paths
# -----------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG_FILE = ROOT / "config" / "config.yaml"
ARTEFACT_DIR = ROOT / ".research" / "iteration39"  # <-- changed from iteration38
IMAGE_DIR = ARTEFACT_DIR / "images"  # future-proof: mandated path for any figures
ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _load_cfg() -> dict:  # noqa: D401 – simple helper
    if not CFG_FILE.exists():  # pragma: no cover – should not happen in tests
        raise FileNotFoundError(f"Config file missing: {CFG_FILE}")
    return yaml.safe_load(CFG_FILE.read_text())


# -----------------------------------------------------------------------------
# Main orchestration – intentionally minimal
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    cfg = _load_cfg()

    all_results = []
    start_wall = time.time()

    for seed in cfg.get("seeds", []):
        # Run (stub) training – returns dict
        run_info = train.run_single(cfg, seed=seed)
        all_results.append(run_info)

        # ------------------------------------------------------------------
        # Persist artefact: one JSON per seed, mandatory directory enforced
        # ------------------------------------------------------------------
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = ARTEFACT_DIR / f"seed_{seed}_{ts}.json"
        out_path.write_text(json.dumps(run_info, indent=2))

        # ALSO print to stdout for verification (as per instructions)
        print(out_path.read_text())

    print("Finished in", round(time.time() - start_wall, 2), "sec")


# Allow execution via `python -m src.main`
if __name__ == "__main__":  # pragma: no cover
    main()