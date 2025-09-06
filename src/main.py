"""src/main.py – single entry-point orchestrating the whole experiment (paths fixed to iteration28)"""
import pathlib, sys, json, yaml, torch
from typing import List, Dict, Any

from .train import run_single
from .evaluate import plot_fid

# -----------------  safety check  ------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for these experiments – aborting.")

# -----------------  load configuration  ------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG_FILE = ROOT / "config" / "config.yaml"
CFG = yaml.safe_load(CFG_FILE.read_text())

# research directories (iteration28)
RESEARCH_DIR = ROOT / ".research" / "iteration28"
IMG_DIR = RESEARCH_DIR / "images"
for p in (RESEARCH_DIR, IMG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# -----------------  orchestrate  -------------------------------

def main():
    all_results: List[Dict[str, Any]] = []
    for model_key in ("fft_dit", "dit"):
        for seed in CFG["seeds"]:
            res = run_single(model_key, seed)
            all_results.append(res)

    # plotting & consolidated JSON
    plot_fid(all_results)
    (RESEARCH_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))
    print("\n=========  FINAL RESULTS  =========")
    print(json.dumps(all_results, indent=2))
    print("Figures:\n  - fid_exp1.pdf")

if __name__ == "__main__":
    main()
