"""src/main.py – single entry-point orchestrating the whole experiment."""
import pathlib, sys, json, yaml, torch
from typing import List, Dict, Any

from .train import run_single
from .evaluate import plot_fid

# -----------------  safety check  ------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for these experiments – aborting.")

# -----------------  load configuration  ------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG_FILE = ROOT / "config" / "exp.yaml"
CFG = yaml.safe_load(CFG_FILE.read_text())

# -----------------  orchestrate  -------------------------------

def main():
    all_results: List[Dict[str,Any]] = []
    for model_key in ("fft_dit", "dit"):
        for seed in CFG["seeds"]:
            res = run_single(model_key, seed)
            all_results.append(res)

    # plotting & consolidated JSON
    plot_fid(all_results)
    (ROOT/"results"/"all_results.json").write_text(json.dumps(all_results, indent=2))
    print("\n=========  FINAL RESULTS  =========")
    print(json.dumps(all_results, indent=2))
    print("Figures:\n  - fid_exp1.pdf")

if __name__ == "__main__":
    main()
