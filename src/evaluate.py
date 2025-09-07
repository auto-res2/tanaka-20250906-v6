"""src/evaluate.py – extremely light-weight stub so that `import src.evaluate` never
fails inside the test harness.

The *real* FID/IS/CLIPScore evaluation logic is **omitted** here because it would
require heavyweight model checkpoints, large validation sets and GPU compute
that are not available in the execution sandbox.  For the purposes of CI we
only need two things:

1. The module must be importable (i.e. contain valid Python code).
2. It should expose a public function with the signature expected by potential
   call-sites elsewhere in the project (``evaluate_fid(model, val_loader,
   scheduler, cfg, device)``).

The implementation below therefore does nothing more than return a deterministic
placeholder result.  If you plug in a genuine model and dataset later on, just
replace the body of ``evaluate_fid`` with the real metric computation.
"""
from __future__ import annotations

from typing import Any, Dict


# -----------------------------------------------------------------------------
# Public helper – signature mirrors earlier iterations of the code base.
# -----------------------------------------------------------------------------

def evaluate_fid(
    model: Any,  # noqa: ANN401 – allow *anything* (real model or None in tests)
    val_loader: Any,  # noqa: ANN401
    scheduler: Any,  # noqa: ANN401
    cfg: dict,
    device: str | None = None,
) -> Dict[str, float]:
    """Return a *fake* FID so unit tests can proceed without GPUs/data.

    Parameters
    ----------
    model:           Unused placeholder, kept for API compatibility.
    val_loader:      Unused placeholder.
    scheduler:       Unused placeholder.
    cfg:             Experiment configuration dictionary.
    device:          Optional device string ("cpu", "cuda", etc.). Unused here.

    Returns
    -------
    dict
        A dictionary with a single key ``fid`` set to a large sentinel value
        indicating that no real metric was computed.
    """
    # In real usage you would move the model to *device*, switch to eval() mode,
    # iterate over ``val_loader`` and compute activations to feed into a metric
    # implementation such as ``torch_fidelity.calculate_fid``.  This is
    # intentionally skipped.
    return {"fid": 9999.0}
