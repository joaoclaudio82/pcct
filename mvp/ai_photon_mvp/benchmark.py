"""Fast deterministic scientific regression benchmark for AI-Photon.

The benchmark deliberately uses a compact synthetic homogeneous ROI so it can
run on every commit. It checks the quantitative chain used by the larger
phantom experiments: VMI simulation -> material decomposition -> iodine
concentration -> iodine load Q.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import spectral as sp


def run_spectral_benchmark(
    shape: tuple[int, int, int] = (16, 32, 32),
    iodine_mgml: float = 3.5,
    spacing_mm: float = 1.5,
    seed: int = 2026,
) -> dict[str, float]:
    """Run a small deterministic full-dose quantitative benchmark."""
    water = np.ones(shape, dtype=np.float32)
    iodine = np.full(shape, iodine_mgml, dtype=np.float32)
    mask = np.ones(shape, dtype=bool)

    rng = np.random.default_rng(seed)
    vmi50 = sp.simulate_vmi(water, iodine, 50, dose_factor=1.0, rng=rng)
    vmi70 = sp.simulate_vmi(water, iodine, 70, dose_factor=1.0, rng=rng)
    _, iodine_est = sp.decompose(vmi50, vmi70)

    concentration_est = float(iodine_est[mask].mean())
    concentration_error_pct = abs(concentration_est - iodine_mgml) / iodine_mgml * 100.0

    q_true = sp.iodine_load_mg(iodine, mask, spacing_mm)
    q_est = sp.iodine_load_mg(iodine_est, mask, spacing_mm)
    q_error_pct = abs(q_est - q_true) / q_true * 100.0

    return {
        "iodine_true_mgml": round(iodine_mgml, 6),
        "iodine_estimated_mgml": round(concentration_est, 6),
        "iodine_error_pct": round(float(concentration_error_pct), 6),
        "q_true_mg": round(float(q_true), 6),
        "q_estimated_mg": round(float(q_est), 6),
        "q_error_pct": round(float(q_error_pct), 6),
    }


def check_against_reference(
    metrics: dict[str, float], reference_path: str | Path
) -> list[str]:
    """Return human-readable failures against the stored tolerances."""
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    failures: list[str] = []

    for metric, maximum in reference["maximum_allowed"].items():
        value = float(metrics[metric])
        if not np.isfinite(value) or value > float(maximum):
            failures.append(f"{metric}={value:.6f} exceeds maximum {maximum}")

    return failures


if __name__ == "__main__":
    result = run_spectral_benchmark()
    print(json.dumps(result, indent=2))
