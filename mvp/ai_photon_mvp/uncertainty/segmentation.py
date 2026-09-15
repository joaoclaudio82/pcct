"""Segmentation uncertainty via geometric perturbation and Monte Carlo masks."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def geometric_volume_interval(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    iterations: int = 1,
) -> tuple[float, float]:
    voxel_ml = float(np.prod(spacing) / 1000.0)
    lo = ndi.binary_erosion(mask, iterations=iterations).sum() * voxel_ml
    hi = ndi.binary_dilation(mask, iterations=iterations).sum() * voxel_ml
    return float(lo), float(hi)


def monte_carlo_volume_interval(
    probability: np.ndarray,
    spacing: tuple[float, float, float],
    n_samples: int = 200,
    seed: int = 2026,
) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(seed)
    p = np.clip(probability.astype(float), 0.0, 1.0)
    voxel_ml = float(np.prod(spacing) / 1000.0)
    volumes = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        volumes[i] = float((rng.random(p.shape) < p).sum() * voxel_ml)
    lo, hi = np.percentile(volumes, [2.5, 97.5])
    return {
        "mean_ml": float(volumes.mean()),
        "std_ml": float(volumes.std(ddof=1)),
        "ci95_ml": [float(lo), float(hi)],
    }
