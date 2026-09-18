"""Radiomic-style quantitative features with optional PyRadiomics integration."""
from __future__ import annotations

import numpy as np

from ai_photon_mvp.validation import paired_arrays, spacing_zyx


def basic_features(
    image: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, float]:
    image, mask = paired_arrays(image, mask)
    spacing = spacing_zyx(spacing)
    m = mask.astype(bool)
    if not m.any():
        return {}
    values = image[m].astype(float)
    voxel_ml = float(np.prod(spacing) / 1000.0)
    idx = np.argwhere(m)
    ext_mm = (idx.max(axis=0) - idx.min(axis=0) + 1) * np.asarray(spacing)
    return {
        "volume_ml": float(m.sum() * voxel_ml),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(values.min()),
        "max": float(values.max()),
        "elongation_bbox": float(ext_mm.max() / max(ext_mm.min(), 1e-6)),
    }


def pyradiomics_features(image_sitk, mask_sitk) -> dict[str, float]:
    """Extract PyRadiomics features when the optional dependency is installed."""
    try:
        from radiomics import featureextractor
    except ImportError as exc:
        raise RuntimeError("instale o extra 'radiomics' para usar PyRadiomics") from exc
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("glcm")
    result = extractor.execute(image_sitk, mask_sitk)
    return {
        key: float(value)
        for key, value in result.items()
        if key.startswith("original_") and np.isscalar(value)
    }
