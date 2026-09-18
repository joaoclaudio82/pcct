"""Quantitative spectral biomarker definitions for lesions."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ai_photon_mvp.spectral import iodine_load_mg
from ai_photon_mvp.validation import paired_arrays, spacing_zyx


@dataclass(slots=True)
class LesionBiomarker:
    volume_ml: float
    iodine_mean_mgml: float
    iodine_std_mgml: float
    iodine_load_mg: float
    hu_50kev: float | None = None
    hu_70kev: float | None = None
    spectral_slope_hu_per_kev: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def compute_lesion_biomarker(
    mask: np.ndarray,
    iodine_map: np.ndarray,
    spacing_mm: float | tuple[float, float, float],
    vmi50: np.ndarray | None = None,
    vmi70: np.ndarray | None = None,
) -> LesionBiomarker:
    mask, iodine_map = paired_arrays(mask, iodine_map)
    for vmi in (vmi50, vmi70):
        if vmi is not None:
            paired_arrays(mask, vmi)
    m = mask.astype(bool)
    if not m.any():
        raise ValueError("a máscara da lesão está vazia")
    voxel_ml = np.prod(spacing_zyx(spacing_mm)) / 1000.0
    i = iodine_map[m].astype(float)
    h50 = float(vmi50[m].mean()) if vmi50 is not None else None
    h70 = float(vmi70[m].mean()) if vmi70 is not None else None
    slope = (h50 - h70) / 20.0 if h50 is not None and h70 is not None else None
    return LesionBiomarker(
        volume_ml=float(m.sum() * voxel_ml),
        iodine_mean_mgml=float(i.mean()),
        iodine_std_mgml=float(i.std()),
        iodine_load_mg=iodine_load_mg(iodine_map, m, spacing_mm),
        hu_50kev=h50,
        hu_70kev=h70,
        spectral_slope_hu_per_kev=slope,
    )
