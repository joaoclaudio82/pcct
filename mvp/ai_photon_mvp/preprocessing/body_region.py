"""Body-region detection prioritizing DICOM metadata over image heuristics."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ai_photon_mvp.io.image import MedicalVolume


def detect_body_region(volume: MedicalVolume) -> str:
    text = " ".join(
        str(volume.metadata.get(k, ""))
        for k in ("BodyPartExamined", "StudyDescription", "SeriesDescription")
    ).upper()
    if any(t in text for t in ("CHEST", "THORAX", "LUNG", "TORAX", "TÓRAX")):
        return "torax"
    if any(t in text for t in ("HEAD", "BRAIN", "CRANI", "NEURO", "CEREBR")):
        return "neuro"
    if any(t in text for t in ("ABDOM", "LIVER", "HEPATIC", "FIGADO", "FÍGADO")):
        return "abdome"
    return detect_body_region_from_image(volume.data, volume.spacing)


def detect_body_region_from_image(ct: np.ndarray, spacing: tuple[float, float, float]) -> str:
    body = np.stack([ndi.binary_fill_holes(b) for b in (ct > -400)])
    core = ndi.binary_erosion(body, iterations=3)
    lunglike = (ct > -950) & (ct < -600) & core
    per_slice = lunglike.sum(axis=(1, 2)) / np.maximum(core.sum(axis=(1, 2)), 1)
    if float(np.mean(per_slice > 0.08)) > 0.35:
        return "torax"
    frac_bone = float(np.mean(ct > 200))
    if frac_bone > 0.04 and ct.shape[0] * spacing[0] < 350:
        return "neuro"
    return "abdome"
