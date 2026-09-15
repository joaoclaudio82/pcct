"""Explicit classical baselines retained for scientific comparison."""
from __future__ import annotations

import numpy as np

from ai_photon_mvp import neuro
from ai_photon_mvp import pipeline as legacy


def segment_lungs(ct: np.ndarray, spacing: tuple[float, float, float]):
    return legacy.segment_lungs(ct, spacing)


def segment_liver(ct: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    return legacy.segment_liver_rough(ct, spacing)


def segment_brain_compartments(ct: np.ndarray, spacing: tuple[float, float, float]):
    result = neuro.atrophy_indices(ct, spacing)
    return result["masks"], result["metrics"]


def find_lung_nodule_candidates(ct: np.ndarray, lungs: np.ndarray, spacing):
    return legacy.nodule_candidates(ct, lungs, spacing)


def find_hypodense_liver_lesions(ct: np.ndarray, liver: np.ndarray, spacing):
    return legacy.hypodense_lesions(ct, liver, spacing)
