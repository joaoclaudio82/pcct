"""Global lesion matching using geometry and optional quantitative features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(slots=True)
class LesionDescriptor:
    label: int
    centroid_mm: tuple[float, float, float]
    volume_ml: float
    iodine_mean: float | None = None
    feature_vector: tuple[float, ...] = ()


def _relative_difference(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-6)
    return abs(a - b) / denom


def match_lesions_hungarian(
    baseline: list[LesionDescriptor],
    followup: list[LesionDescriptor],
    max_distance_mm: float = 30.0,
    weights: dict[str, float] | None = None,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if not np.isfinite(max_distance_mm) or max_distance_mm <= 0:
        raise ValueError("max_distance_mm must be finite and positive")
    weights = {"distance": 0.55, "volume": 0.25, "iodine": 0.20} if weights is None else weights
    if not weights or set(weights) - {"distance", "volume", "iodine", "features"}:
        raise ValueError("provide supported matching weights")
    if any(not np.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be finite, nonnegative and have a positive sum")
    total = sum(weights.values())
    weights = {key: value / total for key, value in weights.items()}
    for lesions in (baseline, followup):
        if len({lesion.label for lesion in lesions}) != len(lesions):
            raise ValueError("lesion labels must be unique within each examination")
        for lesion in lesions:
            centroid = np.asarray(lesion.centroid_mm)
            if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
                raise ValueError("centroids must contain three finite coordinates")
            if not np.isfinite(lesion.volume_ml) or lesion.volume_ml < 0:
                raise ValueError("lesion volumes must be finite and nonnegative")
            if lesion.iodine_mean is not None and not np.isfinite(lesion.iodine_mean):
                raise ValueError("iodine features must be finite")
            if not np.all(np.isfinite(lesion.feature_vector)):
                raise ValueError("feature vectors must be finite")
    if not baseline or not followup:
        return [], [x.label for x in followup], [x.label for x in baseline]

    cost = np.full((len(baseline), len(followup)), 1e6, dtype=float)
    for i, a in enumerate(baseline):
        for j, b in enumerate(followup):
            dist = float(np.linalg.norm(np.asarray(a.centroid_mm) - np.asarray(b.centroid_mm)))
            if dist > max_distance_mm:
                continue
            c = weights.get("distance", 0.0) * dist / max_distance_mm
            c += weights.get("volume", 0.0) * _relative_difference(a.volume_ml, b.volume_ml)
            if a.iodine_mean is not None and b.iodine_mean is not None:
                c += weights.get("iodine", 0.0) * _relative_difference(a.iodine_mean, b.iodine_mean)
            if a.feature_vector and b.feature_vector and len(a.feature_vector) == len(b.feature_vector):
                c += weights.get("features", 0.0) * float(
                    np.linalg.norm(np.asarray(a.feature_vector) - np.asarray(b.feature_vector))
                )
            cost[i, j] = c

    rows, cols = linear_sum_assignment(cost)
    pairs = []
    matched_a, matched_b = set(), set()
    for i, j in zip(rows, cols, strict=True):
        if cost[i, j] >= 1e5:
            continue
        pairs.append((baseline[i].label, followup[j].label, float(cost[i, j])))
        matched_a.add(baseline[i].label)
        matched_b.add(followup[j].label)
    new = [x.label for x in followup if x.label not in matched_b]
    gone = [x.label for x in baseline if x.label not in matched_a]
    return pairs, new, gone
