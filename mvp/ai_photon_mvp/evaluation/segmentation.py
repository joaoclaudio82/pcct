"""Segmentation metrics for quantitative validation."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    p, r = pred.astype(bool), ref.astype(bool)
    denom = p.sum() + r.sum()
    return 1.0 if denom == 0 else float(2 * np.logical_and(p, r).sum() / denom)


def iou(pred: np.ndarray, ref: np.ndarray) -> float:
    p, r = pred.astype(bool), ref.astype(bool)
    union = np.logical_or(p, r).sum()
    return 1.0 if union == 0 else float(np.logical_and(p, r).sum() / union)


def precision_recall(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    p, r = pred.astype(bool), ref.astype(bool)
    tp = np.logical_and(p, r).sum()
    fp = np.logical_and(p, ~r).sum()
    fn = np.logical_and(~p, r).sum()
    precision = 1.0 if tp + fp == 0 else float(tp / (tp + fp))
    recall = 1.0 if tp + fn == 0 else float(tp / (tp + fn))
    return precision, recall


def _surface(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    return m ^ ndi.binary_erosion(m)


def surface_distances_mm(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    ps, rs = _surface(pred), _surface(ref)
    if not ps.any() or not rs.any():
        return np.array([], dtype=float)
    d_to_ref = ndi.distance_transform_edt(~rs, sampling=spacing)
    d_to_pred = ndi.distance_transform_edt(~ps, sampling=spacing)
    return np.concatenate([d_to_ref[ps], d_to_pred[rs]]).astype(float)


def hd95(pred: np.ndarray, ref: np.ndarray, spacing: tuple[float, float, float]) -> float:
    d = surface_distances_mm(pred, ref, spacing)
    return float("nan") if d.size == 0 else float(np.percentile(d, 95))


def average_surface_distance(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    d = surface_distances_mm(pred, ref, spacing)
    return float("nan") if d.size == 0 else float(d.mean())


def evaluate_segmentation(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, float]:
    precision, recall = precision_recall(pred, ref)
    return {
        "dice": dice(pred, ref),
        "iou": iou(pred, ref),
        "precision": precision,
        "recall": recall,
        "hd95_mm": hd95(pred, ref, spacing),
        "asd_mm": average_surface_distance(pred, ref, spacing),
    }
