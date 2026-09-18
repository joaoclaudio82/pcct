"""Experimental response rules; these categories are not clinical RECIST."""
from __future__ import annotations

import math


def classify_response(
    baseline_volume_ml: float,
    followup_volume_ml: float,
    baseline_q_mg: float,
    followup_q_mg: float,
    complete_threshold_ml: float = 0.05,
    partial_q_reduction_pct: float = 30.0,
    progression_q_increase_pct: float = 20.0,
) -> dict[str, float | str | None]:
    values = (baseline_volume_ml, followup_volume_ml, baseline_q_mg, followup_q_mg,
              complete_threshold_ml, partial_q_reduction_pct, progression_q_increase_pct)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("response inputs must be finite and nonnegative")
    if not 0 < partial_q_reduction_pct <= 100 or progression_q_increase_pct <= 0:
        raise ValueError("response thresholds must be positive; reduction cannot exceed 100%")
    delta_q = 100 * (followup_q_mg - baseline_q_mg) / baseline_q_mg if baseline_q_mg else None
    delta_volume = 100 * (followup_volume_ml - baseline_volume_ml) / baseline_volume_ml if baseline_volume_ml else None
    if baseline_volume_ml == 0 or baseline_q_mg == 0:
        category = "not_evaluable"
    elif followup_volume_ml <= complete_threshold_ml:
        category = "complete_response"
    elif delta_q <= -partial_q_reduction_pct:
        category = "partial_response"
    elif delta_q >= progression_q_increase_pct:
        category = "progression"
    else:
        category = "stable"
    return {"category": category, "delta_q_pct": delta_q, "delta_volume_pct": delta_volume}
