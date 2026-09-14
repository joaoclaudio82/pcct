"""Rule-based longitudinal response classification for research experiments."""
from __future__ import annotations


def classify_response(
    baseline_volume_ml: float,
    followup_volume_ml: float,
    baseline_q_mg: float,
    followup_q_mg: float,
    complete_threshold_ml: float = 0.05,
    partial_q_reduction_pct: float = 30.0,
    progression_q_increase_pct: float = 20.0,
) -> dict[str, float | str]:
    if followup_volume_ml <= complete_threshold_ml or followup_q_mg <= 0:
        category = "complete_response"
    else:
        delta_q_pct = 100.0 * (followup_q_mg - baseline_q_mg) / max(baseline_q_mg, 1e-6)
        if delta_q_pct <= -partial_q_reduction_pct:
            category = "partial_response"
        elif delta_q_pct >= progression_q_increase_pct:
            category = "progression"
        else:
            category = "stable"
        return {
            "category": category,
            "delta_q_pct": float(delta_q_pct),
            "delta_volume_pct": float(
                100.0 * (followup_volume_ml - baseline_volume_ml) / max(baseline_volume_ml, 1e-6)
            ),
        }
    return {
        "category": category,
        "delta_q_pct": -100.0,
        "delta_volume_pct": -100.0,
    }
