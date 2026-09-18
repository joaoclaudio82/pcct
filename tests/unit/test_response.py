import pytest
from ai_photon_mvp.longitudinal.response import classify_response


def test_complete_response_keeps_actual_changes():
    result = classify_response(1, 0.04, 2, 0.1)
    assert result["category"] == "complete_response"
    assert result["delta_q_pct"] == pytest.approx(-95)
    assert result["delta_volume_pct"] == pytest.approx(-96)


def test_zero_iodine_does_not_mean_lesion_disappeared():
    assert classify_response(1, 1, 2, 0)["category"] == "partial_response"


def test_missing_baseline_is_not_evaluable():
    result = classify_response(0, 1, 0, 2)
    assert result["category"] == "not_evaluable"
    assert result["delta_q_pct"] is None


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_response_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        classify_response(1, bad, 1, 1)
