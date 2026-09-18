import json

import pytest
from ai_photon_mvp.benchmark import check_against_reference


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 5])
def test_invalid_or_excessive_metric_fails(tmp_path, value):
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"maximum_allowed": {"error": 2}}))
    assert check_against_reference({"error": value}, reference)
