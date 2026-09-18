import numpy as np
from ai_photon_mvp import pipeline as pl


def test_qa_check_clean_volume_scores_100():
    volume = np.zeros((64, 64, 64), dtype=np.float32)
    result = pl.qa_check(volume, spacing=(1.0, 1.0, 1.0))

    assert result["score"] == 100
    assert result["alerts"] == []


def test_qa_check_detects_invalid_values():
    volume = np.zeros((64, 64, 64), dtype=np.float32)
    volume[0, 0, 0] = np.nan
    result = pl.qa_check(volume, spacing=(1.0, 1.0, 1.0))

    assert result["score"] < 100
    assert any("valores invalidos" in alert for alert in result["alerts"])


def test_match_lesions_pairs_nearest_components():
    a = np.zeros((20, 20, 20), dtype=np.int16)
    b = np.zeros_like(a)

    a[4:7, 4:7, 4:7] = 1
    a[12:15, 12:15, 12:15] = 2

    b[5:8, 4:7, 4:7] = 1
    b[12:15, 13:16, 12:15] = 2
    b[2:4, 15:17, 15:17] = 3

    pairs, new, gone = pl.match_lesions(a, b, max_dist_vox=4)

    assert {(i, j) for i, j, _ in pairs} == {(1, 1), (2, 2)}
    assert new == [3]
    assert gone == []


def test_qa_all_invalid_is_json_serializable():
    import json

    result = pl.qa_check(np.full((3, 3, 3), np.nan), (1, 1, 1))
    assert result["hu_range"] == (None, None)
    json.dumps(result, allow_nan=False)
