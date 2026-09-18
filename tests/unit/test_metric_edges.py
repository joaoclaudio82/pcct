import numpy as np
import pytest
from ai_photon_mvp.evaluation.segmentation import dice, hd95

def test_metrics_reject_broadcast_masks():
    with pytest.raises(ValueError):
        dice(np.zeros((2, 2, 2)), np.zeros((1, 2, 2)))

def test_surface_distance_respects_anisotropy():
    a = np.zeros((5, 5, 5)); b = a.copy()
    a[1, 2, 2] = 1; b[2, 2, 2] = 1
    assert hd95(a, b, (3, 1, 1)) == 3
