import numpy as np
import pytest
from ai_photon_mvp.radiomics.features import basic_features

def test_features_physical_volume_and_extent():
    a = np.ones((2, 3, 4))
    result = basic_features(a, a, (4, 2, 1))
    assert result["volume_ml"] == pytest.approx(0.192)
    assert result["elongation_bbox"] == 2

def test_invalid_spacing_even_for_empty_mask():
    a = np.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        basic_features(a, a, (0, 1, 1))
