import numpy as np
import pytest
from ai_photon_mvp.spectral_biomarkers import compute_lesion_biomarker

def test_anisotropic_biomarker():
    a = np.ones((2, 2, 2))
    result = compute_lesion_biomarker(a, a * 2, (2, 3, 4), a * 100, a * 60)
    assert result.volume_ml == pytest.approx(0.192)
    assert result.iodine_load_mg == pytest.approx(0.384)
    assert result.spectral_slope_hu_per_kev == 2
