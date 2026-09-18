import numpy as np
import pytest
from ai_photon_mvp import spectral as sp


@pytest.mark.parametrize("dose", [0, -1, float("nan"), float("inf")])
def test_invalid_dose(dose):
    a = np.ones((2, 2, 2))
    with pytest.raises(ValueError):
        sp.simulate_vmi(a, a, 50, dose_factor=dose)


def test_decomposition_rejects_misaligned_vmi():
    with pytest.raises(ValueError):
        sp.decompose(np.ones((2, 2, 2)), np.ones((1, 2, 2)))


def test_anisotropic_iodine_load_and_integer_mask():
    a = np.full((2, 2, 2), 2.0)
    assert sp.iodine_load_mg(a, np.ones_like(a, dtype=int), (2, 3, 4)) == pytest.approx(0.384)
