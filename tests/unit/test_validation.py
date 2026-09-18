import numpy as np
import pytest
from ai_photon_mvp.validation import spacing_zyx, paired_arrays

@pytest.mark.parametrize("value", [0, -1, float("nan"), (1, 2), (1, 2, 0)])
def test_invalid_spacing(value):
    with pytest.raises(ValueError):
        spacing_zyx(value)

def test_broadcasting_is_not_alignment():
    with pytest.raises(ValueError):
        paired_arrays(np.zeros((2, 2, 2)), np.zeros((1, 2, 2)))
