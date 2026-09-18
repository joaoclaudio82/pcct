import numpy as np
import pytest
from ai_photon_mvp.uncertainty.segmentation import monte_carlo_volume_interval, geometric_volume_interval

@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan")])
def test_invalid_probability(p):
    with pytest.raises(ValueError):
        monte_carlo_volume_interval(np.full((2, 2, 2), p), (1, 1, 1))

def test_insufficient_samples():
    with pytest.raises(ValueError):
        monte_carlo_volume_interval(np.ones((2, 2, 2)), (1, 1, 1), n_samples=1)
