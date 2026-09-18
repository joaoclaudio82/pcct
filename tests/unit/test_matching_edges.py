import pytest
from ai_photon_mvp.longitudinal.matching import LesionDescriptor, match_lesions_hungarian


def test_duplicate_labels_are_rejected():
    a = LesionDescriptor(1, (0, 0, 0), 1)
    with pytest.raises(ValueError):
        match_lesions_hungarian([a, a], [a])


@pytest.mark.parametrize("distance", [0, -1, float("nan")])
def test_invalid_distance_even_for_empty_lists(distance):
    with pytest.raises(ValueError):
        match_lesions_hungarian([], [], max_distance_mm=distance)


def test_outside_gate_is_unmatched():
    a = LesionDescriptor(1, (0, 0, 0), 1)
    b = LesionDescriptor(2, (31, 0, 0), 1)
    assert match_lesions_hungarian([a], [b]) == ([], [2], [1])
