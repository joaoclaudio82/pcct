import numpy as np
from ai_photon_mvp.evaluation.segmentation import dice, hd95, iou
from ai_photon_mvp.io.image import MedicalVolume
from ai_photon_mvp.longitudinal.matching import LesionDescriptor, match_lesions_hungarian
from ai_photon_mvp.uncertainty.segmentation import monte_carlo_volume_interval


def test_medical_volume_roundtrip_geometry():
    volume = MedicalVolume(
        data=np.zeros((4, 5, 6), dtype=np.float32),
        spacing=(2.0, 1.5, 1.0),
        origin=(3.0, 2.0, 1.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    recovered = MedicalVolume.from_sitk(volume.to_sitk())
    assert recovered.data.shape == volume.data.shape
    assert np.allclose(recovered.spacing, volume.spacing)
    assert np.allclose(recovered.origin, volume.origin)


def test_segmentation_metrics_identity_and_shift():
    ref = np.zeros((20, 20, 20), dtype=bool)
    ref[5:15, 5:15, 5:15] = True
    pred = ref.copy()
    assert dice(pred, ref) == 1.0
    assert iou(pred, ref) == 1.0
    assert hd95(pred, ref, (1.0, 1.0, 1.0)) == 0.0


def test_hungarian_matching_uses_global_assignment():
    baseline = [
        LesionDescriptor(1, (0.0, 0.0, 0.0), 2.0, 2.0),
        LesionDescriptor(2, (20.0, 0.0, 0.0), 4.0, 3.0),
    ]
    followup = [
        LesionDescriptor(10, (1.0, 0.0, 0.0), 2.1, 2.1),
        LesionDescriptor(20, (19.0, 0.0, 0.0), 4.1, 3.1),
    ]
    pairs, new, gone = match_lesions_hungarian(baseline, followup)
    assert {(a, b) for a, b, _ in pairs} == {(1, 10), (2, 20)}
    assert new == []
    assert gone == []


def test_monte_carlo_uncertainty_returns_interval():
    probability = np.full((8, 8, 8), 0.5, dtype=float)
    result = monte_carlo_volume_interval(probability, (1.0, 1.0, 1.0), n_samples=50)
    assert result["ci95_ml"][0] < result["mean_ml"] < result["ci95_ml"][1]
