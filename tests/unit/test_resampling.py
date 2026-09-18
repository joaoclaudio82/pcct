import numpy as np
import pytest
import SimpleITK as sitk
from ai_photon_mvp.io.image import MedicalVolume
from ai_photon_mvp.preprocessing.resampling import resample_volume


def test_resampling_preserves_metadata_and_label_ids():
    image = sitk.GetImageFromArray(np.full((3, 3, 3), 7, dtype=np.int16))
    volume = MedicalVolume.from_sitk(image, source="synthetic")
    volume.metadata["BodyPartExamined"] = "HEAD"
    result = resample_volume(volume, 0.5, is_label=True)
    assert set(np.unique(result.data)) <= {0, 7}
    assert result.metadata == volume.metadata
    assert result.source == "synthetic"
    assert result.spacing == (0.5, 0.5, 0.5)


def test_resampling_invalid_spacing():
    volume = MedicalVolume.from_sitk(sitk.Image([3, 3, 3], sitk.sitkFloat32))
    with pytest.raises(ValueError):
        resample_volume(volume, 0)
