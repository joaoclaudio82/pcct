import numpy as np
import pytest
import SimpleITK as sitk
from ai_photon_mvp.io.image import MedicalVolume

@pytest.mark.parametrize("image", [sitk.Image([4, 4], sitk.sitkFloat32), sitk.Image([4, 4, 4], sitk.sitkVectorFloat32, 3)])
def test_reject_non_scalar_3d(image):
    with pytest.raises(ValueError):
        MedicalVolume.from_sitk(image)

def test_oblique_geometry_roundtrip():
    image = sitk.GetImageFromArray(np.zeros((3, 4, 5), dtype=np.float32))
    image.SetDirection((0, -1, 0, 1, 0, 0, 0, 0, 1))
    image.SetOrigin((5, 7, 9)); image.SetSpacing((1, 2, 3))
    recovered = MedicalVolume.from_sitk(image).to_sitk()
    assert recovered.TransformIndexToPhysicalPoint((1, 2, 1)) == image.TransformIndexToPhysicalPoint((1, 2, 1))
