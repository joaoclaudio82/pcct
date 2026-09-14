"""Rigid registration in physical space using SimpleITK."""
from __future__ import annotations

import SimpleITK as sitk

from ai_photon_mvp.io.image import MedicalVolume


def register_rigid(fixed: MedicalVolume, moving: MedicalVolume) -> tuple[MedicalVolume, sitk.Transform]:
    fixed_img = fixed.to_sitk()
    moving_img = moving.to_sitk()

    initial = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.15, seed=2026)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=150,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetInitialTransform(initial, inPlace=False)
    transform = method.Execute(fixed_img, moving_img)
    resampled = sitk.Resample(
        moving_img,
        fixed_img,
        transform,
        sitk.sitkLinear,
        -1024.0,
        moving_img.GetPixelID(),
    )
    return MedicalVolume.from_sitk(resampled, source=moving.source), transform
