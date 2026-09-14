"""Deformable registration using a B-spline transform."""
from __future__ import annotations

import SimpleITK as sitk

from ai_photon_mvp.io.image import MedicalVolume


def register_bspline(
    fixed: MedicalVolume,
    moving: MedicalVolume,
    mesh_size: tuple[int, int, int] = (6, 6, 6),
) -> tuple[MedicalVolume, sitk.Transform]:
    fixed_img = fixed.to_sitk()
    moving_img = moving.to_sitk()
    transform = sitk.BSplineTransformInitializer(fixed_img, list(mesh_size), order=3)

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=40)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.10, seed=2026)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=80,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=500,
        costFunctionConvergenceFactor=1e7,
    )
    method.SetInitialTransform(transform, inPlace=False)
    final_transform = method.Execute(fixed_img, moving_img)
    resampled = sitk.Resample(
        moving_img, fixed_img, final_transform, sitk.sitkLinear, -1024.0, moving_img.GetPixelID()
    )
    return MedicalVolume.from_sitk(resampled, source=moving.source), final_transform
