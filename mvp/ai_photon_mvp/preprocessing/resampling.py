"""Physical-space resampling utilities for medical volumes."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from ai_photon_mvp.io.image import MedicalVolume


def resample_volume(
    volume: MedicalVolume,
    target_spacing_mm: float | tuple[float, float, float] = 1.0,
    is_label: bool = False,
) -> MedicalVolume:
    image = volume.to_sitk()
    if isinstance(target_spacing_mm, (int, float)):
        target_zyx = (float(target_spacing_mm),) * 3
    else:
        target_zyx = tuple(float(x) for x in target_spacing_mm)
    target_xyz = tuple(reversed(target_zyx))

    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    new_size = [
        max(1, int(round(sz * sp / tsp)))
        for sz, sp, tsp in zip(old_size, old_spacing, target_xyz)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_xyz)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    out = resampler.Execute(image)
    return MedicalVolume.from_sitk(out, source=volume.source)


def voxel_volume_ml(spacing_zyx: tuple[float, float, float]) -> float:
    return float(np.prod(spacing_zyx) / 1000.0)
