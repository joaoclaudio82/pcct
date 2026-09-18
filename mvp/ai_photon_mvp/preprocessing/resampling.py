"""Physical-space resampling utilities for medical volumes."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from ai_photon_mvp.io.image import MedicalVolume
from ai_photon_mvp.validation import spacing_zyx as _validated_spacing


def resample_volume(
    volume: MedicalVolume,
    target_spacing_mm: float | tuple[float, float, float] = 1.0,
    is_label: bool = False,
    default_value: float | None = None,
) -> MedicalVolume:
    image = volume.to_sitk()
    target_zyx = _validated_spacing(target_spacing_mm)
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
    if default_value is None:
        default_value = 0.0 if is_label else -1024.0
    if not np.isfinite(default_value):
        raise ValueError("default_value must be finite")
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    out = resampler.Execute(image)
    result = MedicalVolume.from_sitk(out, source=volume.source)
    result.metadata = dict(volume.metadata)
    return result


def voxel_volume_ml(spacing_zyx: tuple[float, float, float]) -> float:
    return float(np.prod(_validated_spacing(spacing_zyx)) / 1000.0)
