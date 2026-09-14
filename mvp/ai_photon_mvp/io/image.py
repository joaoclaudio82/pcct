"""Medical-image I/O preserving physical geometry and lightweight metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(slots=True)
class MedicalVolume:
    data: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    metadata: dict[str, str] = field(default_factory=dict)
    source: str | None = None

    @classmethod
    def from_sitk(cls, image: sitk.Image, source: str | None = None) -> "MedicalVolume":
        md = {k: image.GetMetaData(k) for k in image.GetMetaDataKeys()}
        return cls(
            data=sitk.GetArrayFromImage(image).astype(np.float32),
            spacing=tuple(float(x) for x in image.GetSpacing()[::-1]),
            origin=tuple(float(x) for x in image.GetOrigin()[::-1]),
            direction=tuple(float(x) for x in image.GetDirection()),
            metadata=md,
            source=source,
        )

    def to_sitk(self) -> sitk.Image:
        image = sitk.GetImageFromArray(self.data)
        image.SetSpacing(tuple(reversed(self.spacing)))
        image.SetOrigin(tuple(reversed(self.origin)))
        image.SetDirection(self.direction)
        for key, value in self.metadata.items():
            try:
                image.SetMetaData(str(key), str(value))
            except RuntimeError:
                pass
        return image


def _read_dicom_folder(path: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(path))
    if not series_ids:
        raise ValueError(f"nenhuma série DICOM encontrada em {path}")
    # MVP policy: choose the series with most instances instead of first arbitrary UID.
    candidates = []
    for uid in series_ids:
        files = reader.GetGDCMSeriesFileNames(str(path), uid)
        candidates.append((len(files), uid, files))
    _, _, files = max(candidates, key=lambda item: item[0])
    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOff()
    return reader.Execute()


def load_medical_volume(path: str | Path) -> MedicalVolume:
    p = Path(path)
    image = _read_dicom_folder(p) if p.is_dir() else sitk.ReadImage(str(p))
    volume = MedicalVolume.from_sitk(image, source=str(p))
    if p.is_dir():
        # Capture common DICOM descriptors from the first slice when available.
        try:
            first = sitk.ReadImage(str(_read_first_dicom_file(p)))
            for tag, name in {
                "0008|0060": "Modality",
                "0018|0015": "BodyPartExamined",
                "0008|1030": "StudyDescription",
                "0008|103e": "SeriesDescription",
            }.items():
                if first.HasMetaDataKey(tag):
                    volume.metadata[name] = first.GetMetaData(tag).strip()
        except Exception:
            pass
    return volume


def _read_first_dicom_file(path: Path) -> Path:
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(path))
    if not ids:
        raise ValueError("nenhuma série DICOM")
    files = reader.GetGDCMSeriesFileNames(str(path), ids[0])
    return Path(files[0])
