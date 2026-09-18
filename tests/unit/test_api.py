import numpy as np
import pytest
import SimpleITK as sitk
from fastapi.testclient import TestClient

from api import main

client = TestClient(main.app)


def test_health():
    assert client.get("/health").json()["use"] == "research-only"


@pytest.mark.parametrize(
    "name,data,status",
    [
        ("exam.exe", b"bad", 415),
        ("exam.nii", b"", 422),
        ("exam.nii", b"invalid", 422),
        ("exam.mha", b"ElementDataFile = /etc/passwd\n", 422),
        ("exam.nrrd", b"NRRD0004\ndata file: /etc/passwd\n\n", 422),
    ],
)
def test_invalid_uploads(name, data, status):
    response = client.post("/v1/exams/inspect", files={"file": (name, data)})
    assert response.status_code == status
    assert "/etc/" not in response.text
    assert "Traceback" not in response.text


def test_upload_size_limit(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 4)
    response = client.post("/v1/exams/inspect", files={"file": ("exam.nii", b"12345")})
    assert response.status_code == 413


@pytest.mark.parametrize("suffix", [".nii.gz", ".nii", ".mha", ".nrrd"])
def test_real_image_upload(tmp_path, suffix):
    path = tmp_path / ("synthetic" + suffix)
    image = sitk.GetImageFromArray(np.zeros((8, 9, 10), dtype=np.float32))
    image.SetSpacing((1, 2, 3))
    sitk.WriteImage(image, str(path))
    response = client.post("/v1/exams/inspect", files={"file": (path.name, path.read_bytes())})
    assert response.status_code == 200, response.text
    assert response.json()["spacing_mm"] == [3, 2, 1]
    assert response.json()["shape"] == [8, 9, 10]


def test_decoded_voxel_limit(tmp_path, monkeypatch):
    path = tmp_path / "exam.nii.gz"
    sitk.WriteImage(sitk.Image([4, 4, 4], sitk.sitkFloat32), str(path))
    monkeypatch.setattr(main, "MAX_VOXELS", 63)
    response = client.post("/v1/exams/inspect", files={"file": (path.name, path.read_bytes())})
    assert response.status_code == 413


def test_temporary_directory_removed_on_reader_failure(monkeypatch):
    paths = []

    def fail(path):
        paths.append(path)
        raise RuntimeError("private path must not be exposed")

    monkeypatch.setattr(main, "_inspect_path", fail)
    response = client.post("/v1/exams/inspect", files={"file": ("exam.nii", b"bytes")})
    assert response.status_code == 422
    assert "private path" not in response.text
    assert paths and not paths[0].parent.exists()
