"""Local research API. Put authentication and request limits at the deployment boundary."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

import SimpleITK as sitk
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from ai_photon_mvp.io.image import load_medical_volume
from ai_photon_mvp.pipeline import qa_check
from ai_photon_mvp.preprocessing.body_region import detect_body_region
from ai_photon_mvp.version import __version__

MAX_UPLOAD_BYTES = 128 * 1024 * 1024
MAX_VOXELS = 64_000_000
SUPPORTED_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".mha")
app = FastAPI(title="AI-Photon API", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ai-photon", "version": __version__, "use": "research-only"}


def _inspect_path(path: Path) -> dict:
    # Uploaded headers must not instruct the reader to open other server files.
    if path.suffix in {".nrrd", ".mha"}:
        with path.open("rb") as stream:
            header = stream.read(65536)
        terminated = False
        for raw_line in header.splitlines():
            line = raw_line.strip().lower()
            if path.suffix == ".nrrd":
                if not line:
                    terminated = True
                    break
                if line.startswith((b"data file:", b"datafile:")):
                    raise ValueError("detached image data is not supported")
            elif line.startswith(b"elementdatafile"):
                if line.partition(b"=")[2].strip() != b"local":
                    raise ValueError("detached image data is not supported")
                terminated = True
                break
        if not terminated:
            raise ValueError("invalid or oversized image header")
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    if reader.GetDimension() != 3 or reader.GetNumberOfComponents() != 1:
        raise ValueError("expected scalar 3D image")
    voxels = 1
    for size in reader.GetSize():
        voxels *= size
    if voxels > MAX_VOXELS:
        raise HTTPException(status_code=413, detail="Image exceeds the voxel limit")
    volume = load_medical_volume(path)
    qa = qa_check(volume.data, volume.spacing)
    return {
        "body_region": detect_body_region(volume),
        "shape": list(volume.data.shape),
        "spacing_mm": list(volume.spacing),
        "qa": qa,
        "warning": "research prototype; not for clinical decision making",
    }


@app.post("/v1/exams/inspect")
async def inspect_exam(file: Annotated[UploadFile, File()]) -> dict:
    try:
        name = (file.filename or "").lower()
        suffix = next((s for s in SUPPORTED_SUFFIXES if name.endswith(s)), None)
        if suffix is None:
            raise HTTPException(status_code=415, detail="Use .nii, .nii.gz, .nrrd or .mha")
        with tempfile.TemporaryDirectory(prefix="ai-photon-") as directory:
            path = Path(directory) / ("exam" + suffix)
            size = 0
            with path.open("wb") as target:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Upload exceeds 128 MiB limit")
                    target.write(chunk)
            if not size:
                raise HTTPException(status_code=422, detail="Empty image upload")
            try:
                return await run_in_threadpool(_inspect_path, path)
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(status_code=422, detail="Invalid or unsupported medical image") from exc
    finally:
        await file.close()
