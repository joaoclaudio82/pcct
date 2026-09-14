"""FastAPI service exposing the research pipeline.

Research use only. This API is not a medical device and must not support
clinical diagnosis or treatment decisions.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from ai_photon_mvp.io.image import load_medical_volume
from ai_photon_mvp.pipeline import qa_check
from ai_photon_mvp.preprocessing.body_region import detect_body_region

app = FastAPI(title="AI-Photon API", version="0.2.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ai-photon", "use": "research-only"}


@app.post("/v1/exams/inspect")
async def inspect_exam(file: UploadFile = File(...)) -> dict:
    suffix = "".join(Path(file.filename or "exam.bin").suffixes) or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        volume = load_medical_volume(tmp_path)
        qa = qa_check(volume.data, volume.spacing)
        return {
            "body_region": detect_body_region(volume),
            "shape": list(volume.data.shape),
            "spacing_mm": list(volume.spacing),
            "qa": json.loads(json.dumps(qa, default=str)),
            "warning": "research prototype; not for clinical decision making",
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
