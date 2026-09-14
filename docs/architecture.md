# AI-Photon V2 architecture

The V2 evolves the research MVP without breaking the original entry points.

## Core flow

```text
DICOM/NIfTI/NRRD
      |
      v
MedicalVolume (data + spacing + origin + direction + metadata)
      |
      v
Physical resampling + QA + body-region detection
      |
      +--> classical segmentation baseline
      +--> optional MONAI/TorchScript segmentation
      |
      v
Segmentation validation (Dice, IoU, precision, recall, HD95, ASD)
      |
      v
Radiomic + spectral biomarkers
      |
      v
Rigid/deformable registration
      |
      v
Hungarian multi-feature lesion matching
      |
      v
Longitudinal response + uncertainty
```

## Engineering principles

- Physical geometry must be preserved throughout the pipeline.
- Classical methods remain explicit baselines rather than being silently replaced.
- Heavy ML dependencies are optional.
- Quantitative spectral regressions are protected by deterministic benchmarks.
- Model weights are never bundled implicitly; each model must have provenance and validation metadata.
- The FastAPI layer is a research interface only and carries no clinical claim.

## Optional subsystems

- `radiomics`: PyRadiomics feature extraction.
- `ml`: PyTorch/MONAI inference adapter.
- `tracking`: MLflow experiment tracking.
- `api`: FastAPI/uvicorn service.

## Validation target

Every learned segmentation should be compared against the classical baseline and report at minimum Dice, IoU, HD95 and average surface distance on a held-out dataset with documented reference masks.
