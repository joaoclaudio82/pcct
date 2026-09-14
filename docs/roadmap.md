# AI-Photon roadmap

## Completed infrastructure

- reproducible packaging and CI;
- deterministic spectral benchmark;
- medical-volume geometry abstraction;
- metadata-first body-region detection;
- isotropic physical resampling;
- segmentation metrics;
- explicit classical baselines;
- MONAI/TorchScript and nnU-Net adapter contracts;
- radiomics and spectral lesion biomarkers;
- rigid and deformable registration;
- global Hungarian lesion matching;
- uncertainty utilities;
- FastAPI, Docker and MLflow integration.

## Data/model milestones

1. Select public reference datasets for lung, liver/lesion and brain tasks.
2. Establish classical baseline tables with Dice/IoU/HD95/ASD.
3. Add validated pretrained MONAI/nnU-Net checkpoints with provenance files.
4. Run external spectral phantom experiments when PCCT spectral datasets are available.
5. Validate longitudinal matching on real repeated examinations.
6. Calibrate uncertainty against observed segmentation/quantification errors.
7. Freeze a research release and archive experiment metadata/model cards.

## Translation milestones

Any step toward clinical deployment requires a separate quality-management, cybersecurity, data-governance, ethics and regulatory workstream. The research repository itself does not constitute a clinical product.
