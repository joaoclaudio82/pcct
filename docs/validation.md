# Validation protocol

AI-Photon is research software. Validation must be task-specific and dataset-specific.

## Segmentation

For every learned or classical segmentation method, report:

- dataset and license;
- number of exams and split strategy;
- reference-mask provenance;
- Dice;
- IoU;
- precision and recall;
- HD95 in mm;
- average surface distance in mm;
- confidence intervals when sample size permits.

## Spectral quantification

Report concentration bias, MAE/RMSE, iodine-load error, dose condition, reconstruction condition and lesion size. The deterministic phantom regression in CI protects numerical regressions but does not replace external validation.

## Longitudinal analysis

Report registration error in physical units, lesion-matching accuracy, new/disappeared lesion accuracy and error in change of quantitative biomarkers.

## Learned models

Every checkpoint must be accompanied by model provenance, training dataset, validation dataset, preprocessing, software version, Git commit and performance table. External nnU-Net/MONAI weights must not be treated as clinically validated merely because inference runs successfully.

## Clinical boundary

No output from this repository is intended for diagnosis, prognosis or treatment decisions without appropriate clinical validation, ethics/regulatory review and applicable device-software controls.
