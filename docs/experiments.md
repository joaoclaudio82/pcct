# Experiment tracking

Use MLflow for experiments whose results may enter a report, article or model comparison.

Record at minimum:

- Git commit;
- dataset/version;
- preprocessing configuration;
- model/checkpoint identifier;
- random seed;
- hyperparameters;
- segmentation metrics;
- spectral metrics;
- output figures and tables.

The helper `ai_photon_mvp.tracking.mlflow_utils.log_experiment` provides the minimal integration. For reproducible studies, store the exact configuration YAML used by the run as an artifact.
