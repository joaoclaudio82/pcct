"""Optional MLflow experiment tracking helpers."""
from __future__ import annotations


def log_experiment(
    run_name: str,
    params: dict[str, object],
    metrics: dict[str, float],
    artifacts: list[str] | None = None,
    experiment_name: str = "ai-photon",
) -> str:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("instale o extra 'tracking' para usar MLflow") from exc

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(metrics)
        for artifact in artifacts or []:
            mlflow.log_artifact(artifact)
        return run.info.run_id
