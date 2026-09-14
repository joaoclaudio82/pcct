from pathlib import Path

from ai_photon_mvp.benchmark import check_against_reference, run_spectral_benchmark


def test_spectral_quantification_regression():
    metrics = run_spectral_benchmark()
    reference = Path(__file__).resolve().parents[2] / "benchmarks" / "reference_metrics.json"
    failures = check_against_reference(metrics, reference)

    assert failures == [], "\n".join(failures)
