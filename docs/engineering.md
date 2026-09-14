# AI-Photon engineering workflow

This document describes the minimum engineering contract for changes to the research code.

## Local setup

```bash
uv sync --extra dev
```

To run the Streamlit application as well:

```bash
uv sync --extra dev --extra app
```

## Quality checks

Run lint and tests before opening a pull request:

```bash
uv run ruff check mvp/ai_photon_mvp tests
uv run pytest --cov=ai_photon_mvp --cov-report=term-missing
```

The initial coverage floor is intentionally modest because the repository predates the test suite. It must increase as legacy modules receive tests; new production code should be accompanied by tests.

## Scientific regression benchmark

The compact deterministic benchmark exercises the quantitative spectral chain:

```text
VMI simulation -> material decomposition -> iodine concentration -> iodine load Q
```

Run it with:

```bash
uv run python -m ai_photon_mvp.benchmark
uv run pytest tests/regression/test_spectral_benchmark.py -q
```

Reference tolerances live in `benchmarks/reference_metrics.json`. A change that intentionally modifies the quantitative model must document the scientific rationale and update the reference only after the new result has been independently reviewed.

## CI policy

`.github/workflows/ci.yml` runs lint, unit/regression tests and coverage on Python 3.11 and 3.12.

`.github/workflows/benchmark.yml` runs the scientific spectral regression when the spectral model, benchmark definition or reference tolerances change.

## Next engineering milestones

1. Move the package from `mvp/ai_photon_mvp` to a `src/ai_photon` layout after behavior is covered by tests.
2. Introduce a `MedicalVolume` abstraction preserving spacing, origin and direction.
3. Replace bin-shrink preprocessing with explicit physical resampling.
4. Separate classical segmentation baselines from learned segmentation models.
5. Add Dice, IoU, HD95 and surface-distance evaluation against public reference masks.
6. Replace nearest-centroid longitudinal matching with a multi-feature assignment model.
