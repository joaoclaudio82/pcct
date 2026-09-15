.PHONY: sync test lint benchmark api app

sync:
	uv sync --extra dev

test:
	uv run pytest --cov=ai_photon_mvp --cov-report=term-missing

lint:
	uv run ruff check mvp/ai_photon_mvp tests

benchmark:
	uv run python -m ai_photon_mvp.benchmark
	uv run pytest tests/regression/test_spectral_benchmark.py -q

api:
	uv run --extra api uvicorn api.main:app --reload

app:
	uv run --extra app streamlit run mvp/app.py
