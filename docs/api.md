# Research API

Install with:

```bash
uv sync --extra api
```

Run:

```bash
uv run uvicorn api.main:app --reload
```

Endpoints:

- `GET /health`: service health and research-use status.
- `POST /v1/exams/inspect`: upload a supported image file and return geometry, QA and detected body region.

The API is a research interface. Authentication, rate limiting, persistent job orchestration and clinical-data governance are deployment responsibilities and are not implied by this prototype.
