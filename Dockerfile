FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY mvp/ai_photon_mvp ./mvp/ai_photon_mvp
RUN pip install --no-cache-dir ".[api,app]"
COPY --chown=10001:10001 . .
RUN useradd --uid 10001 --create-home photon && chown -R photon:photon /app
USER photon
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
