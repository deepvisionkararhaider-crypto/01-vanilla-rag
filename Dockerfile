# Vanilla RAG API image.
#
# By default this installs only the core dependencies, so the service runs with
# the offline hashing embedder + numpy store (small, fast, no model downloads).
# Build with real transformer embeddings + FAISS via:
#   docker build --build-arg INSTALL_ML=true -t vanilla-rag:ml .

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ARG INSTALL_ML=false
RUN if [ "$INSTALL_ML" = "true" ]; then \
        pip install --no-cache-dir -r requirements-ml.txt ; \
    fi

COPY app ./app
COPY scripts ./scripts

RUN useradd -m appuser \
    && mkdir -p /app/data /app/index \
    && chown -R appuser:appuser /app
USER appuser

ENV HOST=0.0.0.0 \
    PORT=8000 \
    DATA_DIR=/app/data \
    INDEX_DIR=/app/index

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
