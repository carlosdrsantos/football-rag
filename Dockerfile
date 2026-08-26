# The models are baked in rather than fetched on boot: 420 MB downloaded during
# a deploy is a slow first request and an outage when the hub is unreachable.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e . \
 && python -c "from lotg.retrieval import embedder, reranker; embedder.dimensions(); reranker.model()"

# Built by `make fetch parse chunk`; the BM25 index is rebuilt from it at startup.
COPY data/processed/chunks.jsonl ./data/processed/chunks.jsonl

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "lotg.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
