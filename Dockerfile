FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
# Entrypoint runs ensure_large_corpus; skip duplicate work in FastAPI lifespan.
ENV CORPUS_BUILD_ON_STARTUP=false

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake 10k-row evaluation corpus + inverted index into the image (seed: review_corpus.jsonl).
RUN python scripts/ensure_large_corpus.py

RUN chmod +x scripts/docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '8000')}/api/v1/health\", timeout=3)"

# Entrypoint re-checks corpus on cold start (e.g. ephemeral disk wiped) then serves API.
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
