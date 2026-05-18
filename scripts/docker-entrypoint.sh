#!/bin/sh
set -e

# Fast no-op when image was built with corpus; rebuilds only if files are missing.
python scripts/ensure_large_corpus.py

exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
