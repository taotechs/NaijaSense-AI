FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '8000')}/api/v1/health\", timeout=3)"

# Shell form so $PORT is expanded. Local docker-compose leaves PORT unset
# (defaults to 8000); hosts like Render/Fly inject PORT and the server
# rebinds automatically. No code change required to redeploy elsewhere.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
