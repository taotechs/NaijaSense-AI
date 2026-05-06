"""FastAPI application setup."""

from fastapi import FastAPI

from api.routes import router
from utils.config import settings

app = FastAPI(
    title="NaijaSense AI",
    description="Context-aware user simulation and recommendation service.",
    version="1.0.0",
)

app.include_router(router, prefix=settings.api_prefix)

