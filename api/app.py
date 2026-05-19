"""FastAPI application setup."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.agent_routes import router as agent_router
from api.feedback_routes import router as feedback_router
from api.landing import router as landing_router
from api.routes import router
from api.task_routes import router as task_router
from utils.config import settings
from utils.logger import get_logger

_startup_logger = get_logger("startup")


def _ensure_corpus_sync() -> None:
    """Non-Docker runs: build corpus on first boot if files are missing."""
    if not settings.corpus_build_on_startup:
        return
    try:
        from scripts.ensure_large_corpus import ensure_large_corpus

        ensure_large_corpus()
    except Exception as exc:  # pragma: no cover
        _startup_logger.warning("Corpus ensure skipped or failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app  # unused
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _ensure_corpus_sync)
    yield


app = FastAPI(
    title="NaijaSense AI",
    description=(
        "DSN × BCT LLM Agent Challenge - Task A (user modeling) and Task B "
        "(recommendation) with dual submission endpoints."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(landing_router)
app.include_router(task_router)
app.include_router(router, prefix=settings.api_prefix)
app.include_router(agent_router, prefix="/api/agent")
app.include_router(feedback_router, prefix="/api/agent")


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": "Request body failed validation.",
            "issues": exc.errors(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {"error": "http_error", "detail": str(exc.detail)}
    return JSONResponse(
        status_code=exc.status_code,
        content={**detail, "path": str(request.url.path)},
    )
