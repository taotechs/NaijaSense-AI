"""FastAPI application setup."""

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

app = FastAPI(
    title="NaijaSense AI",
    description=(
        "DSN × BCT LLM Agent Challenge — Task A (user modeling) and Task B "
        "(recommendation) with dual submission endpoints."
    ),
    version="2.0.0",
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
