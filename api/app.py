"""FastAPI application setup."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.agent_routes import router as agent_router
from api.routes import router
from utils.config import settings
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="NaijaSense AI",
    description="Context-aware user simulation and recommendation service.",
    version="1.0.0",
)
origins = [
    "https://naija-sense-ai.vercel.app",  # Your production frontend
    "http://localhost:3000",             # For local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, etc.
    allow_headers=["*"],  # Allows all headers
)

app.include_router(router, prefix=settings.api_prefix)
app.include_router(agent_router, prefix="/api/agent")


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

