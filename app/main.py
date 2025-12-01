"""Main FastAPI application.

This module sets up the FastAPI application with routers,
CORS middleware, and error handling.

Requirements: 3.1, 3.4, 5.5
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.exceptions import (
    AuthorizationError,
    NotFoundError,
    ProcessingError,
    ValidationError,
)
from app.models.schemas import ErrorResponse

app = FastAPI(
    title="Document Chat RAG API",
    description="RAG application for chatting with uploaded documents",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Error handlers - Requirements: 3.4, 5.5
@app.exception_handler(ValidationError)
async def validation_error_handler(
    request: Request,
    exc: ValidationError
) -> JSONResponse:
    """Handle validation errors with 422 status."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="ValidationError",
            detail=str(exc),
            code="VALIDATION_ERROR"
        ).model_dump()
    )


@app.exception_handler(PydanticValidationError)
async def pydantic_validation_error_handler(
    request: Request,
    exc: PydanticValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors with 422 status."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="ValidationError",
            detail=str(exc),
            code="VALIDATION_ERROR"
        ).model_dump()
    )


@app.exception_handler(AuthorizationError)
async def authorization_error_handler(
    request: Request,
    exc: AuthorizationError
) -> JSONResponse:
    """Handle authorization errors with 403 status."""
    return JSONResponse(
        status_code=403,
        content=ErrorResponse(
            error="AuthorizationError",
            detail=str(exc),
            code="AUTHORIZATION_ERROR"
        ).model_dump()
    )


@app.exception_handler(NotFoundError)
async def not_found_error_handler(
    request: Request,
    exc: NotFoundError
) -> JSONResponse:
    """Handle not found errors with 404 status."""
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="NotFoundError",
            detail=str(exc),
            code="NOT_FOUND"
        ).model_dump()
    )


@app.exception_handler(ProcessingError)
async def processing_error_handler(
    request: Request,
    exc: ProcessingError
) -> JSONResponse:
    """Handle processing errors with 500 status."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="ProcessingError",
            detail=str(exc),
            code="PROCESSING_ERROR"
        ).model_dump()
    )


# Include routers
app.include_router(chat_router)
app.include_router(documents_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
