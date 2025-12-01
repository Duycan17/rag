"""API Layer - REST endpoints for the RAG application."""

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.exceptions import (
    AuthorizationError,
    NotFoundError,
    ProcessingError,
    ValidationError,
)

__all__ = [
    "chat_router",
    "documents_router",
    "AuthorizationError",
    "NotFoundError",
    "ProcessingError",
    "ValidationError",
]
