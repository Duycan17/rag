"""Chat API router.

This module provides the chat endpoint for document-scoped conversations.

Requirements: 3.1, 3.3
"""
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.models.schemas import ChatRequest, ChatResponse, SourceReference
from app.services.chat_service import (
    ChatService,
    AuthorizationError as ChatAuthError,
    DocumentNotFoundError as ChatDocNotFoundError,
)
from app.api.exceptions import AuthorizationError, NotFoundError

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message for a document.
    
    Accepts user_id, document_id, and message parameters.
    Returns a ChatResponse with answer and sources.
    
    Requirements: 3.1, 3.3
    """
    settings = get_settings()
    chat_service = ChatService(settings)
    
    try:
        result = chat_service.chat(
            user_id=request.user_id,
            document_id=request.document_id,
            message=request.message
        )
        
        # Convert service response to API response
        sources = [
            SourceReference(
                content=src.content,
                metadata=src.metadata
            )
            for src in result.sources
        ]
        
        return ChatResponse(
            answer=result.answer,
            sources=sources
        )
    except ChatAuthError as e:
        raise AuthorizationError(str(e))
    except ChatDocNotFoundError as e:
        raise NotFoundError(str(e))
