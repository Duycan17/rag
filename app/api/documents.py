"""Documents API router.

This module provides the document processing endpoint.

Requirements: 5.1, 5.4, 5.5
"""
from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import ProcessRequest, ProcessResponse
from app.services.document_service import (
    DocumentService,
    DocumentNotFoundError as DocNotFoundError,
    DocumentProcessingError,
)
from app.api.exceptions import NotFoundError, ProcessingError

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/process", response_model=ProcessResponse)
def process_document(request: ProcessRequest) -> ProcessResponse:
    """Process a document for chat.
    
    Accepts document_id and user_id parameters.
    Returns a ProcessResponse with processing statistics.
    
    Requirements: 5.1, 5.4
    """
    settings = get_settings()
    
    with DocumentService(settings) as doc_service:
        try:
            result = doc_service.process_document(
                document_id=request.document_id,
                user_id=request.user_id
            )
            
            return ProcessResponse(
                status=result.status.value,
                chunks_created=result.chunks_created,
                document_id=result.document_id
            )
        except DocNotFoundError as e:
            raise NotFoundError(str(e))
        except DocumentProcessingError as e:
            raise ProcessingError(str(e))
