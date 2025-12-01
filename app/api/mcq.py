"""MCQ API router.

This module provides the MCQ generation endpoint.

Requirements: 3.1, 3.3, 3.4
"""
from fastapi import APIRouter

from app.config import get_settings
from app.models.mcq_schemas import MCQRequest, MCQResponse
from app.services.mcq_service import (
    MCQService,
    AuthorizationError as MCQAuthError,
    DocumentNotFoundError as MCQDocNotFoundError,
)
from app.chains.mcq_chain import MCQGenerationError
from app.api.exceptions import AuthorizationError, NotFoundError, ProcessingError

router = APIRouter(prefix="/api/mcq", tags=["mcq"])


@router.post("/generate", response_model=MCQResponse)
def generate_mcqs(request: MCQRequest) -> MCQResponse:
    """Generate MCQs from a user's document.
    
    Accepts user_id, document_id, num_questions, and difficulty parameters.
    Returns an MCQResponse with generated questions.
    
    Requirements: 3.1, 3.3, 3.4
    """
    settings = get_settings()
    mcq_service = MCQService(settings)
    
    try:
        return mcq_service.generate_mcqs(
            user_id=request.user_id,
            document_id=request.document_id,
            num_questions=request.num_questions,
            difficulty=request.difficulty
        )
    except MCQAuthError as e:
        raise AuthorizationError(str(e))
    except MCQDocNotFoundError as e:
        raise NotFoundError(str(e))
    except MCQGenerationError as e:
        raise ProcessingError(str(e))
