"""MCQ Service module for generating multiple choice questions.

This module provides the MCQService class that handles MCQ generation
requests, validates document ownership, and orchestrates the MCQ chain.

Requirements: 1.4, 2.4, 5.2
"""
from uuid import UUID

from supabase import Client, create_client

from app.config import Settings
from app.chains.mcq_chain import MCQChain, MCQGenerationResult, MCQGenerationError
from app.models.mcq_schemas import DifficultyLevel, MCQResponse


class AuthorizationError(Exception):
    """Raised when user is not authorized to access a document."""
    pass


class DocumentNotFoundError(Exception):
    """Raised when document is not found."""
    pass


class MCQService:
    """Handles MCQ generation requests.
    
    Validates user ownership of documents and orchestrates the MCQ chain
    to generate practice questions.
    
    Requirements:
    - 1.4: Reject requests for documents user doesn't own
    - 2.4: Default to medium difficulty when not specified
    - 5.2: Use same authorization logic as chat feature
    """
    
    USER_DOCS_TABLE = "user_docs"
    
    def __init__(self, settings: Settings) -> None:
        """Initialize the MCQ service.
        
        Args:
            settings: Application settings
        """
        self._settings = settings
        self._supabase: Client = create_client(
            settings.supabase_url,
            settings.supabase_key
        )
        self._mcq_chain = MCQChain(settings)

    def validate_document_ownership(
        self,
        document_id: UUID,
        user_id: UUID
    ) -> None:
        """Validate that a user owns a document.
        
        Reuses the same authorization logic as ChatService.
        
        Args:
            document_id: UUID of the document
            user_id: UUID of the user
            
        Raises:
            DocumentNotFoundError: If document doesn't exist
            AuthorizationError: If user doesn't own the document
            
        Requirements: 1.4, 5.2
        """
        # First check if document exists
        result = self._supabase.table(self.USER_DOCS_TABLE).select("*").eq(
            "id", str(document_id)
        ).execute()
        
        if not result.data:
            raise DocumentNotFoundError(
                f"Document {document_id} not found"
            )
        
        document = result.data[0]
        
        # Check ownership
        if document.get("user_id") != str(user_id):
            raise AuthorizationError(
                f"User {user_id} is not authorized to access document {document_id}"
            )

    def generate_mcqs(
        self,
        user_id: UUID,
        document_id: UUID,
        num_questions: int = 5,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    ) -> MCQResponse:
        """Generate MCQs from a user's document.
        
        Validates document ownership and invokes the MCQ chain to
        generate practice questions.
        
        Args:
            user_id: UUID of the user requesting MCQs
            document_id: UUID of the document to generate MCQs from
            num_questions: Number of questions to generate (default 5)
            difficulty: Difficulty level (default medium)
            
        Returns:
            MCQResponse with generated questions
            
        Raises:
            DocumentNotFoundError: If document doesn't exist
            AuthorizationError: If user doesn't own the document
            MCQGenerationError: If MCQ generation fails
            
        Requirements: 1.4, 2.4, 5.2
        """
        # Validate document ownership (Requirements: 1.4, 5.2)
        self.validate_document_ownership(document_id, user_id)
        
        # Invoke MCQ chain
        result: MCQGenerationResult = self._mcq_chain.invoke(
            document_id=document_id,
            num_questions=num_questions,
            difficulty=difficulty
        )
        
        # Format response with generated questions
        return MCQResponse(
            questions=result.questions,
            document_id=document_id,
            difficulty=difficulty.value,
            generated_count=len(result.questions)
        )
