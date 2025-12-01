"""Chat Service module for document-scoped conversations.

This module provides the ChatService class that handles user chat
requests, validates document ownership, and invokes the RAG chain.

Requirements: 1.1, 1.2, 1.3, 1.4
"""
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from supabase import Client, create_client

from app.config import Settings
from app.chains.rag_chain import RAGChain, RAGResponse


@dataclass
class SourceReference:
    """Represents a source reference in a chat response."""
    content: str
    metadata: dict[str, Any]


@dataclass
class ChatResponse:
    """Response from the chat service."""
    answer: str
    sources: list[SourceReference]
    document_id: UUID
    has_context: bool


class AuthorizationError(Exception):
    """Raised when user is not authorized to access a document."""
    pass


class DocumentNotFoundError(Exception):
    """Raised when document is not found."""
    pass


class ChatService:
    """Handles document-scoped chat conversations.
    
    Validates user ownership of documents and invokes the RAG chain
    to generate context-aware responses.
    
    Requirements:
    - 1.1: Retrieve relevant chunks and generate contextual response
    - 1.2: Include source references in response
    - 1.3: Only search within embeddings belonging to specific document
    - 1.4: Reject requests for documents user doesn't own
    """
    
    USER_DOCS_TABLE = "user_docs"
    
    def __init__(self, settings: Settings) -> None:
        """Initialize the chat service.
        
        Args:
            settings: Application settings
        """
        self._settings = settings
        self._supabase: Client = create_client(
            settings.supabase_url,
            settings.supabase_key
        )
        self._rag_chain = RAGChain(settings)

    def validate_document_ownership(
        self,
        document_id: UUID,
        user_id: UUID
    ) -> dict[str, Any]:
        """Validate that a user owns a document.
        
        Args:
            document_id: UUID of the document
            user_id: UUID of the user
            
        Returns:
            Document record if validation passes
            
        Raises:
            DocumentNotFoundError: If document doesn't exist
            AuthorizationError: If user doesn't own the document
            
        Requirements: 1.4
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
        
        return document
    
    def _format_sources(
        self,
        rag_sources: list[dict[str, Any]]
    ) -> list[SourceReference]:
        """Format RAG sources into SourceReference objects.
        
        Args:
            rag_sources: Source dictionaries from RAG response
            
        Returns:
            List of SourceReference objects
            
        Requirements: 1.2
        """
        return [
            SourceReference(
                content=source["content"],
                metadata=source["metadata"]
            )
            for source in rag_sources
        ]
    
    def chat(
        self,
        user_id: UUID,
        document_id: UUID,
        message: str
    ) -> ChatResponse:
        """Process a chat message for a document.
        
        Validates user ownership, retrieves relevant context from the
        document, and generates a response using the RAG chain.
        
        Args:
            user_id: UUID of the user sending the message
            document_id: UUID of the document to query
            message: The user's question/message
            
        Returns:
            ChatResponse with answer and source references
            
        Raises:
            DocumentNotFoundError: If document doesn't exist
            AuthorizationError: If user doesn't own the document
            
        Requirements: 1.1, 1.2, 1.3, 1.4
        """
        # Validate document ownership (Requirement 1.4)
        self.validate_document_ownership(document_id, user_id)
        
        # Invoke RAG chain with document-scoped retrieval (Requirements 1.1, 1.3)
        rag_response: RAGResponse = self._rag_chain.invoke(
            question=message,
            document_id=document_id
        )
        
        # Format response with sources (Requirement 1.2)
        sources = self._format_sources(rag_response.sources)
        
        return ChatResponse(
            answer=rag_response.answer,
            sources=sources,
            document_id=document_id,
            has_context=rag_response.has_context
        )
