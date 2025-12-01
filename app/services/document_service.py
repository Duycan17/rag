"""Document Service module for integrated document processing.

This module provides the main DocumentService class that orchestrates
document downloading, text extraction, chunking, embedding generation,
and vector store storage.

Requirements: 2.4, 2.5
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import UUID

from supabase import Client, create_client

from app.config import Settings
from app.db.vector_store import SupabaseVectorStore
from app.services.document_processor import (
    DocumentProcessor,
    DocumentDownloadError,
    TextExtractionError,
)
from app.services.embedding_service import EmbeddingService
from app.services.text_chunker import TextChunker


class DocumentStatus(Enum):
    """Document processing status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass
class ProcessingResult:
    """Result of document processing."""
    document_id: UUID
    status: DocumentStatus
    chunks_created: int
    error_message: str | None = None


class DocumentNotFoundError(Exception):
    """Raised when document is not found."""
    pass


class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass


class DocumentService:
    """Orchestrates document processing pipeline.
    
    Integrates document downloading, text extraction, chunking,
    embedding generation, and vector store storage.
    
    Requirements:
    - 2.4: Store embeddings with document_id and user_id metadata
    - 2.5: Update document status after processing
    """
    
    USER_DOCS_TABLE = "user_docs"
    
    def __init__(self, settings: Settings) -> None:
        """Initialize the document service.
        
        Args:
            settings: Application settings
        """
        self._settings = settings
        self._supabase: Client = create_client(
            settings.supabase_url,
            settings.supabase_key
        )
        self._document_processor = DocumentProcessor(settings)
        self._text_chunker = TextChunker(settings)
        self._embedding_service = EmbeddingService(settings)
        self._vector_store = SupabaseVectorStore(settings)
    
    def get_document(self, document_id: UUID, user_id: UUID) -> dict[str, Any]:
        """Get document metadata from user_docs table.
        
        Args:
            document_id: UUID of the document
            user_id: UUID of the document owner
            
        Returns:
            Document record as dictionary
            
        Raises:
            DocumentNotFoundError: If document not found or not owned by user
        """
        result = self._supabase.table(self.USER_DOCS_TABLE).select("*").eq(
            "id", str(document_id)
        ).eq("user_id", str(user_id)).execute()
        
        if not result.data:
            raise DocumentNotFoundError(
                f"Document {document_id} not found for user {user_id}"
            )
        
        return result.data[0]
    
    def update_document_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        error_message: str | None = None
    ) -> None:
        """Update document status in user_docs table.
        
        Args:
            document_id: UUID of the document
            status: New status value
            error_message: Optional error message for failed status
            
        Requirements: 2.5
        """
        update_data: dict[str, Any] = {"status": status.value}
        if error_message:
            update_data["error_message"] = error_message
        
        self._supabase.table(self.USER_DOCS_TABLE).update(
            update_data
        ).eq("id", str(document_id)).execute()
    
    def process_document(
        self,
        document_id: UUID,
        user_id: UUID
    ) -> ProcessingResult:
        """Process a document for chat.
        
        Downloads the document, extracts text, chunks it, generates
        embeddings, and stores them in the vector store.
        
        Args:
            document_id: UUID of the document to process
            user_id: UUID of the document owner
            
        Returns:
            ProcessingResult with status and statistics
            
        Raises:
            DocumentNotFoundError: If document not found
            DocumentProcessingError: If processing fails
            
        Requirements: 2.4, 2.5
        """
        # Get document metadata
        try:
            document = self.get_document(document_id, user_id)
        except DocumentNotFoundError:
            raise
        
        # Update status to processing
        self.update_document_status(document_id, DocumentStatus.PROCESSING)
        
        try:
            # Get document URL
            file_url = document.get("file_url")
            if not file_url:
                raise DocumentProcessingError("Document has no file URL")
            
            # Download and extract text
            extracted = self._document_processor.process_document_from_url(
                document_id, file_url
            )
            
            # Chunk the text
            chunks = self._text_chunker.chunk_text_raw(extracted.content)
            
            if not chunks:
                # No content to process
                self.update_document_status(document_id, DocumentStatus.READY)
                return ProcessingResult(
                    document_id=document_id,
                    status=DocumentStatus.READY,
                    chunks_created=0
                )
            
            # Generate embeddings
            embeddings = self._embedding_service.generate_embeddings(chunks)
            
            # Store in vector store with metadata
            metadata = {
                "document_type": extracted.document_type.value,
                "page_count": extracted.page_count
            }
            
            self._vector_store.add_embeddings(
                document_id=document_id,
                user_id=user_id,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata
            )
            
            # Update status to ready
            self.update_document_status(document_id, DocumentStatus.READY)
            
            return ProcessingResult(
                document_id=document_id,
                status=DocumentStatus.READY,
                chunks_created=len(chunks)
            )
            
        except (DocumentDownloadError, TextExtractionError) as e:
            # Update status to failed
            self.update_document_status(
                document_id,
                DocumentStatus.FAILED,
                str(e)
            )
            return ProcessingResult(
                document_id=document_id,
                status=DocumentStatus.FAILED,
                chunks_created=0,
                error_message=str(e)
            )
        except Exception as e:
            # Update status to failed for unexpected errors
            self.update_document_status(
                document_id,
                DocumentStatus.FAILED,
                f"Unexpected error: {str(e)}"
            )
            raise DocumentProcessingError(
                f"Failed to process document: {str(e)}"
            ) from e
    
    def delete_document_embeddings(self, document_id: UUID) -> int:
        """Delete all embeddings for a document.
        
        Args:
            document_id: UUID of the document
            
        Returns:
            Number of embeddings deleted
        """
        return self._vector_store.delete_document_embeddings(document_id)
    
    def close(self) -> None:
        """Close resources."""
        self._document_processor.close()
    
    def __enter__(self) -> "DocumentService":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
