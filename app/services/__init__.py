# Services Layer

from app.services.document_processor import (
    DocumentProcessor,
    DocumentType,
    ExtractedDocument,
    DocumentDownloadError,
    TextExtractionError,
)
from app.services.text_chunker import TextChunker, TextChunk
from app.services.embedding_service import EmbeddingService, EmbeddedChunk
from app.services.document_service import (
    DocumentService,
    DocumentStatus,
    ProcessingResult,
    DocumentNotFoundError,
    DocumentProcessingError,
)
from app.services.chat_service import (
    ChatService,
    ChatResponse,
    SourceReference,
    AuthorizationError,
)

__all__ = [
    # Document Processor
    "DocumentProcessor",
    "DocumentType",
    "ExtractedDocument",
    "DocumentDownloadError",
    "TextExtractionError",
    # Text Chunker
    "TextChunker",
    "TextChunk",
    # Embedding Service
    "EmbeddingService",
    "EmbeddedChunk",
    # Document Service
    "DocumentService",
    "DocumentStatus",
    "ProcessingResult",
    "DocumentNotFoundError",
    "DocumentProcessingError",
    # Chat Service
    "ChatService",
    "ChatResponse",
    "SourceReference",
    "AuthorizationError",
]
