"""Embedding Service module using Google Generative AI.

This module provides embedding generation functionality using
Google's Generative AI embeddings model.

Requirements: 2.3
"""
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import Settings


@dataclass
class EmbeddedChunk:
    """Represents a chunk with its embedding."""
    content: str
    embedding: list[float]
    metadata: dict[str, Any]


class EmbeddingService:
    """Handles embedding generation using Google Generative AI.
    
    Uses langchain-google-genai to generate embeddings for text chunks.
    The embedding model produces 768-dimensional vectors.
    
    Requirements: 2.3
    """
    
    EMBEDDING_MODEL = "models/embedding-001"
    EMBEDDING_DIMENSION = 768
    
    def __init__(self, settings: Settings) -> None:
        """Initialize the embedding service.
        
        Args:
            settings: Application settings containing Gemini API key
        """
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self.EMBEDDING_MODEL,
            google_api_key=settings.gemini_api_key
        )
    
    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            768-dimensional embedding vector
            
        Requirements: 2.3
        """
        return self._embeddings.embed_query(text)
    
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of 768-dimensional embedding vectors
            
        Requirements: 2.3
        """
        if not texts:
            return []
        
        return self._embeddings.embed_documents(texts)
    
    def embed_chunks_with_metadata(
        self,
        chunks: list[str],
        document_id: UUID,
        user_id: UUID,
        additional_metadata: dict[str, Any] | None = None
    ) -> list[EmbeddedChunk]:
        """Generate embeddings for chunks with metadata.
        
        Creates EmbeddedChunk objects containing the chunk content,
        its embedding, and metadata including document_id and user_id.
        
        Args:
            chunks: List of text chunks to embed
            document_id: UUID of the source document
            user_id: UUID of the document owner
            additional_metadata: Optional extra metadata to include
            
        Returns:
            List of EmbeddedChunk objects
            
        Requirements: 2.3
        """
        if not chunks:
            return []
        
        embeddings = self.generate_embeddings(chunks)
        
        result = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata = {
                "document_id": str(document_id),
                "user_id": str(user_id),
                "chunk_index": i,
                **(additional_metadata or {})
            }
            result.append(EmbeddedChunk(
                content=chunk,
                embedding=embedding,
                metadata=metadata
            ))
        
        return result
