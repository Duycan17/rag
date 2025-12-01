"""Supabase Vector Store module for document embeddings.

This module provides the SupabaseVectorStore class for storing and
retrieving document embeddings using Supabase's pgvector extension.

Requirements: 2.4, 1.3
"""
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from supabase import Client, create_client

from app.config import Settings


@dataclass
class EmbeddingRecord:
    """Represents a document embedding record."""
    id: UUID
    document_id: UUID
    user_id: UUID
    content: str
    embedding: list[float] | None
    metadata: dict[str, Any] | None
    

@dataclass
class SearchResult:
    """Represents a similarity search result."""
    content: str
    metadata: dict[str, Any]
    similarity: float


class SupabaseVectorStore:
    """Vector store implementation using Supabase pgvector.
    
    This class handles storing document embeddings and performing
    similarity searches with document-scoped filtering.
    
    Requirements:
    - 2.4: Store embeddings with document and user metadata
    - 1.3: Only search within embeddings belonging to specific document
    """
    
    TABLE_NAME = "document_embeddings"
    EMBEDDING_DIMENSION = 768
    
    def __init__(self, settings: Settings) -> None:
        """Initialize the vector store with Supabase client.
        
        Args:
            settings: Application settings containing Supabase credentials
        """
        self._client: Client = create_client(
            settings.supabase_url,
            settings.supabase_key
        )

    def add_embeddings(
        self,
        document_id: UUID,
        user_id: UUID,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: dict[str, Any] | None = None
    ) -> list[UUID]:
        """Add document embeddings to the vector store.
        
        Stores chunks with their embeddings and associated metadata.
        Each chunk is stored as a separate record linked to the document.
        
        Args:
            document_id: UUID of the source document
            user_id: UUID of the document owner
            chunks: List of text chunks to store
            embeddings: List of embedding vectors (must match chunks length)
            metadata: Optional additional metadata to store with each chunk
            
        Returns:
            List of UUIDs for the created embedding records
            
        Raises:
            ValueError: If chunks and embeddings have different lengths
            
        Requirements: 2.4
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings must have same length. "
                f"Got {len(chunks)} chunks and {len(embeddings)} embeddings."
            )
        
        records = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_metadata = {
                "chunk_index": i,
                "document_id": str(document_id),
                "user_id": str(user_id),
                **(metadata or {})
            }
            records.append({
                "document_id": str(document_id),
                "user_id": str(user_id),
                "content": chunk,
                "embedding": embedding,
                "metadata": chunk_metadata
            })
        
        result = self._client.table(self.TABLE_NAME).insert(records).execute()
        
        return [UUID(record["id"]) for record in result.data]

    def similarity_search(
        self,
        query_embedding: list[float],
        document_id: UUID,
        k: int = 4
    ) -> list[SearchResult]:
        """Search for similar chunks within a specific document.
        
        Performs cosine similarity search filtered by document_id to ensure
        document-scoped queries (Requirement 1.3).
        
        Args:
            query_embedding: The embedding vector to search for
            document_id: UUID of the document to search within
            k: Number of results to return (default: 4)
            
        Returns:
            List of SearchResult objects ordered by similarity (highest first)
            
        Requirements: 1.3, 2.4
        """
        # Use Supabase RPC to call pgvector similarity search
        # The function performs cosine similarity and filters by document_id
        result = self._client.rpc(
            "match_document_embeddings",
            {
                "query_embedding": query_embedding,
                "filter_document_id": str(document_id),
                "match_count": k
            }
        ).execute()
        
        return [
            SearchResult(
                content=row["content"],
                metadata=row["metadata"] or {},
                similarity=row["similarity"]
            )
            for row in result.data
        ]
    
    def delete_document_embeddings(self, document_id: UUID) -> int:
        """Delete all embeddings for a specific document.
        
        Args:
            document_id: UUID of the document whose embeddings to delete
            
        Returns:
            Number of records deleted
        """
        result = self._client.table(self.TABLE_NAME).delete().eq(
            "document_id", str(document_id)
        ).execute()
        
        return len(result.data)
    
    def get_document_embedding_count(self, document_id: UUID) -> int:
        """Get the count of embeddings for a document.
        
        Args:
            document_id: UUID of the document
            
        Returns:
            Number of embedding records for the document
        """
        result = self._client.table(self.TABLE_NAME).select(
            "id", count="exact"
        ).eq("document_id", str(document_id)).execute()
        
        return result.count or 0
