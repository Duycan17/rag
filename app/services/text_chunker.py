"""Text Chunking module using LangChain.

This module provides text chunking functionality using LangChain's
RecursiveCharacterTextSplitter for splitting document content into
chunks suitable for embedding.

Requirements: 2.2
"""
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    content: str
    index: int
    start_char: int | None = None
    end_char: int | None = None


class TextChunker:
    """Handles text chunking using LangChain's RecursiveCharacterTextSplitter.
    
    The chunker splits text into overlapping chunks suitable for embedding,
    using configurable chunk_size and chunk_overlap from settings.
    
    Requirements: 2.2
    """
    
    def __init__(self, settings: Settings) -> None:
        """Initialize the text chunker with settings.
        
        Args:
            settings: Application settings containing chunk_size and chunk_overlap
        """
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap
        
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    @property
    def chunk_size(self) -> int:
        """Get the configured chunk size."""
        return self._chunk_size
    
    @property
    def chunk_overlap(self) -> int:
        """Get the configured chunk overlap."""
        return self._chunk_overlap
    
    def chunk_text(self, text: str) -> list[TextChunk]:
        """Split text into chunks.
        
        Uses RecursiveCharacterTextSplitter to split text into overlapping
        chunks. Each chunk will have length <= chunk_size.
        
        Args:
            text: The text content to split
            
        Returns:
            List of TextChunk objects with content and metadata
            
        Requirements: 2.2
        """
        if not text or not text.strip():
            return []
        
        # Use LangChain's splitter to get chunks
        chunks = self._splitter.split_text(text)
        
        # Create TextChunk objects with index
        return [
            TextChunk(content=chunk, index=i)
            for i, chunk in enumerate(chunks)
        ]
    
    def chunk_text_raw(self, text: str) -> list[str]:
        """Split text into chunks and return raw strings.
        
        Convenience method that returns just the chunk strings
        without metadata.
        
        Args:
            text: The text content to split
            
        Returns:
            List of chunk strings
            
        Requirements: 2.2
        """
        if not text or not text.strip():
            return []
        
        return self._splitter.split_text(text)
