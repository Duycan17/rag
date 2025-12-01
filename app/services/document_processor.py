"""Document Processor module for text extraction and processing.

This module handles downloading documents from URLs and extracting
text content from various file formats (PDF, TXT).

Requirements: 2.1, 5.2
"""
import io
from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO
from uuid import UUID

import httpx
from pypdf import PdfReader

from app.config import Settings


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    TXT = "txt"
    UNKNOWN = "unknown"


@dataclass
class ExtractedDocument:
    """Represents extracted document content."""
    document_id: UUID
    content: str
    document_type: DocumentType
    page_count: int | None = None


class DocumentDownloadError(Exception):
    """Raised when document download fails."""
    pass


class TextExtractionError(Exception):
    """Raised when text extraction fails."""
    pass


class DocumentProcessor:
    """Handles document downloading and text extraction.
    
    Requirements:
    - 2.1: Extract text content from document files
    - 5.2: Download files from stored URLs
    """
    
    def __init__(self, settings: Settings) -> None:
        """Initialize the document processor.
        
        Args:
            settings: Application settings
        """
        self._settings = settings
        self._http_client = httpx.Client(timeout=30.0)
    
    def download_document(self, url: str) -> bytes:
        """Download a document from a URL.
        
        Args:
            url: The URL to download the document from
            
        Returns:
            The document content as bytes
            
        Raises:
            DocumentDownloadError: If download fails
            
        Requirements: 5.2
        """
        try:
            response = self._http_client.get(url)
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            raise DocumentDownloadError(
                f"Failed to download document: HTTP {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            raise DocumentDownloadError(
                f"Failed to download document: {str(e)}"
            ) from e
    
    def detect_document_type(self, url: str, content: bytes) -> DocumentType:
        """Detect the document type from URL or content.
        
        Args:
            url: The document URL (used for extension detection)
            content: The document content bytes
            
        Returns:
            The detected DocumentType
        """
        # Check URL extension first
        url_lower = url.lower()
        if url_lower.endswith(".pdf"):
            return DocumentType.PDF
        if url_lower.endswith(".txt"):
            return DocumentType.TXT
        
        # Check content magic bytes for PDF
        if content[:4] == b"%PDF":
            return DocumentType.PDF
        
        # Try to decode as text
        try:
            content.decode("utf-8")
            return DocumentType.TXT
        except UnicodeDecodeError:
            pass
        
        return DocumentType.UNKNOWN
    
    def extract_text_from_pdf(self, content: bytes) -> tuple[str, int]:
        """Extract text from PDF content.
        
        Args:
            content: PDF file content as bytes
            
        Returns:
            Tuple of (extracted text, page count)
            
        Raises:
            TextExtractionError: If PDF parsing fails
            
        Requirements: 2.1
        """
        try:
            pdf_file = io.BytesIO(content)
            reader = PdfReader(pdf_file)
            
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            return "\n\n".join(text_parts), len(reader.pages)
        except Exception as e:
            raise TextExtractionError(
                f"Failed to extract text from PDF: {str(e)}"
            ) from e
    
    def extract_text_from_txt(self, content: bytes) -> str:
        """Extract text from plain text content.
        
        Args:
            content: Text file content as bytes
            
        Returns:
            The decoded text content
            
        Raises:
            TextExtractionError: If text decoding fails
            
        Requirements: 2.1
        """
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return content.decode("latin-1")
            except Exception as e:
                raise TextExtractionError(
                    f"Failed to decode text file: {str(e)}"
                ) from e
    
    def extract_text(
        self,
        content: bytes,
        document_type: DocumentType
    ) -> tuple[str, int | None]:
        """Extract text from document content.
        
        Args:
            content: Document content as bytes
            document_type: The type of document
            
        Returns:
            Tuple of (extracted text, page count or None)
            
        Raises:
            TextExtractionError: If extraction fails or type unsupported
            
        Requirements: 2.1
        """
        if document_type == DocumentType.PDF:
            return self.extract_text_from_pdf(content)
        elif document_type == DocumentType.TXT:
            return self.extract_text_from_txt(content), None
        else:
            raise TextExtractionError(
                f"Unsupported document type: {document_type.value}"
            )
    
    def process_document_from_url(
        self,
        document_id: UUID,
        url: str
    ) -> ExtractedDocument:
        """Download and extract text from a document URL.
        
        Args:
            document_id: UUID of the document
            url: URL to download the document from
            
        Returns:
            ExtractedDocument with content and metadata
            
        Raises:
            DocumentDownloadError: If download fails
            TextExtractionError: If extraction fails
            
        Requirements: 2.1, 5.2
        """
        # Download the document
        content = self.download_document(url)
        
        # Detect document type
        doc_type = self.detect_document_type(url, content)
        
        # Extract text
        text, page_count = self.extract_text(content, doc_type)
        
        return ExtractedDocument(
            document_id=document_id,
            content=text,
            document_type=doc_type,
            page_count=page_count
        )
    
    def close(self) -> None:
        """Close the HTTP client."""
        self._http_client.close()
    
    def __enter__(self) -> "DocumentProcessor":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
