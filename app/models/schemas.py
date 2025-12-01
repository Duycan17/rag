"""Pydantic models for API request/response schemas.

This module defines the data models used by the REST API endpoints.

Requirements: 3.1, 3.2, 3.3, 5.1, 5.4
"""
from typing import Any, Dict, List
from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint.
    
    Requirements: 3.1, 3.2
    """
    user_id: UUID = Field(..., description="UUID of the user sending the message")
    document_id: UUID = Field(..., description="UUID of the document to query")
    message: str = Field(..., min_length=1, description="The user's question/message")


class SourceReference(BaseModel):
    """Source reference in a chat response.
    
    Requirements: 1.2, 3.3
    """
    content: str = Field(..., description="The source content chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source metadata")


class ChatResponse(BaseModel):
    """Response model for chat endpoint.
    
    Requirements: 3.3
    """
    answer: str = Field(..., description="The generated answer")
    sources: List[SourceReference] = Field(default_factory=list, description="Source references")


class ProcessRequest(BaseModel):
    """Request model for document processing endpoint.
    
    Requirements: 5.1
    """
    user_id: UUID = Field(..., description="UUID of the document owner")
    document_id: UUID = Field(..., description="UUID of the document to process")


class ProcessResponse(BaseModel):
    """Response model for document processing endpoint.
    
    Requirements: 5.4
    """
    status: str = Field(..., description="Processing status")
    chunks_created: int = Field(..., description="Number of chunks created")
    document_id: UUID = Field(..., description="UUID of the processed document")


class ErrorResponse(BaseModel):
    """Standard error response model.
    
    Requirements: 3.4, 5.5
    """
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    code: str = Field(..., description="Error code")
