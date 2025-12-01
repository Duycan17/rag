"""Pydantic models for MCQ generation API request/response schemas.

This module defines the data models used by the MCQ generation endpoint.

Requirements: 3.1, 3.2, 3.3
"""
from enum import Enum
from typing import List
from uuid import UUID

from pydantic import BaseModel, Field, validator


class DifficultyLevel(str, Enum):
    """Difficulty level for MCQ generation.
    
    Requirements: 2.1, 2.2, 2.3, 2.4
    """
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class MCQRequest(BaseModel):
    """Request model for MCQ generation endpoint.
    
    Requirements: 3.1, 3.2
    """
    user_id: UUID = Field(..., description="UUID of the user requesting MCQs")
    document_id: UUID = Field(..., description="UUID of the document to generate MCQs from")
    num_questions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of questions to generate (1-20)"
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM,
        description="Difficulty level of questions"
    )


class MCQQuestion(BaseModel):
    """Model for a single MCQ question.
    
    Requirements: 1.2, 3.3, 4.1
    """
    question: str = Field(..., min_length=1, description="The question text")
    options: List[str] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="Exactly 4 answer options"
    )
    correct_answer_index: int = Field(
        ...,
        ge=0,
        le=3,
        description="Index of the correct answer (0-3)"
    )
    explanation: str = Field(..., min_length=1, description="Explanation for the correct answer")

    @validator("options")
    def validate_options_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure all options are non-empty strings."""
        for i, option in enumerate(v):
            if not option or not option.strip():
                raise ValueError(f"Option at index {i} cannot be empty")
        return v


class MCQResponse(BaseModel):
    """Response model for MCQ generation endpoint.
    
    Requirements: 3.3
    """
    questions: List[MCQQuestion] = Field(..., description="List of generated MCQ questions")
    document_id: UUID = Field(..., description="UUID of the source document")
    difficulty: str = Field(..., description="Difficulty level used for generation")
    generated_count: int = Field(..., ge=0, description="Number of questions generated")
