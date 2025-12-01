"""Configuration module with environment validation.

This module implements the Settings class using pydantic-settings
for loading and validating environment variables.

Requirements: 4.1, 4.2
"""
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.
    
    Required environment variables:
    - SUPABASE_URL: The Supabase project URL
    - SUPABASE_KEY: The Supabase service role key
    - GEMINI_API_KEY: The Google Gemini API key
    
    Optional environment variables with defaults:
    - CHUNK_SIZE: Size of text chunks (default: 1000)
    - CHUNK_OVERLAP: Overlap between chunks (default: 200)
    - RETRIEVAL_K: Number of chunks to retrieve (default: 4)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Required settings - will raise ValidationError if missing
    supabase_url: str
    supabase_key: str
    gemini_api_key: str
    
    # Optional settings with defaults
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    
    @field_validator("supabase_url")
    @classmethod
    def validate_supabase_url(cls, v: str) -> str:
        """Validate that supabase_url is not empty."""
        if not v or not v.strip():
            raise ValueError("SUPABASE_URL cannot be empty")
        return v.strip()
    
    @field_validator("supabase_key")
    @classmethod
    def validate_supabase_key(cls, v: str) -> str:
        """Validate that supabase_key is not empty."""
        if not v or not v.strip():
            raise ValueError("SUPABASE_KEY cannot be empty")
        return v.strip()
    
    @field_validator("gemini_api_key")
    @classmethod
    def validate_gemini_api_key(cls, v: str) -> str:
        """Validate that gemini_api_key is not empty."""
        if not v or not v.strip():
            raise ValueError("GEMINI_API_KEY cannot be empty")
        return v.strip()
    
    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        """Validate chunk_size is positive."""
        if v <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        return v
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int) -> int:
        """Validate chunk_overlap is non-negative."""
        if v < 0:
            raise ValueError("CHUNK_OVERLAP must be non-negative")
        return v
    
    @field_validator("retrieval_k")
    @classmethod
    def validate_retrieval_k(cls, v: int) -> int:
        """Validate retrieval_k is positive."""
        if v <= 0:
            raise ValueError("RETRIEVAL_K must be positive")
        return v


def get_settings() -> Settings:
    """Get application settings.
    
    Raises:
        ValidationError: If required environment variables are missing
    """
    return Settings()
