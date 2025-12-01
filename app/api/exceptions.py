"""Custom exceptions for the API layer.

This module defines custom exception classes used throughout the API.

Requirements: 3.4, 5.5
"""


class ValidationError(Exception):
    """Raised when request validation fails."""
    pass


class AuthorizationError(Exception):
    """Raised when user is not authorized to access a resource."""
    pass


class NotFoundError(Exception):
    """Raised when a requested resource is not found."""
    pass


class ProcessingError(Exception):
    """Raised when document processing fails."""
    pass
