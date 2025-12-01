# Shared fixtures for tests
import pytest


@pytest.fixture
def sample_document_id():
    """Sample document ID for testing."""
    return "123e4567-e89b-12d3-a456-426614174000"


@pytest.fixture
def sample_user_id():
    """Sample user ID for testing."""
    return "987fcdeb-51a2-3bc4-d567-890123456789"
