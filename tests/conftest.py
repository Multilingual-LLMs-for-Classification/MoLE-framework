"""
Pytest configuration and fixtures for the classification service tests.
"""

import os

# Must be set before any app imports so Settings() picks them up at module load time.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_users.db")
os.environ.setdefault("SERVICE_MODE", "coordinator")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-ci-only")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """Create a test client for the FastAPI application (shared across the session)."""
    with TestClient(app) as c:
        yield c
