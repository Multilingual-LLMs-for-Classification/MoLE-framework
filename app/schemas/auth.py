"""
Authentication schemas for OAuth2/JWT.
"""

from typing import Optional

from pydantic import BaseModel, Field


class Token(BaseModel):
    """OAuth2 token response."""

    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Data extracted from JWT token."""

    username: Optional[str] = None


class User(BaseModel):
    """User model."""

    username: str
    disabled: bool = False


class UserInDB(User):
    """User model with hashed password (internal use)."""

    hashed_password: str


class UserCreate(BaseModel):
    """Schema for user registration."""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
