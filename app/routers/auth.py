"""
Authentication router for OAuth2/JWT endpoints.
"""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.config import settings
from app.schemas.auth import Token, UserCreate, User
from app.services.auth_service import (
    authenticate_user,
    create_access_token,
    create_user,
    get_user,
)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    """
    OAuth2 compatible token login.

    Args:
        form_data: OAuth2 password request form with username and password

    Returns:
        Token with access_token and token_type

    Raises:
        HTTPException: If authentication fails
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=settings.jwt_access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )

    return Token(access_token=access_token, token_type="bearer")


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate) -> User:
    """
    Register a new user.

    Args:
        user_data: User registration data with username and password

    Returns:
        Created user (without password)

    Raises:
        HTTPException: If username already exists
    """
    if get_user(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    user = create_user(user_data.username, user_data.password)
    return User(username=user.username, disabled=user.disabled)
