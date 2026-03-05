"""
Authentication service for JWT token management.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import settings
from app.db import UserRecord
from app.schemas.auth import TokenData, UserInDB


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def get_user(db: Session, username: str) -> Optional[UserInDB]:
    """Get user from database by username."""
    record = db.query(UserRecord).filter(UserRecord.username == username).first()
    if record is None:
        return None
    return UserInDB(
        username=record.username,
        hashed_password=record.hashed_password,
        disabled=record.disabled,
    )


def authenticate_user(db: Session, username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with username and password."""
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_user(db: Session, username: str, password: str) -> UserInDB:
    """Create a new user in the database."""
    if db.query(UserRecord).filter(UserRecord.username == username).first():
        raise ValueError(f"User {username} already exists")

    hashed_password = get_password_hash(password)
    record = UserRecord(username=username, hashed_password=hashed_password, disabled=False)
    db.add(record)
    db.commit()
    db.refresh(record)
    return UserInDB(
        username=record.username,
        hashed_password=record.hashed_password,
        disabled=record.disabled,
    )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in the token (typically {"sub": username})
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.jwt_access_token_expire_minutes
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenData with username if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        username: str = payload.get("sub")
        if username is None:
            return None
        return TokenData(username=username)
    except JWTError:
        return None
