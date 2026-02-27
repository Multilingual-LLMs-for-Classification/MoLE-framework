"""
Authentication service for JWT token management.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings
from app.schemas.auth import TokenData, UserInDB


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Simple in-memory user store (replace with database in production)
_users_db: Dict[str, UserInDB] = {}


def _init_users_from_config():
    """Initialize users from config (for demo purposes)."""
    global _users_db
    if settings.users_store:
        for user_entry in settings.users_store.split(","):
            if ":" in user_entry:
                username, hashed_password = user_entry.split(":", 1)
                _users_db[username.strip()] = UserInDB(
                    username=username.strip(),
                    hashed_password=hashed_password.strip(),
                    disabled=False
                )


# Initialize users on module load
_init_users_from_config()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from store by username."""
    return _users_db.get(username)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_user(username: str, password: str) -> UserInDB:
    """Create a new user."""
    if username in _users_db:
        raise ValueError(f"User {username} already exists")

    hashed_password = get_password_hash(password)
    user = UserInDB(
        username=username,
        hashed_password=hashed_password,
        disabled=False
    )
    _users_db[username] = user
    return user


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
