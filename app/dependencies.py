"""
FastAPI dependencies for dependency injection.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.schemas.auth import User
from app.services.auth_service import decode_token, get_user
from app.services.routing_service import routing_service, RoutingService
from app.services.gateway_service import gateway_service, GatewayService


# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)]
) -> User:
    """
    Dependency to get current authenticated user from JWT token.

    Args:
        token: JWT token from Authorization header

    Returns:
        User object if token is valid

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_data = decode_token(token)
    if token_data is None or token_data.username is None:
        raise credentials_exception

    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception

    return User(username=user.username, disabled=user.disabled)


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    Dependency to get current active (non-disabled) user.

    Args:
        current_user: User from get_current_user dependency

    Returns:
        User object if user is active

    Raises:
        HTTPException: If user is disabled
    """
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def get_routing_service() -> RoutingService:
    """
    Dependency to get the routing service singleton.

    Returns:
        RoutingService instance
    """
    return routing_service


def get_gateway_service() -> GatewayService:
    """
    Dependency to get the gateway service singleton (coordinator mode only).

    Returns:
        GatewayService instance, or None if running in monolithic mode.
    """
    return routing_service._gateway


# Type aliases for cleaner dependency injection
CurrentUser = Annotated[User, Depends(get_current_active_user)]
RoutingServiceDep = Annotated[RoutingService, Depends(get_routing_service)]
GatewayServiceDep = Annotated[GatewayService, Depends(get_gateway_service)]
