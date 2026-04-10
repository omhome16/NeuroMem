"""
Simple API key authentication middleware.

Uses X-API-KEY header for single-user model.
Production would use JWT with tenant isolation.
"""
import logging
from typing import Optional

from fastapi import Request, HTTPException, status, Security, Header
from fastapi.security import APIKeyHeader

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    x_user_id: Optional[str] = Header("default", alias="x-user-id"),
) -> str:
    """
    Validate the API key from X-API-KEY header.

    Returns the user_id (defaults to 'default' if X-USER-ID header is missing).
    This allows the dashboard to pass a unique session UUID for testing isolation.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-KEY header",
        )

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return x_user_id
