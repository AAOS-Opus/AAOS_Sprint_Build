"""
AAOS Authentication Module
API Key validation for HTTP and WebSocket endpoints

Security-critical: Uses constant-time comparison to prevent timing attacks
"""

import os
import secrets
import logging
from typing import Optional

from fastapi import HTTPException, Header, WebSocket, status

logger = logging.getLogger("aaos.auth")

# API key from environment (MUST be set in production)
AAOS_API_KEY: Optional[str] = os.getenv("AAOS_API_KEY")

if not AAOS_API_KEY:
    logger.warning(
        "AAOS_API_KEY not set - authentication DISABLED (development mode only)"
    )


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> str:
    """
    FastAPI dependency for HTTP endpoints - validates API key from X-API-Key header.

    Security:
    - Uses constant-time comparison (secrets.compare_digest) to prevent timing attacks
    - Only accepts header-based auth (no query params to avoid URL logging exposure)

    Returns:
        str: The validated API key (or "dev-bypass" in development mode)

    Raises:
        HTTPException: 401 if key missing, 403 if key invalid
    """
    # Development mode bypass (only when AAOS_API_KEY not configured)
    if not AAOS_API_KEY:
        logger.debug("Authentication bypassed - AAOS_API_KEY not configured")
        return "dev-bypass"

    # Check for missing key
    if not x_api_key:
        logger.warning("Request rejected - missing X-API-Key header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(x_api_key.encode("utf-8"), AAOS_API_KEY.encode("utf-8")):
        # Log partial key for debugging (first 8 chars only)
        key_preview = x_api_key[:8] if len(x_api_key) >= 8 else x_api_key
        logger.warning(f"Request rejected - invalid API key: {key_preview}...")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )

    return x_api_key


async def verify_websocket_auth(websocket: WebSocket) -> bool:
    """
    Verify WebSocket authentication BEFORE calling websocket.accept().

    Security:
    - Only checks X-API-Key header (no query params)
    - Uses constant-time comparison
    - Must be called BEFORE websocket.accept() to reject unauthenticated connections

    Args:
        websocket: The WebSocket connection to verify

    Returns:
        bool: True if authenticated, False otherwise
    """
    # Development mode bypass
    if not AAOS_API_KEY:
        logger.debug("WebSocket auth bypassed - AAOS_API_KEY not configured")
        return True

    # Get API key from header only (not query params for security)
    api_key = websocket.headers.get("X-API-Key")

    if not api_key:
        logger.warning(
            f"WebSocket rejected - missing X-API-Key header "
            f"(client: {websocket.client.host if websocket.client else 'unknown'})"
        )
        return False

    # Constant-time comparison
    if not secrets.compare_digest(api_key.encode("utf-8"), AAOS_API_KEY.encode("utf-8")):
        key_preview = api_key[:8] if len(api_key) >= 8 else api_key
        logger.warning(
            f"WebSocket rejected - invalid API key: {key_preview}... "
            f"(client: {websocket.client.host if websocket.client else 'unknown'})"
        )
        return False

    return True


def generate_api_key(length: int = 32) -> str:
    """
    Generate a cryptographically secure API key.

    Usage:
        python -c "from src.auth import generate_api_key; print(generate_api_key())"

    Args:
        length: Number of random bytes (default 32 = 256 bits)

    Returns:
        str: URL-safe base64 encoded key
    """
    return secrets.token_urlsafe(length)


def is_auth_enabled() -> bool:
    """Check if authentication is enabled (API key configured)."""
    return AAOS_API_KEY is not None and len(AAOS_API_KEY) > 0
