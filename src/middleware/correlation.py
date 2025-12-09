# src/middleware/correlation.py
"""
Correlation ID Middleware - DevZen Enhancement #7
Ensures every request/response has a traceable correlation ID for telemetry
"""

import uuid
import logging
from contextvars import ContextVar
from typing import Optional
from datetime import datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Context variable for correlation ID (thread-safe for async)
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='no-corr-id')


def get_correlation_id() -> str:
    """Get current correlation ID from context"""
    return correlation_id_var.get()


def generate_correlation_id() -> str:
    """Generate a new correlation ID"""
    return f"corr-{uuid.uuid4().hex[:12]}"


class CorrelationIdFilter(logging.Filter):
    """Logging filter to inject correlation ID into log records"""

    def filter(self, record):
        record.correlation_id = get_correlation_id()
        record.iso_timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")
        return True


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that ensures every request has a correlation ID.

    DevZen Telemetry Requirements:
    - Every log entry must include ISO timestamp and correlation ID
    - Format: [2025-11-26T00:00:00.000000] [corr-id:abc123] message
    """

    HEADER_NAME = "X-Correlation-ID"

    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract or generate correlation ID
        corr_id = request.headers.get(self.HEADER_NAME)
        if not corr_id:
            corr_id = generate_correlation_id()

        # Set context variable for use in handlers
        token = correlation_id_var.set(corr_id)

        try:
            # Process request
            response = await call_next(request)

            # Add correlation ID to response headers
            response.headers[self.HEADER_NAME] = corr_id

            return response
        finally:
            # Reset context variable
            correlation_id_var.reset(token)


class StructuredLogger:
    """
    Structured logger wrapper for DevZen telemetry format compliance.

    Output format:
    [2025-11-26T00:00:00.000000] [corr-id:abc123] Task created: task_id=...
    """

    def __init__(self, name: str = "aaos"):
        self.logger = logging.getLogger(name)
        # Add correlation ID filter if not already present
        if not any(isinstance(f, CorrelationIdFilter) for f in self.logger.filters):
            self.logger.addFilter(CorrelationIdFilter())

    def _format_message(self, message: str) -> str:
        """Format message with DevZen telemetry format"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")
        corr_id = get_correlation_id()
        return f"[{timestamp}] [corr-id:{corr_id}] {message}"

    def info(self, message: str, **kwargs):
        self.logger.info(self._format_message(message), **kwargs)

    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format_message(message), **kwargs)

    def error(self, message: str, **kwargs):
        self.logger.error(self._format_message(message), **kwargs)

    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format_message(message), **kwargs)

    def critical(self, message: str, **kwargs):
        self.logger.critical(self._format_message(message), **kwargs)


# Convenience function to get a structured logger
def get_structured_logger(name: str = "aaos.orchestrator") -> StructuredLogger:
    """Get a structured logger with correlation ID support"""
    return StructuredLogger(name)
