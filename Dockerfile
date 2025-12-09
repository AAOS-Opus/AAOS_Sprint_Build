# AAOS Production Dockerfile
# Multi-stage build for optimized production image

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# =============================================================================
# Stage 2: Production
# =============================================================================
FROM python:3.11-slim as production

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 aaos

# Copy installed packages from builder
COPY --from=builder /root/.local /home/aaos/.local

# Copy application code
COPY --chown=aaos:aaos . .

# Create logs directory with proper permissions
RUN mkdir -p /app/logs && chown -R aaos:aaos /app/logs && chmod 755 /app/logs

# Set environment variables
ENV PATH=/home/aaos/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

# Switch to non-root user
USER aaos

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: Run with uvicorn (can be overridden in docker-compose)
CMD ["python", "-m", "uvicorn", "src.orchestrator.core:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--log-level", "info"]
