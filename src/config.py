# src/config.py
"""
AAOS Configuration Module - Environment-based configuration for all environments
DevZen Enhanced: Supports dev, staging, and production configurations
"""

import os
from typing import Optional
from functools import lru_cache

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables


class Settings:
    """Application settings loaded from environment variables"""

    # ==========================================================================
    # Application Settings
    # ==========================================================================
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ==========================================================================
    # FastAPI Settings
    # ==========================================================================
    FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
    FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", "8000"))
    FASTAPI_WORKERS: int = int(os.getenv("FASTAPI_WORKERS", "1"))

    # ==========================================================================
    # Database Configuration
    # ==========================================================================
    @property
    def DATABASE_URL(self) -> str:
        """Get database URL with fallback for development"""
        url = os.getenv("DATABASE_URL")
        if url:
            return url
        # Default to SQLite for development
        return "sqlite:///./aaos.db"

    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))

    # ==========================================================================
    # Redis Configuration
    # ==========================================================================
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD") or None

    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL from components"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ==========================================================================
    # Telemetry Settings
    # ==========================================================================
    TELEMETRY_ENABLED: bool = os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"
    TELEMETRY_SAMPLE_INTERVAL: int = int(os.getenv("TELEMETRY_SAMPLE_INTERVAL", "60"))
    TELEMETRY_FLUSH_INTERVAL: int = int(os.getenv("TELEMETRY_FLUSH_INTERVAL", "300"))
    TELEMETRY_MAX_LOG_SIZE_MB: int = int(os.getenv("TELEMETRY_MAX_LOG_SIZE_MB", "100"))

    # ==========================================================================
    # Load Testing Settings
    # ==========================================================================
    LOAD_TASKS_PER_HOUR: int = int(os.getenv("LOAD_TASKS_PER_HOUR", "100"))
    LOAD_AGENT_COUNT: int = int(os.getenv("LOAD_AGENT_COUNT", "5"))
    LOAD_DURATION_HOURS: int = int(os.getenv("LOAD_DURATION_HOURS", "24"))

    # ==========================================================================
    # Computed Properties
    # ==========================================================================
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT.lower() == "testing"

    def get_log_config(self) -> dict:
        """Get structured logging configuration"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%dT%H:%M:%S"
                },
                "structured": {
                    "format": "[%(asctime)s] [corr-id:%(correlation_id)s] %(message)s",
                    "datefmt": "%Y-%m-%dT%H:%M:%S.%f"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "default",
                    "filename": "aaos_prod.log",
                    "maxBytes": self.TELEMETRY_MAX_LOG_SIZE_MB * 1024 * 1024,
                    "backupCount": 5
                }
            },
            "root": {
                "level": self.LOG_LEVEL,
                "handlers": ["console", "file"]
            }
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
