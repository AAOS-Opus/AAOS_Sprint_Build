# src/models/task.py
from sqlalchemy import Column, String, Integer, Text, DateTime
from datetime import datetime
from .database import Base


class Task(Base):
    """Task model matching Phase 0 discovery evidence

    Fix #3: Added timeout/retry fields:
    - assigned_at: timestamp when task was assigned to an agent
    - retry_count: number of retries attempted (default 0)
    - max_retries: maximum retry attempts (default 3)
    - retry_after: timestamp for exponential backoff (next eligible retry time)
    """
    __tablename__ = "tasks"

    task_id = Column(String(36), primary_key=True, index=True)
    task_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    priority = Column(Integer, default=5)
    status = Column(String(20), default="pending")
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Fix #3: Task Timeout + Retry fields
    assigned_at = Column(DateTime, nullable=True)  # When task was assigned to an agent
    retry_count = Column(Integer, default=0)  # Number of retries attempted
    max_retries = Column(Integer, default=3)  # Max retry attempts before marking failed
    retry_after = Column(DateTime, nullable=True)  # Exponential backoff: next eligible retry time
