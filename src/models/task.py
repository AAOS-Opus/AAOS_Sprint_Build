# src/models/task.py
from sqlalchemy import Column, String, Integer, Text, DateTime
from datetime import datetime
from .database import Base


class Task(Base):
    """Task model matching Phase 0 discovery evidence"""
    __tablename__ = "tasks"

    task_id = Column(String(36), primary_key=True, index=True)
    task_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    priority = Column(Integer, default=5)
    status = Column(String(20), default="pending")
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
