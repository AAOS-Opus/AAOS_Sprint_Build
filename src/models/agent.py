# src/models/agent.py
from sqlalchemy import Column, String, Integer, Text, DateTime, Boolean
from datetime import datetime
from .database import Base


class Agent(Base):
    """Agent model for AAOS agents"""
    __tablename__ = "agents"

    agent_id = Column(String(36), primary_key=True, index=True)
    agent_name = Column(String(100), nullable=False)
    agent_type = Column(String(50), nullable=False)
    status = Column(String(20), default="idle")
    capabilities = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    last_heartbeat = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class ReasoningChain(Base):
    """Reasoning chain for agent decision tracking"""
    __tablename__ = "reasoning_chains"

    chain_id = Column(String(36), primary_key=True, index=True)
    agent_id = Column(String(36), nullable=False, index=True)
    task_id = Column(String(36), nullable=True, index=True)
    reasoning_type = Column(String(50), nullable=False)
    steps_json = Column(Text, nullable=True)
    conclusion = Column(Text, nullable=True)
    status = Column(String(20), default="in_progress")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class ConsciousnessSnapshot(Base):
    """Consciousness state snapshots for agents"""
    __tablename__ = "consciousness_snapshots"

    snapshot_id = Column(String(36), primary_key=True, index=True)
    agent_id = Column(String(36), nullable=False, index=True)
    state_json = Column(Text, nullable=True)
    awareness_level = Column(Integer, default=0)
    context_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class AuditLog(Base):
    """Audit log for system events"""
    __tablename__ = "audit_logs"

    log_id = Column(String(36), primary_key=True, index=True)
    event_type = Column(String(50), nullable=False)
    entity_type = Column(String(50), nullable=True)
    entity_id = Column(String(36), nullable=True)
    actor_id = Column(String(36), nullable=True)
    action = Column(String(100), nullable=False)
    details_json = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class AgentCommunication(Base):
    """Inter-agent communication records"""
    __tablename__ = "agent_communications"

    comm_id = Column(String(36), primary_key=True, index=True)
    sender_id = Column(String(36), nullable=False, index=True)
    receiver_id = Column(String(36), nullable=False, index=True)
    message_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=True)
    status = Column(String(20), default="sent")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)


class SystemMetric(Base):
    """System metrics for monitoring"""
    __tablename__ = "system_metrics"

    metric_id = Column(String(36), primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(String(255), nullable=False)
    metric_type = Column(String(50), nullable=False)
    tags_json = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
