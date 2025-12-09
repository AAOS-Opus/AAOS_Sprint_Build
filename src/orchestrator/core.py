# src/orchestrator/core.py
"""
AAOS Orchestrator Core - Phase 1 & 2 Implementation
Cross-platform, production-ready with Redis health checks, structured logging,
session-safe database operations, and WebSocket agent lifecycle support.

Integration 2A: Agent Lifecycle Health Monitor integrated
"""

import logging
import sys
import platform
import uuid
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List, Set
from enum import Enum

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, validator
from pydantic.types import constr, conint
from sqlalchemy import inspect
from sqlalchemy.orm import Session
import redis

# Import models
from src.models import Base, engine, SessionLocal, get_db, Task, Agent

# Import agent lifecycle module (Integration 2A)
from src.agent_lifecycle import LifecycleOrchestrator, HealthMonitor, CircuitBreaker, CircuitState

# Configure unified logging with structured context
import os
log_dir = os.environ.get("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "aaos.log")

# Redis configuration from environment
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("aaos.orchestrator")

# Create FastAPI application
app = FastAPI(
    title="AAOS Orchestrator",
    description="Autonomous Agent Orchestration System",
    version="1.0.0"
)


# =============================================================================
# Agent Lifecycle Orchestrator (Integration 2A)
# =============================================================================

# Global lifecycle orchestrator instance
lifecycle_orchestrator: Optional[LifecycleOrchestrator] = None


# =============================================================================
# Redis Health Check
# =============================================================================

def verify_redis_health():
    """Verify Redis is available before queue operations"""
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        if not r.ping():
            raise ConnectionError("Redis ping failed")
        return r
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        raise HTTPException(status_code=503, detail="Redis unavailable - cannot queue tasks")


# =============================================================================
# Schema Verification Guard (Startup Event)
# =============================================================================

@app.on_event("startup")
async def verify_schema():
    """Verify all required tables exist on startup"""
    inspector = inspect(engine)
    required_tables = [
        "tasks",
        "agents",
        "reasoning_chains",
        "consciousness_snapshots",
        "audit_logs",
        "agent_communications",
        "system_metrics"
    ]
    existing_tables = inspector.get_table_names()

    missing = [t for t in required_tables if t not in existing_tables]
    if missing:
        logger.error(f"Schema verification failed. Missing tables: {missing}")
        raise RuntimeError(
            f"Missing tables: {missing}. Run: alembic upgrade head"
        )
    logger.info(f"Schema verification passed - all {len(required_tables)} tables present")


@app.on_event("startup")
async def initialize_lifecycle_orchestrator():
    """
    Initialize the Agent Lifecycle Health Monitor (Integration 2A)
    - Creates LifecycleOrchestrator
    - Registers 9 ensemble agents
    - Initializes circuit breakers in CLOSED state
    - Waits for heartbeat stabilization (3s minimum)
    """
    global lifecycle_orchestrator

    logger.info("Initializing Agent Lifecycle Health Monitor...")

    # Create and initialize the orchestrator
    lifecycle_orchestrator = LifecycleOrchestrator()
    await lifecycle_orchestrator.initialize()

    # Start the orchestrator (registers agents, waits for stabilization)
    await lifecycle_orchestrator.start()

    # Verify all circuit breakers are in CLOSED state
    if lifecycle_orchestrator.verify_circuit_breakers_closed():
        logger.info("All circuit breakers initialized in CLOSED state")
    else:
        logger.warning("Some circuit breakers not in CLOSED state")

    # Log registered agents
    agent_ids = lifecycle_orchestrator.get_agent_ids()
    logger.info(f"Registered {len(agent_ids)} ensemble agents: {agent_ids}")

    # Verify GATE condition
    if lifecycle_orchestrator.readiness():
        logger.info("GATE PASSED: health_monitor.readiness() == True")
    else:
        logger.warning("GATE PENDING: health_monitor.readiness() != True")


@app.on_event("shutdown")
async def shutdown_lifecycle_orchestrator():
    """Stop the lifecycle orchestrator on shutdown"""
    global lifecycle_orchestrator
    if lifecycle_orchestrator:
        await lifecycle_orchestrator.stop()
        logger.info("Lifecycle orchestrator stopped")


# =============================================================================
# Pydantic Models (Phase 4)
# =============================================================================

class TaskType(str, Enum):
    """Strict enum from Phase 0 protocol discovery"""
    code = "code"
    research = "research"
    qa = "qa"
    documentation = "documentation"
    analysis = "analysis"
    synthesis = "synthesis"


class TaskCreate(BaseModel):
    """Validated task creation with meaningful constraints"""
    task_type: TaskType = Field(..., description="Task category from enum")
    description: constr(min_length=3, max_length=2000) = Field(..., description="Task description")
    priority: Optional[conint(ge=0, le=10)] = Field(default=5, description="Priority 0-10")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")

    @validator('description')
    def description_meaningful(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Description must be meaningful (>=3 characters after trimming)')
        return v.strip()

    @validator('metadata')
    def metadata_valid_json(cls, v):
        if v is None:
            return {}
        return v

    class Config:
        extra = "forbid"  # Reject unknown fields
        use_enum_values = True


class TaskResponse(BaseModel):
    """Response model for task operations"""
    task_id: str
    status: str
    message: str
    queued_at: str


class TaskDetail(BaseModel):
    """Full task details response"""
    task_id: str
    task_type: str
    status: str
    description: str
    priority: int
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


# =============================================================================
# Health Check Endpoint
# =============================================================================

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "platform": platform.system(),
        "timestamp": datetime.utcnow().isoformat()
    }


# =============================================================================
# Agent Health Probe Endpoint (Integration 2A)
# =============================================================================

@app.get("/health/agents")
async def get_agent_health():
    """
    Health probe endpoint for agent lifecycle monitoring.
    Returns status of all 9 ensemble agents and circuit breakers.

    Integration 2A: GET /health/agents
    """
    global lifecycle_orchestrator

    if not lifecycle_orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Lifecycle orchestrator not initialized"
        )

    # Get comprehensive health status
    status = lifecycle_orchestrator.get_health_status()

    # Add readiness gate check
    status["gate_passed"] = lifecycle_orchestrator.readiness()

    return status


# =============================================================================
# POST /tasks Endpoint (Phase 5)
# =============================================================================

@app.post("/tasks", status_code=201, response_model=TaskResponse)
async def create_task(
    task: TaskCreate,
    db: Session = Depends(get_db)
):
    """
    Create task with validation, Redis queueing, and structured logging
    """
    task_id = uuid.uuid4()

    # Verify Redis health before queue operation
    redis_client = verify_redis_health()

    try:
        # Convert metadata to JSON string for SQLite
        metadata_json = json.dumps(task.metadata) if task.metadata else "{}"

        # Create task record
        db_task = Task(
            task_id=str(task_id),
            task_type=task.task_type,
            description=task.description,
            priority=task.priority,
            status="pending",
            metadata_json=metadata_json,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        db.add(db_task)
        db.commit()
        db.refresh(db_task)

        # Queue to Redis with error handling
        try:
            redis_client.lpush("task_queue", str(task_id))
        except Exception as redis_error:
            logger.error(f"Redis queue failed for task {task_id}: {redis_error}")
            # Rollback DB if Redis fails for consistency
            db.rollback()
            raise HTTPException(status_code=503, detail="Task queue unavailable")

        # Structured logging with context
        log_context = {
            "task_id": str(task_id),
            "type": task.task_type,
            "priority": task.priority,
            "description_preview": task.description[:50]
        }
        logger.info(f"Task created successfully | {json.dumps(log_context)}")

        return TaskResponse(
            task_id=str(task_id),
            status="pending",
            message="Task queued successfully",
            queued_at=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        db.rollback()
        logger.error(f"Task creation failed: {str(e)}", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Task creation failed: {str(e)}")


# =============================================================================
# GET /tasks/{task_id} Endpoint
# =============================================================================

@app.get("/tasks/{task_id}", response_model=TaskDetail)
async def get_task(task_id: str, db: Session = Depends(get_db)):
    """
    Retrieve task with database session management
    """
    task = db.query(Task).filter(Task.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return TaskDetail(
        task_id=task.task_id,
        task_type=task.task_type,
        status=task.status,
        description=task.description,
        priority=task.priority,
        metadata=json.loads(task.metadata_json) if task.metadata_json else {},
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat()
    )


# =============================================================================
# WebSocket Connection Manager (Phase 2)
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections for agents"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.agent_info: Dict[str, Dict[str, Any]] = {}
        self.agent_tasks: Dict[str, str] = {}  # agent_id -> task_id mapping
        self.completed_tasks_total: int = 0

    async def connect(self, websocket: WebSocket, agent_id: str):
        await websocket.accept()
        self.active_connections[agent_id] = websocket

    def disconnect(self, agent_id: str):
        if agent_id in self.active_connections:
            del self.active_connections[agent_id]
        if agent_id in self.agent_info:
            self.agent_info[agent_id]["status"] = "disconnected"

    async def send_message(self, agent_id: str, message: Dict[str, Any]):
        if agent_id in self.active_connections:
            await self.active_connections[agent_id].send_json(message)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections.values():
            await connection.send_json(message)

    def register_agent(self, agent_id: str, agent_type: str, capabilities: Dict[str, Any]):
        self.agent_info[agent_id] = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "capabilities": capabilities,
            "status": "idle",
            "registered_at": datetime.utcnow().isoformat(),
            "last_heartbeat": datetime.utcnow().isoformat()
        }

    def update_heartbeat(self, agent_id: str):
        if agent_id in self.agent_info:
            self.agent_info[agent_id]["last_heartbeat"] = datetime.utcnow().isoformat()

    def get_idle_agent(self) -> Optional[str]:
        """Get an idle agent for task assignment"""
        for agent_id, info in self.agent_info.items():
            if info["status"] == "idle" and agent_id in self.active_connections:
                return agent_id
        return None

    def set_agent_status(self, agent_id: str, status: str):
        if agent_id in self.agent_info:
            self.agent_info[agent_id]["status"] = status

    def get_all_agents(self) -> List[Dict[str, Any]]:
        return list(self.agent_info.values())

    def assign_task_to_agent(self, agent_id: str, task_id: str):
        """Track which task is assigned to which agent"""
        self.agent_tasks[agent_id] = task_id

    def complete_task_for_agent(self, agent_id: str):
        """Clear task assignment and increment counter"""
        if agent_id in self.agent_tasks:
            del self.agent_tasks[agent_id]
        self.completed_tasks_total += 1

    def get_assigned_task(self, agent_id: str) -> Optional[str]:
        """Get the task currently assigned to an agent"""
        return self.agent_tasks.get(agent_id)

    def get_active_agent_count(self) -> int:
        """Count of currently connected agents"""
        return len(self.active_connections)

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        queue_len = redis_client.llen("task_queue")
        return {
            "active_agents": self.get_active_agent_count(),
            "queued_tasks": queue_len,
            "completed_tasks_total": self.completed_tasks_total,
            "agents_busy": len(self.agent_tasks),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global connection manager
manager = ConnectionManager()

# Background task for assigning tasks to agents
task_assignment_running = False


async def task_assignment_loop():
    """Background loop that assigns pending tasks to idle agents (FIFO-preserving)"""
    global task_assignment_running
    task_assignment_running = True
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    while task_assignment_running:
        try:
            # FIFO FIX: Peek at next task without removing (lindex -1 = rightmost = oldest)
            task_id = redis_client.lindex("task_queue", -1)
            if not task_id:
                await asyncio.sleep(0.1)
                continue

            # Find an idle agent BEFORE popping
            agent_id = manager.get_idle_agent()
            if not agent_id:
                # No agent available - leave task in queue (preserves FIFO)
                await asyncio.sleep(0.1)
                continue

            # Agent available - now safely pop the task (atomic FIFO preservation)
            task_id = redis_client.rpop("task_queue")
            if not task_id:
                # Race condition: another consumer got it - continue
                continue

            logger.info(f"Dequeued task {task_id} for agent {agent_id}")  # DevZen Enhancement #2

            # Get task details from database
            db = SessionLocal()
            try:
                task = db.query(Task).filter(Task.task_id == task_id).first()
                if task and task.status == "pending":
                    # Send task assignment to agent
                    assignment = {
                        "type": "task_assignment",
                        "task_id": task_id,
                        "task_type": task.task_type,
                        "description": task.description,
                        "priority": task.priority,
                        "metadata": json.loads(task.metadata_json) if task.metadata_json else {}
                    }
                    await manager.send_message(agent_id, assignment)

                    # Update task and agent status
                    task.status = "assigned"
                    task.updated_at = datetime.utcnow()
                    db.commit()

                    manager.set_agent_status(agent_id, "busy")
                    manager.assign_task_to_agent(agent_id, task_id)

                    logger.info(f"Task {task_id} assigned to {agent_id}")  # Traceability
                else:
                    # Task not found or already processed, skip
                    logger.warning(f"Task {task_id} not pending (status={task.status if task else 'NOT FOUND'})")
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Task assignment error: {e}")
            await asyncio.sleep(1)


@app.on_event("startup")
async def start_task_assignment():
    """Start the background task assignment loop"""
    asyncio.create_task(task_assignment_loop())


@app.on_event("shutdown")
async def stop_task_assignment():
    """Stop the background task assignment loop"""
    global task_assignment_running
    task_assignment_running = False


# =============================================================================
# WebSocket Endpoint for Agents
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for agent communication
    Protocol: register -> registration_ack -> heartbeat/task_assignment/task_complete
    """
    agent_id = None

    try:
        # Accept connection but wait for registration
        await websocket.accept()

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "register":
                # Agent registration
                agent_id = data.get("agent_id")
                agent_type = data.get("agent_type", "unknown")
                capabilities = data.get("capabilities", {})

                # Store connection
                manager.active_connections[agent_id] = websocket
                manager.register_agent(agent_id, agent_type, capabilities)

                # Persist to database
                db = SessionLocal()
                try:
                    existing = db.query(Agent).filter(Agent.agent_id == agent_id).first()
                    if existing:
                        existing.status = "idle"
                        existing.is_active = True
                        existing.last_heartbeat = datetime.utcnow()
                        existing.updated_at = datetime.utcnow()
                    else:
                        new_agent = Agent(
                            agent_id=agent_id,
                            agent_name=agent_id,
                            agent_type=agent_type,
                            status="idle",
                            capabilities=json.dumps(capabilities),
                            is_active=True,
                            last_heartbeat=datetime.utcnow(),
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow()
                        )
                        db.add(new_agent)
                    db.commit()
                finally:
                    db.close()

                # Send acknowledgment
                ack = {
                    "type": "registration_ack",
                    "agent_id": agent_id,
                    "status": "registered",
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_json(ack)

                logger.info(f"Agent registered: {agent_id}")

            elif msg_type == "heartbeat":
                # Update heartbeat timestamp
                hb_agent_id = data.get("agent_id", agent_id)
                if hb_agent_id:
                    manager.update_heartbeat(hb_agent_id)

                    # Update database
                    db = SessionLocal()
                    try:
                        agent = db.query(Agent).filter(Agent.agent_id == hb_agent_id).first()
                        if agent:
                            agent.last_heartbeat = datetime.utcnow()
                            db.commit()
                    finally:
                        db.close()

            elif msg_type == "task_complete":
                # Task completion from agent
                task_id = data.get("task_id")
                completing_agent_id = data.get("agent_id", agent_id)
                result = data.get("result", {})
                reasoning_steps = data.get("reasoning_steps", [])

                # Update task in database
                db = SessionLocal()
                try:
                    task = db.query(Task).filter(Task.task_id == task_id).first()
                    if task:
                        task.status = "completed"
                        task.updated_at = datetime.utcnow()
                        # Store result in metadata
                        existing_meta = json.loads(task.metadata_json) if task.metadata_json else {}
                        existing_meta["result"] = result
                        existing_meta["reasoning_steps"] = reasoning_steps
                        existing_meta["completed_by"] = completing_agent_id
                        existing_meta["completed_at"] = datetime.utcnow().isoformat()
                        task.metadata_json = json.dumps(existing_meta)
                        db.commit()

                        logger.info(f"Task completed: {task_id} by {completing_agent_id}")
                finally:
                    db.close()

                # Set agent back to idle and track completion
                if completing_agent_id:
                    manager.set_agent_status(completing_agent_id, "idle")
                    manager.complete_task_for_agent(completing_agent_id)

            else:
                logger.warning(f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        if agent_id:
            # Check if agent had an assigned task - requeue it
            assigned_task_id = manager.get_assigned_task(agent_id)
            if assigned_task_id:
                redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
                # Reset task status to pending and requeue
                db = SessionLocal()
                try:
                    task = db.query(Task).filter(Task.task_id == assigned_task_id).first()
                    if task and task.status == "assigned":
                        task.status = "pending"
                        task.updated_at = datetime.utcnow()
                        db.commit()
                        redis_client.lpush("task_queue", assigned_task_id)
                        logger.info(f"Task requeued due to agent disconnect: {assigned_task_id}")
                finally:
                    db.close()

            manager.disconnect(agent_id)

            # Update database
            db = SessionLocal()
            try:
                agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
                if agent:
                    agent.status = "disconnected"
                    agent.is_active = False
                    agent.updated_at = datetime.utcnow()
                    db.commit()
            finally:
                db.close()

            logger.info(f"Agent disconnected: {agent_id}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if agent_id:
            manager.disconnect(agent_id)


# =============================================================================
# GET /agents Endpoint
# =============================================================================

@app.get("/agents")
async def get_agents():
    """
    Get list of all registered agents
    """
    return manager.get_all_agents()


# =============================================================================
# GET /metrics Endpoint (Phase 3)
# =============================================================================

@app.get("/metrics")
async def get_metrics():
    """
    Get system metrics for monitoring
    """
    return manager.get_metrics()


# =============================================================================
# GET /tasks List Endpoint (Phase 3)
# =============================================================================

class TaskListItem(BaseModel):
    """Task list item response"""
    task_id: str
    task_type: str
    status: str
    description: str
    priority: int
    created_at: str
    updated_at: str


@app.get("/tasks")
async def list_tasks(db: Session = Depends(get_db)):
    """
    List all tasks with optional status filter
    """
    tasks = db.query(Task).order_by(Task.created_at.desc()).limit(100).all()
    return [
        {
            "task_id": t.task_id,
            "task_type": t.task_type,
            "status": t.status,
            "description": t.description,
            "priority": t.priority,
            "metadata": json.loads(t.metadata_json) if t.metadata_json else {},
            "agent_id": json.loads(t.metadata_json).get("completed_by") if t.metadata_json else None,
            "created_at": t.created_at.isoformat(),
            "updated_at": t.updated_at.isoformat()
        }
        for t in tasks
    ]


# =============================================================================
# Main entry point for development
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
