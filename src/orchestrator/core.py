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

# Import authentication module (Security Fix #1)
from src.auth import verify_api_key, verify_websocket_auth, is_auth_enabled

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
# Task Circuit Breaker (Fix #4)
# =============================================================================

# Circuit Breaker configuration from environment
CIRCUIT_BREAKER_ENABLED = os.environ.get("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
CIRCUIT_MIN_AGENTS = int(os.environ.get("CIRCUIT_MIN_AGENTS", "1"))
CIRCUIT_MAX_QUEUE_SIZE = int(os.environ.get("CIRCUIT_MAX_QUEUE_SIZE", "1000"))
CIRCUIT_RECOVERY_TIMEOUT = int(os.environ.get("CIRCUIT_RECOVERY_TIMEOUT", "30"))
CIRCUIT_TEST_TASK_TIMEOUT = int(os.environ.get("CIRCUIT_TEST_TASK_TIMEOUT", "60"))


class TaskCircuitState(str, Enum):
    """Circuit breaker states for task submission"""
    CLOSED = "closed"      # Normal operation, accept tasks
    OPEN = "open"          # Reject tasks with 503
    HALF_OPEN = "half_open"  # Testing recovery, allow limited tasks


class TaskCircuitBreaker:
    """
    Circuit breaker for task submission (Fix #4).

    Prevents task queue buildup when no agents are available to process them.

    States:
    - CLOSED: Normal operation, accept all tasks
    - OPEN: No agents OR queue too large, reject with 503
    - HALF_OPEN: Testing recovery, allow one test task through

    Transitions:
    - CLOSED -> OPEN: When active_agents < min_agents OR queue_size >= max_queue
    - OPEN -> HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN -> CLOSED: If test task picked up within test_task_timeout
    - HALF_OPEN -> OPEN: If test task not picked up in time
    """

    def __init__(
        self,
        min_agents: int = CIRCUIT_MIN_AGENTS,
        max_queue_size: int = CIRCUIT_MAX_QUEUE_SIZE,
        recovery_timeout: int = CIRCUIT_RECOVERY_TIMEOUT,
        test_task_timeout: int = CIRCUIT_TEST_TASK_TIMEOUT,
        enabled: bool = CIRCUIT_BREAKER_ENABLED
    ):
        self.min_agents = min_agents
        self.max_queue_size = max_queue_size
        self.recovery_timeout = recovery_timeout
        self.test_task_timeout = test_task_timeout
        self.enabled = enabled

        self._state = TaskCircuitState.CLOSED
        self._opened_at: Optional[datetime] = None
        self._test_task_id: Optional[str] = None
        self._test_task_submitted_at: Optional[datetime] = None
        self._last_state_change: datetime = datetime.utcnow()
        self._open_reason: Optional[str] = None

        # Metrics
        self._tasks_rejected: int = 0
        self._state_changes: int = 0

    @property
    def state(self) -> TaskCircuitState:
        """Get current circuit state, checking for automatic transitions"""
        if not self.enabled:
            return TaskCircuitState.CLOSED

        # Check for OPEN -> HALF_OPEN transition
        if self._state == TaskCircuitState.OPEN:
            if self._opened_at and (datetime.utcnow() - self._opened_at).total_seconds() >= self.recovery_timeout:
                self._transition_to(TaskCircuitState.HALF_OPEN, "recovery_timeout_elapsed")

        # Check for HALF_OPEN -> OPEN transition (test task timeout)
        if self._state == TaskCircuitState.HALF_OPEN:
            if self._test_task_submitted_at:
                elapsed = (datetime.utcnow() - self._test_task_submitted_at).total_seconds()
                if elapsed >= self.test_task_timeout:
                    self._transition_to(TaskCircuitState.OPEN, "test_task_timeout")
                    self._test_task_id = None
                    self._test_task_submitted_at = None

        return self._state

    def _transition_to(self, new_state: TaskCircuitState, reason: str):
        """Transition to a new state with logging"""
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.utcnow()
        self._state_changes += 1

        if new_state == TaskCircuitState.OPEN:
            self._opened_at = datetime.utcnow()
            self._open_reason = reason
        elif new_state == TaskCircuitState.HALF_OPEN:
            self._test_task_id = None
            self._test_task_submitted_at = None

        logger.info(
            f"Circuit breaker state change: {old_state.value} -> {new_state.value} | "
            f"reason={reason}"
        )

    def check_conditions(self, active_agents: int, queue_size: int) -> bool:
        """
        Check circuit breaker conditions and update state.

        Returns True if tasks should be allowed, False if they should be rejected.
        """
        if not self.enabled:
            return True

        # Check current state first (triggers automatic transitions)
        current_state = self.state

        # Evaluate conditions
        agents_ok = active_agents >= self.min_agents
        queue_ok = queue_size < self.max_queue_size

        if current_state == TaskCircuitState.CLOSED:
            # Check if we need to open the circuit
            if not agents_ok:
                self._transition_to(TaskCircuitState.OPEN, f"insufficient_agents ({active_agents}/{self.min_agents})")
                return False
            if not queue_ok:
                self._transition_to(TaskCircuitState.OPEN, f"queue_overflow ({queue_size}/{self.max_queue_size})")
                return False
            return True

        elif current_state == TaskCircuitState.OPEN:
            # Already open, reject
            return False

        elif current_state == TaskCircuitState.HALF_OPEN:
            # Allow one test task if none submitted yet
            if self._test_task_id is None:
                return True  # Will be marked as test task in allow_task()
            return False  # Already have a test task pending

        return False

    def allow_task(self, task_id: str) -> bool:
        """
        Check if a task should be allowed through.

        For HALF_OPEN state, marks the first task as a test task.
        Returns True if allowed, False if rejected.
        """
        if not self.enabled:
            return True

        current_state = self.state

        if current_state == TaskCircuitState.CLOSED:
            return True

        elif current_state == TaskCircuitState.OPEN:
            self._tasks_rejected += 1
            return False

        elif current_state == TaskCircuitState.HALF_OPEN:
            if self._test_task_id is None:
                # This is our test task
                self._test_task_id = task_id
                self._test_task_submitted_at = datetime.utcnow()
                logger.info(f"Circuit breaker HALF_OPEN: allowing test task {task_id}")
                return True
            else:
                # Already have a test task, reject others
                self._tasks_rejected += 1
                return False

        return False

    def on_task_assigned(self, task_id: str):
        """
        Called when a task is picked up by an agent.

        If this is our test task and we're in HALF_OPEN, transition to CLOSED.
        """
        if not self.enabled:
            return

        if self._state == TaskCircuitState.HALF_OPEN and self._test_task_id == task_id:
            # Test task was picked up successfully!
            logger.info(f"Circuit breaker test task {task_id} picked up - closing circuit")
            self._transition_to(TaskCircuitState.CLOSED, "test_task_success")
            self._test_task_id = None
            self._test_task_submitted_at = None

    def force_check(self, active_agents: int, queue_size: int):
        """
        Force a condition check (useful for background monitoring).

        This can close an OPEN circuit if conditions have improved,
        or open a CLOSED circuit if conditions have degraded.
        """
        if not self.enabled:
            return

        agents_ok = active_agents >= self.min_agents
        queue_ok = queue_size < self.max_queue_size

        if self._state == TaskCircuitState.CLOSED:
            if not agents_ok:
                self._transition_to(TaskCircuitState.OPEN, f"insufficient_agents ({active_agents}/{self.min_agents})")
            elif not queue_ok:
                self._transition_to(TaskCircuitState.OPEN, f"queue_overflow ({queue_size}/{self.max_queue_size})")

        elif self._state == TaskCircuitState.OPEN:
            # If conditions are now OK and we've waited long enough, go to HALF_OPEN
            if agents_ok and queue_ok and self._opened_at:
                elapsed = (datetime.utcnow() - self._opened_at).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._transition_to(TaskCircuitState.HALF_OPEN, "conditions_improved")

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status for health endpoint"""
        return {
            "enabled": self.enabled,
            "state": self.state.value,
            "open_reason": self._open_reason if self._state == TaskCircuitState.OPEN else None,
            "opened_at": self._opened_at.isoformat() if self._opened_at else None,
            "last_state_change": self._last_state_change.isoformat(),
            "test_task_id": self._test_task_id,
            "test_task_submitted_at": self._test_task_submitted_at.isoformat() if self._test_task_submitted_at else None,
            "tasks_rejected": self._tasks_rejected,
            "state_changes": self._state_changes,
            "config": {
                "min_agents": self.min_agents,
                "max_queue_size": self.max_queue_size,
                "recovery_timeout": self.recovery_timeout,
                "test_task_timeout": self.test_task_timeout
            }
        }

    def get_retry_after(self) -> int:
        """Get seconds until circuit might allow tasks again"""
        if self._state == TaskCircuitState.OPEN and self._opened_at:
            elapsed = (datetime.utcnow() - self._opened_at).total_seconds()
            remaining = self.recovery_timeout - elapsed
            return max(1, int(remaining))
        return self.recovery_timeout


# Global task circuit breaker instance
task_circuit_breaker = TaskCircuitBreaker()


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
    """Basic health check endpoint with circuit breaker status (Fix #4)"""
    # Get basic system metrics
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        queue_size = redis_client.llen("task_queue")
        redis_healthy = True
    except Exception:
        queue_size = -1
        redis_healthy = False

    active_agents = manager.get_active_agent_count()

    # Get circuit breaker status
    circuit_status = task_circuit_breaker.get_status()

    # Determine overall health
    overall_status = "healthy"
    if circuit_status["state"] == "open":
        overall_status = "degraded"
    elif not redis_healthy:
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "platform": platform.system(),
        "timestamp": datetime.utcnow().isoformat(),
        "active_agents": active_agents,
        "queue_size": queue_size,
        "redis_healthy": redis_healthy,
        "circuit_breaker": circuit_status
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
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Create task with two-phase commit: Redis-first, then DB.

    Fix #2: Two-Phase Commit Protocol
    ----------------------------------
    Phase 1: Push task_id to Redis queue FIRST (reversible)
    Phase 2: Commit to database
    Rollback: If DB fails, remove task_id from Redis queue

    This prevents orphaned tasks in DB that never get queued.

    Fix #4: Circuit Breaker
    -----------------------
    Checks circuit breaker state before accepting task.
    Returns 503 if circuit is OPEN (no agents or queue overflow).
    """
    task_id = uuid.uuid4()
    task_id_str = str(task_id)
    redis_queued = False

    # Verify Redis health before queue operation
    redis_client = verify_redis_health()

    # =================================================================
    # FIX #4: Circuit Breaker Check
    # =================================================================
    if task_circuit_breaker.enabled:
        # Get current system state
        active_agents = manager.get_active_agent_count()
        queue_size = redis_client.llen("task_queue")

        # Check conditions and update circuit state
        if not task_circuit_breaker.check_conditions(active_agents, queue_size):
            # Circuit is open - reject task
            retry_after = task_circuit_breaker.get_retry_after()
            circuit_status = task_circuit_breaker.get_status()

            logger.warning(
                f"Task rejected by circuit breaker | "
                f"state={circuit_status['state']} | "
                f"reason={circuit_status.get('open_reason', 'unknown')} | "
                f"active_agents={active_agents} | queue_size={queue_size}"
            )

            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Service unavailable",
                    "reason": "circuit_open",
                    "circuit_state": circuit_status['state'],
                    "open_reason": circuit_status.get('open_reason'),
                    "retry_after": retry_after,
                    "active_agents": active_agents,
                    "queue_size": queue_size
                }
            )

        # For HALF_OPEN state, mark this as the test task
        if not task_circuit_breaker.allow_task(task_id_str):
            # This shouldn't happen if check_conditions passed, but safety check
            retry_after = task_circuit_breaker.get_retry_after()
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Service unavailable",
                    "reason": "circuit_open",
                    "retry_after": retry_after
                }
            )

    try:
        # =================================================================
        # PHASE 1: Queue to Redis FIRST (this is reversible)
        # =================================================================
        try:
            redis_client.lpush("task_queue", task_id_str)
            redis_queued = True
            logger.debug(f"Phase 1 complete: Task {task_id_str} queued to Redis")
        except Exception as redis_error:
            logger.error(f"Phase 1 failed - Redis queue error for task {task_id_str}: {redis_error}")
            raise HTTPException(status_code=503, detail="Task queue unavailable")

        # =================================================================
        # PHASE 2: Commit to database
        # =================================================================
        try:
            # Convert metadata to JSON string for SQLite
            metadata_json = json.dumps(task.metadata) if task.metadata else "{}"

            # Create task record
            db_task = Task(
                task_id=task_id_str,
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
            logger.debug(f"Phase 2 complete: Task {task_id_str} committed to DB")

        except Exception as db_error:
            # =================================================================
            # ROLLBACK: DB failed, remove from Redis queue
            # =================================================================
            logger.error(f"Phase 2 failed - DB commit error for task {task_id_str}: {db_error}")

            if redis_queued:
                try:
                    # Remove the task_id from Redis queue (LREM removes by value)
                    removed_count = redis_client.lrem("task_queue", 1, task_id_str)
                    if removed_count > 0:
                        logger.info(f"Rollback successful: Removed task {task_id_str} from Redis queue")
                    else:
                        logger.warning(f"Rollback warning: Task {task_id_str} not found in Redis queue (may have been consumed)")
                except Exception as rollback_error:
                    # CRITICAL: Both DB and rollback failed - manual intervention required
                    logger.critical(
                        f"ROLLBACK FAILED - MANUAL INTERVENTION REQUIRED | "
                        f"task_id={task_id_str} is in Redis queue but NOT in DB | "
                        f"DB error: {db_error} | Rollback error: {rollback_error}"
                    )

            db.rollback()
            raise HTTPException(status_code=500, detail=f"Task creation failed: {str(db_error)}")

        # =================================================================
        # SUCCESS: Both phases complete
        # =================================================================
        log_context = {
            "task_id": task_id_str,
            "type": task.task_type,
            "priority": task.priority,
            "description_preview": task.description[:50],
            "two_phase_commit": "success"
        }
        logger.info(f"Task created successfully (2PC) | {json.dumps(log_context)}")

        return TaskResponse(
            task_id=task_id_str,
            status="pending",
            message="Task queued successfully",
            queued_at=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        # Unexpected error - attempt rollback if Redis was queued
        if redis_queued:
            try:
                redis_client.lrem("task_queue", 1, task_id_str)
                logger.info(f"Cleanup: Removed task {task_id_str} from Redis after unexpected error")
            except Exception as cleanup_error:
                logger.critical(
                    f"CLEANUP FAILED - MANUAL INTERVENTION REQUIRED | "
                    f"task_id={task_id_str} may be orphaned in Redis | "
                    f"Original error: {e} | Cleanup error: {cleanup_error}"
                )
        db.rollback()
        logger.error(f"Task creation failed: {str(e)}", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Task creation failed: {str(e)}")


# =============================================================================
# GET /tasks/{task_id} Endpoint
# =============================================================================

@app.get("/tasks/{task_id}", response_model=TaskDetail)
async def get_task(
    task_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
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

                    # Update task and agent status (Fix #3: set assigned_at for timeout tracking)
                    task.status = "assigned"
                    task.assigned_at = datetime.utcnow()  # Fix #3: Track assignment time
                    task.updated_at = datetime.utcnow()
                    db.commit()

                    manager.set_agent_status(agent_id, "busy")
                    manager.assign_task_to_agent(agent_id, task_id)

                    # Fix #4: Notify circuit breaker that task was assigned
                    task_circuit_breaker.on_task_assigned(task_id)

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
# Background Reconciliation Job (Fix #2)
# =============================================================================

# Reconciliation configuration
RECONCILIATION_INTERVAL_SECONDS = 300  # 5 minutes
ORPHAN_THRESHOLD_MINUTES = 10  # Tasks pending > 10 minutes without being in Redis

reconciliation_running = False

# =============================================================================
# Task Timeout + Retry Configuration (Fix #3)
# =============================================================================

TASK_TIMEOUT_SECONDS = int(os.environ.get("TASK_TIMEOUT_SECONDS", "300"))  # 5 minutes
TASK_MAX_RETRIES = int(os.environ.get("TASK_MAX_RETRIES", "3"))
TASK_TIMEOUT_CHECK_INTERVAL = int(os.environ.get("TASK_TIMEOUT_CHECK_INTERVAL", "30"))  # Check every 30s

task_timeout_running = False


async def task_reconciliation_loop():
    """
    Background job that reconciles orphaned tasks.

    Fix #2: Reconciliation Job
    --------------------------
    Runs every 5 minutes to find tasks that:
    - Are in DB with status "pending"
    - Are older than 10 minutes
    - Are NOT in the Redis queue

    These orphaned tasks are either re-queued or marked as failed.
    """
    global reconciliation_running
    reconciliation_running = True

    logger.info(
        f"Reconciliation job started | interval={RECONCILIATION_INTERVAL_SECONDS}s | "
        f"orphan_threshold={ORPHAN_THRESHOLD_MINUTES}min"
    )

    while reconciliation_running:
        try:
            await asyncio.sleep(RECONCILIATION_INTERVAL_SECONDS)

            if not reconciliation_running:
                break

            await run_reconciliation()

        except Exception as e:
            logger.error(f"Reconciliation loop error: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying on error


async def run_reconciliation():
    """
    Execute a single reconciliation pass.
    Can be called manually or by the background loop.
    """
    from datetime import timedelta

    logger.info("Running task reconciliation...")
    reconciled_count = 0
    failed_count = 0

    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        # Get all task IDs currently in Redis queue
        redis_queue_ids = set(redis_client.lrange("task_queue", 0, -1))
        logger.debug(f"Redis queue contains {len(redis_queue_ids)} tasks")

        # Calculate the threshold timestamp
        threshold_time = datetime.utcnow() - timedelta(minutes=ORPHAN_THRESHOLD_MINUTES)

        # Find orphaned tasks in DB
        db = SessionLocal()
        try:
            orphaned_tasks = db.query(Task).filter(
                Task.status == "pending",
                Task.created_at < threshold_time
            ).all()

            for task in orphaned_tasks:
                if task.task_id not in redis_queue_ids:
                    # This task is orphaned - in DB but not in Redis
                    logger.warning(
                        f"Orphaned task detected: {task.task_id} | "
                        f"created_at={task.created_at.isoformat()} | "
                        f"age_minutes={(datetime.utcnow() - task.created_at).total_seconds() / 60:.1f}"
                    )

                    # Try to re-queue the task
                    try:
                        redis_client.lpush("task_queue", task.task_id)
                        task.updated_at = datetime.utcnow()

                        # Add reconciliation metadata
                        existing_meta = json.loads(task.metadata_json) if task.metadata_json else {}
                        existing_meta["reconciled_at"] = datetime.utcnow().isoformat()
                        existing_meta["reconciliation_reason"] = "orphaned_task_recovery"
                        task.metadata_json = json.dumps(existing_meta)

                        db.commit()
                        reconciled_count += 1
                        logger.info(f"Reconciled task {task.task_id} - re-queued to Redis")

                    except Exception as requeue_error:
                        # Failed to re-queue, mark as failed
                        logger.error(f"Failed to re-queue task {task.task_id}: {requeue_error}")
                        try:
                            task.status = "failed"
                            task.updated_at = datetime.utcnow()

                            existing_meta = json.loads(task.metadata_json) if task.metadata_json else {}
                            existing_meta["failed_at"] = datetime.utcnow().isoformat()
                            existing_meta["failure_reason"] = f"reconciliation_requeue_failed: {str(requeue_error)}"
                            task.metadata_json = json.dumps(existing_meta)

                            db.commit()
                            failed_count += 1
                            logger.warning(f"Marked task {task.task_id} as failed due to reconciliation failure")
                        except Exception as mark_error:
                            logger.critical(
                                f"RECONCILIATION CRITICAL - Cannot mark task {task.task_id} as failed: {mark_error}"
                            )
                            db.rollback()

        finally:
            db.close()

        # Log reconciliation summary
        if reconciled_count > 0 or failed_count > 0:
            logger.info(
                f"Reconciliation complete | re-queued={reconciled_count} | failed={failed_count}"
            )
        else:
            logger.debug("Reconciliation complete | no orphaned tasks found")

        return {"reconciled": reconciled_count, "failed": failed_count}

    except Exception as e:
        logger.error(f"Reconciliation error: {e}")
        return {"error": str(e)}


@app.on_event("startup")
async def start_reconciliation():
    """Start the background reconciliation loop"""
    asyncio.create_task(task_reconciliation_loop())


@app.on_event("shutdown")
async def stop_reconciliation():
    """Stop the background reconciliation loop"""
    global reconciliation_running
    reconciliation_running = False
    logger.info("Reconciliation job stopped")


# =============================================================================
# Manual Reconciliation Endpoint (for testing/operations)
# =============================================================================

@app.post("/admin/reconcile", include_in_schema=True)
async def trigger_reconciliation(api_key: str = Depends(verify_api_key)):
    """
    Manually trigger task reconciliation.
    Useful for testing and operational recovery.

    Returns count of reconciled and failed tasks.
    """
    logger.info("Manual reconciliation triggered")
    result = await run_reconciliation()
    return {
        "status": "completed",
        "result": result,
        "timestamp": datetime.utcnow().isoformat()
    }


# =============================================================================
# Task Timeout Detection + Retry Loop (Fix #3)
# =============================================================================

def calculate_backoff_seconds(retry_count: int) -> int:
    """
    Calculate exponential backoff delay.
    backoff_seconds = min(60, 2 ** retry_count)
    """
    return min(60, 2 ** retry_count)


async def task_timeout_detection_loop():
    """
    Background loop that detects stale tasks and retries them.

    Fix #3: Task Timeout + Retry with Idempotency
    ---------------------------------------------
    Runs every TASK_TIMEOUT_CHECK_INTERVAL seconds (default 30s) to find tasks that:
    - Have status "assigned"
    - assigned_at > TASK_TIMEOUT_SECONDS ago (default 300s / 5 minutes)
    - Are NOT in Redis "processing_tasks" set (idempotency check)

    For each stale task:
    - If retry_count < max_retries: increment retry_count, calculate backoff, requeue
    - If retry_count >= max_retries: mark as "failed" with reason "timeout_max_retries"
    """
    global task_timeout_running
    task_timeout_running = True

    logger.info(
        f"Task timeout detection started | "
        f"interval={TASK_TIMEOUT_CHECK_INTERVAL}s | "
        f"timeout={TASK_TIMEOUT_SECONDS}s | "
        f"max_retries={TASK_MAX_RETRIES}"
    )

    while task_timeout_running:
        try:
            await asyncio.sleep(TASK_TIMEOUT_CHECK_INTERVAL)

            if not task_timeout_running:
                break

            await run_timeout_detection()

        except Exception as e:
            logger.error(f"Timeout detection loop error: {e}")
            await asyncio.sleep(10)  # Wait before retrying on error


async def run_timeout_detection():
    """
    Execute a single timeout detection pass.
    Can be called manually or by the background loop.
    """
    from datetime import timedelta

    logger.debug("Running task timeout detection...")
    retried_count = 0
    failed_count = 0
    skipped_idempotent = 0

    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        # Calculate the timeout threshold
        timeout_threshold = datetime.utcnow() - timedelta(seconds=TASK_TIMEOUT_SECONDS)

        # Get tasks currently being processed (idempotency check)
        processing_tasks = redis_client.smembers("processing_tasks")

        db = SessionLocal()
        try:
            # Find stale assigned tasks
            stale_tasks = db.query(Task).filter(
                Task.status == "assigned",
                Task.assigned_at.isnot(None),
                Task.assigned_at < timeout_threshold
            ).all()

            for task in stale_tasks:
                # Idempotency check: skip if task is actively being processed
                if task.task_id in processing_tasks:
                    logger.debug(f"Skipping task {task.task_id} - in processing_tasks set (idempotent)")
                    skipped_idempotent += 1
                    continue

                # Check if backoff period has passed
                if task.retry_after and task.retry_after > datetime.utcnow():
                    logger.debug(
                        f"Skipping task {task.task_id} - backoff period not expired "
                        f"(retry_after={task.retry_after.isoformat()})"
                    )
                    continue

                # Get effective max_retries (use task-specific or global default)
                effective_max_retries = task.max_retries if task.max_retries is not None else TASK_MAX_RETRIES
                current_retry_count = task.retry_count or 0

                if current_retry_count < effective_max_retries:
                    # Retry the task
                    try:
                        # Increment retry count
                        task.retry_count = current_retry_count + 1

                        # Calculate exponential backoff
                        backoff_seconds = calculate_backoff_seconds(task.retry_count)
                        task.retry_after = datetime.utcnow() + timedelta(seconds=backoff_seconds)

                        # Reset status to pending for requeue
                        task.status = "pending"
                        task.assigned_at = None
                        task.updated_at = datetime.utcnow()

                        # Add retry metadata
                        existing_meta = json.loads(task.metadata_json) if task.metadata_json else {}
                        if "retry_history" not in existing_meta:
                            existing_meta["retry_history"] = []
                        existing_meta["retry_history"].append({
                            "retry_number": task.retry_count,
                            "reason": "timeout",
                            "backoff_seconds": backoff_seconds,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        task.metadata_json = json.dumps(existing_meta)

                        db.commit()

                        # Requeue to Redis
                        redis_client.lpush("task_queue", task.task_id)

                        retried_count += 1
                        logger.info(
                            f"Task {task.task_id} timed out - requeued | "
                            f"retry={task.retry_count}/{effective_max_retries} | "
                            f"backoff={backoff_seconds}s"
                        )

                    except Exception as retry_error:
                        logger.error(f"Failed to retry task {task.task_id}: {retry_error}")
                        db.rollback()

                else:
                    # Max retries exceeded - mark as failed
                    try:
                        task.status = "failed"
                        task.updated_at = datetime.utcnow()

                        existing_meta = json.loads(task.metadata_json) if task.metadata_json else {}
                        existing_meta["failed_at"] = datetime.utcnow().isoformat()
                        existing_meta["failure_reason"] = "timeout_max_retries"
                        existing_meta["final_retry_count"] = current_retry_count
                        task.metadata_json = json.dumps(existing_meta)

                        db.commit()

                        failed_count += 1
                        logger.warning(
                            f"Task {task.task_id} failed - max retries exceeded | "
                            f"retries={current_retry_count}/{effective_max_retries}"
                        )

                    except Exception as fail_error:
                        logger.error(f"Failed to mark task {task.task_id} as failed: {fail_error}")
                        db.rollback()

        finally:
            db.close()

        # Log summary if any tasks were processed
        if retried_count > 0 or failed_count > 0 or skipped_idempotent > 0:
            logger.info(
                f"Timeout detection complete | retried={retried_count} | "
                f"failed={failed_count} | skipped_idempotent={skipped_idempotent}"
            )
        else:
            logger.debug("Timeout detection complete | no stale tasks found")

        return {
            "retried": retried_count,
            "failed": failed_count,
            "skipped_idempotent": skipped_idempotent
        }

    except Exception as e:
        logger.error(f"Timeout detection error: {e}")
        return {"error": str(e)}


@app.on_event("startup")
async def start_timeout_detection():
    """Start the background task timeout detection loop"""
    asyncio.create_task(task_timeout_detection_loop())


@app.on_event("shutdown")
async def stop_timeout_detection():
    """Stop the background task timeout detection loop"""
    global task_timeout_running
    task_timeout_running = False
    logger.info("Task timeout detection stopped")


# =============================================================================
# Manual Timeout Detection Endpoint (for testing/operations)
# =============================================================================

@app.post("/admin/timeout-check", include_in_schema=True)
async def trigger_timeout_check(api_key: str = Depends(verify_api_key)):
    """
    Manually trigger task timeout detection.
    Useful for testing and operational recovery.

    Returns count of retried, failed, and skipped tasks.
    """
    logger.info("Manual timeout check triggered")
    result = await run_timeout_detection()
    return {
        "status": "completed",
        "result": result,
        "timestamp": datetime.utcnow().isoformat()
    }


# =============================================================================
# WebSocket Endpoint for Agents
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for agent communication
    Protocol: AUTH -> register -> registration_ack -> heartbeat/task_assignment/task_complete

    Security: Validates X-API-Key header BEFORE accepting connection
    """
    agent_id = None

    try:
        # SECURITY FIX #1: Verify API key BEFORE accepting connection
        if not await verify_websocket_auth(websocket):
            await websocket.close(code=1008, reason="Authentication required")
            logger.warning(
                f"WebSocket connection rejected - authentication failed "
                f"(client: {websocket.client.host if websocket.client else 'unknown'})"
            )
            return

        # Accept connection after authentication
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
async def list_tasks(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
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
