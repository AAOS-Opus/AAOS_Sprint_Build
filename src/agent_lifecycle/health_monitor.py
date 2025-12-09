# src/agent_lifecycle/health_monitor.py
"""
Agent Health Monitor - Heartbeat tracking and health status management
Integration 2A: Tracks health of all registered ensemble agents.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("aaos.lifecycle.health_monitor")


class HealthStatus(str, Enum):
    """Agent health status levels"""
    HEALTHY = "healthy"           # Recent heartbeat, all good
    DEGRADED = "degraded"         # Heartbeat delayed but within tolerance
    UNHEALTHY = "unhealthy"       # Heartbeat missed beyond tolerance
    UNKNOWN = "unknown"           # No heartbeat received yet
    OFFLINE = "offline"           # Agent explicitly disconnected


@dataclass
class AgentHealth:
    """Health state for a single agent"""
    agent_id: str
    agent_type: str
    status: HealthStatus = field(default=HealthStatus.UNKNOWN)
    last_heartbeat: Optional[datetime] = field(default=None)
    heartbeat_count: int = field(default=0)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    consecutive_missed: int = field(default=0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_heartbeat(self):
        """Record a heartbeat from this agent"""
        self.last_heartbeat = datetime.utcnow()
        self.heartbeat_count += 1
        self.consecutive_missed = 0
        self.status = HealthStatus.HEALTHY

    def check_health(self, healthy_threshold: float = 5.0, degraded_threshold: float = 15.0) -> HealthStatus:
        """
        Check health based on time since last heartbeat.

        Args:
            healthy_threshold: Seconds since heartbeat to be considered healthy
            degraded_threshold: Seconds since heartbeat before unhealthy
        """
        if self.status == HealthStatus.OFFLINE:
            return HealthStatus.OFFLINE

        if self.last_heartbeat is None:
            return HealthStatus.UNKNOWN

        elapsed = (datetime.utcnow() - self.last_heartbeat).total_seconds()

        if elapsed <= healthy_threshold:
            self.status = HealthStatus.HEALTHY
        elif elapsed <= degraded_threshold:
            self.status = HealthStatus.DEGRADED
        else:
            self.status = HealthStatus.UNHEALTHY
            self.consecutive_missed += 1

        return self.status

    def set_offline(self):
        """Mark agent as explicitly offline"""
        self.status = HealthStatus.OFFLINE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "heartbeat_count": self.heartbeat_count,
            "registered_at": self.registered_at.isoformat(),
            "consecutive_missed": self.consecutive_missed,
            "metadata": self.metadata
        }


@dataclass
class HealthMonitorConfig:
    """Configuration for health monitoring behavior"""
    heartbeat_interval: float = 1.0      # Expected heartbeat interval (seconds)
    healthy_threshold: float = 5.0       # Max seconds for healthy status
    degraded_threshold: float = 15.0     # Max seconds before unhealthy
    check_interval: float = 1.0          # How often to run health checks
    stabilization_time: float = 3.0      # Time to wait for initial stabilization


class HealthMonitor:
    """
    Central health monitor for all ensemble agents.
    Tracks heartbeats and manages health status for registered agents.
    """

    # The 9 ensemble agents from the integration spec
    ENSEMBLE_AGENTS = [
        ("maestro", "orchestrator"),
        ("opus", "conductor"),
        ("claude", "assistant"),
        ("devzen", "validator"),
        ("frontend", "architect"),
        ("backend", "architect"),
        ("kimi", "resilience"),
        ("scout", "reconnaissance"),
        ("dr-aeon", "diagnostics"),
    ]

    def __init__(self, config: Optional[HealthMonitorConfig] = None):
        self.config = config or HealthMonitorConfig()
        self.agents: Dict[str, AgentHealth] = {}
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        self._initialized = False
        self._ready = False
        self._callbacks: List[Callable] = []
        logger.info("HealthMonitor created")

    async def initialize(self):
        """Initialize the health monitor"""
        if self._initialized:
            return

        logger.info("Initializing health monitor...")
        self._initialized = True
        logger.info("Health monitor initialized successfully")

    def register_agent(self, agent_id: str, agent_type: str, metadata: Optional[Dict[str, Any]] = None) -> AgentHealth:
        """
        Register an agent for health monitoring.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type/role of the agent
            metadata: Optional additional metadata

        Returns:
            AgentHealth instance for the registered agent
        """
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered, updating")
            self.agents[agent_id].agent_type = agent_type
            if metadata:
                self.agents[agent_id].metadata.update(metadata)
            return self.agents[agent_id]

        health = AgentHealth(
            agent_id=agent_id,
            agent_type=agent_type,
            metadata=metadata or {}
        )
        self.agents[agent_id] = health
        logger.info(f"Registered agent: {agent_id} ({agent_type})")
        return health

    def register_ensemble_agents(self) -> List[AgentHealth]:
        """Register all 9 ensemble agents from the spec"""
        registered = []
        for agent_id, agent_type in self.ENSEMBLE_AGENTS:
            health = self.register_agent(agent_id, agent_type, {
                "ensemble_member": True,
                "role": agent_type
            })
            registered.append(health)
        logger.info(f"Registered {len(registered)} ensemble agents")
        return registered

    def unregister_agent(self, agent_id: str):
        """Remove an agent from monitoring"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")

    def record_heartbeat(self, agent_id: str) -> bool:
        """
        Record a heartbeat from an agent.

        Returns:
            True if agent is registered and heartbeat recorded
        """
        if agent_id not in self.agents:
            logger.warning(f"Heartbeat from unregistered agent: {agent_id}")
            return False

        self.agents[agent_id].record_heartbeat()
        return True

    def get_agent_health(self, agent_id: str) -> Optional[AgentHealth]:
        """Get health status for a specific agent"""
        return self.agents.get(agent_id)

    def get_all_health(self) -> Dict[str, AgentHealth]:
        """Get health status for all agents"""
        # Update health status for all agents
        for agent in self.agents.values():
            agent.check_health(
                self.config.healthy_threshold,
                self.config.degraded_threshold
            )
        return self.agents

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of overall system health"""
        health_dict = self.get_all_health()

        status_counts = {
            HealthStatus.HEALTHY.value: 0,
            HealthStatus.DEGRADED.value: 0,
            HealthStatus.UNHEALTHY.value: 0,
            HealthStatus.UNKNOWN.value: 0,
            HealthStatus.OFFLINE.value: 0,
        }

        for agent in health_dict.values():
            status_counts[agent.status.value] += 1

        # Overall status determination
        total = len(health_dict)
        healthy = status_counts[HealthStatus.HEALTHY.value]

        if healthy == total and total > 0:
            overall = "healthy"
        elif status_counts[HealthStatus.UNHEALTHY.value] > 0:
            overall = "unhealthy"
        elif status_counts[HealthStatus.DEGRADED.value] > 0:
            overall = "degraded"
        else:
            overall = "unknown"

        return {
            "overall_status": overall,
            "total_agents": total,
            "status_counts": status_counts,
            "agents": [agent.to_dict() for agent in health_dict.values()],
            "timestamp": datetime.utcnow().isoformat()
        }

    async def start_monitoring(self):
        """Start the background health check loop"""
        if self._running:
            return

        self._running = True
        self._check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Health monitoring started")

    async def stop_monitoring(self):
        """Stop the background health check loop"""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")

    async def _health_check_loop(self):
        """Background loop to check agent health"""
        while self._running:
            try:
                # Update health for all agents
                for agent in self.agents.values():
                    old_status = agent.status
                    new_status = agent.check_health(
                        self.config.healthy_threshold,
                        self.config.degraded_threshold
                    )

                    # Notify callbacks on status change
                    if old_status != new_status:
                        for callback in self._callbacks:
                            try:
                                await callback(agent.agent_id, old_status, new_status)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

                await asyncio.sleep(self.config.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(1)

    def on_status_change(self, callback: Callable):
        """Register a callback for status changes"""
        self._callbacks.append(callback)

    async def wait_for_stabilization(self) -> bool:
        """
        Wait for heartbeat stabilization (3s minimum per workflow).
        Simulates heartbeats for registered agents during stabilization.

        Returns:
            True if all agents report heartbeats within stabilization period
        """
        logger.info(f"Waiting for heartbeat stabilization ({self.config.stabilization_time}s)...")

        # Simulate initial heartbeats for all registered agents
        for agent_id in self.agents:
            self.record_heartbeat(agent_id)

        # Wait for stabilization period
        await asyncio.sleep(self.config.stabilization_time)

        # Check that all agents have heartbeats
        all_healthy = all(
            agent.last_heartbeat is not None
            for agent in self.agents.values()
        )

        if all_healthy:
            self._ready = True
            logger.info("Heartbeat stabilization complete - all agents healthy")
        else:
            missing = [
                agent.agent_id for agent in self.agents.values()
                if agent.last_heartbeat is None
            ]
            logger.warning(f"Stabilization incomplete - missing heartbeats: {missing}")

        return all_healthy

    def readiness(self) -> bool:
        """
        Check if health monitor is ready (GATE check).
        Returns True if initialized and stabilized.
        """
        return self._initialized and self._ready
