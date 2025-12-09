# src/agent_lifecycle/orchestrator.py
"""
Lifecycle Orchestrator - Coordinates health monitoring, circuit breakers, state machine, and consensus
Integration 2A: Health monitoring and circuit breakers
Integration 2B: Agent lifecycle orchestration (state transitions)
Integration 2C: BFT Consensus (Byzantine Fault Tolerance)
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from .health_monitor import HealthMonitor, HealthMonitorConfig, HealthStatus, AgentHealth
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from .state_machine import (
    AgentStateMachine,
    AgentState,
    StateMachineConfig,
    StateTransition,
    InvalidTransitionError,
)

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from src.bft_consensus import ConsensusManager, ConsensusConfig

logger = logging.getLogger("aaos.lifecycle.orchestrator")


class LifecycleOrchestrator:
    """
    Central orchestrator for agent lifecycle management.
    Integrates health monitoring, circuit breakers, state machine, and BFT consensus.

    Integration 2A: Health monitoring and circuit breakers
    Integration 2B: State machine for lifecycle transitions (INIT → DISCOVERY → READY)
    Integration 2C: BFT Consensus (Byzantine Fault Tolerance)
    """

    def __init__(
        self,
        health_config: Optional[HealthMonitorConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        state_config: Optional[StateMachineConfig] = None,
        consensus_config: Optional["ConsensusConfig"] = None
    ):
        self.health_config = health_config or HealthMonitorConfig()
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.state_config = state_config or StateMachineConfig()
        self.consensus_config = consensus_config  # May be None initially

        self.health_monitor = HealthMonitor(self.health_config)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.state_machine = AgentStateMachine(self.state_config)
        self.consensus_manager: Optional["ConsensusManager"] = None  # Integration 2C

        self._initialized = False
        self._ready = False
        self._states_ready = False  # Integration 2B: All agents in READY state
        self._consensus_ready = False  # Integration 2C: Consensus operational
        logger.info("LifecycleOrchestrator created (Integration 2A + 2B + 2C)")

    async def initialize(self):
        """Initialize the orchestrator and all components"""
        if self._initialized:
            return

        logger.info("Initializing LifecycleOrchestrator...")

        # Initialize health monitor
        await self.health_monitor.initialize()

        # Register ensemble agents
        agents = self.health_monitor.register_ensemble_agents()

        # Create circuit breaker and state machine entry for each agent
        for agent in agents:
            # Circuit breaker (Integration 2A)
            self.circuit_breakers[agent.agent_id] = CircuitBreaker(
                agent_id=agent.agent_id,
                config=self.circuit_config
            )
            # State machine entry (Integration 2B) - starts in INIT
            self.state_machine.register_agent(
                agent_id=agent.agent_id,
                agent_type=agent.agent_type,
                metadata={"registered_at": datetime.utcnow().isoformat()}
            )

        # Register callbacks
        self.health_monitor.on_status_change(self._on_health_change)
        self.state_machine.on_state_change(self._on_state_change)

        self._initialized = True
        logger.info(f"LifecycleOrchestrator initialized with {len(agents)} agents (all in INIT state)")

    async def start(self):
        """Start all lifecycle services"""
        if not self._initialized:
            await self.initialize()

        # Start health monitoring
        await self.health_monitor.start_monitoring()

        # Wait for heartbeat stabilization (3s minimum)
        stabilized = await self.health_monitor.wait_for_stabilization()

        if stabilized:
            self._ready = True
            logger.info("LifecycleOrchestrator health monitoring ready")

            # Integration 2B: Advance all agents through INIT → DISCOVERY → READY
            logger.info("Advancing agents through lifecycle states...")
            results = await self.state_machine.advance_all_to_ready()

            # Check if all agents reached READY
            all_ready = all(results.values())
            self._states_ready = all_ready

            if all_ready:
                logger.info("All agents reached READY state (Integration 2B GATE PASSED)")

                # Integration 2C: Initialize and run consensus
                await self._initialize_consensus()
            else:
                failed_agents = [aid for aid, success in results.items() if not success]
                logger.warning(f"Some agents failed to reach READY: {failed_agents}")
        else:
            logger.warning("LifecycleOrchestrator started but not all agents stabilized")

    async def _initialize_consensus(self):
        """Initialize BFT Consensus (Integration 2C)"""
        # Lazy import to avoid circular dependency
        from src.bft_consensus import ConsensusManager, ConsensusConfig

        logger.info("Initializing BFT Consensus (Integration 2C)...")

        # Create consensus config if not provided
        if not self.consensus_config:
            self.consensus_config = ConsensusConfig(
                total_agents=9,
                max_byzantine_faults=2,
                leader_id="maestro"
            )

        # Create consensus manager with reference to this orchestrator
        self.consensus_manager = ConsensusManager(
            config=self.consensus_config,
            lifecycle_orchestrator=self
        )

        # Initialize with agent IDs
        await self.consensus_manager.initialize(self.get_agent_ids())

        # Run initial consensus round to verify system is operational
        logger.info("Running initial consensus round...")
        result = await self.consensus_manager.run_full_round(
            value="system_initialization",
            leader_id="maestro"
        )

        if result.success:
            self._consensus_ready = True
            logger.info(
                f"BFT Consensus operational (Integration 2C GATE PASSED): "
                f"prepare={result.prepare_votes}, commit={result.commit_votes}, "
                f"duration={result.duration_ms:.1f}ms"
            )
        else:
            logger.warning(
                f"Initial consensus failed at phase {result.phase_reached.value}"
            )

    async def stop(self):
        """Stop all lifecycle services"""
        await self.health_monitor.stop_monitoring()
        self._ready = False
        logger.info("LifecycleOrchestrator stopped")

    async def _on_health_change(self, agent_id: str, old_status: HealthStatus, new_status: HealthStatus):
        """
        Callback for health status changes.
        Integrates circuit breaker behavior and state machine with health status.
        """
        logger.info(f"Agent {agent_id} health: {old_status.value} -> {new_status.value}")

        if agent_id not in self.circuit_breakers:
            return

        breaker = self.circuit_breakers[agent_id]
        agent_state = self.state_machine.get_agent_state(agent_id)

        # If agent becomes unhealthy, record as failure and transition to DEGRADED or FAILED
        if new_status == HealthStatus.UNHEALTHY:
            await breaker.record_failure()

            # Integration 2B: Transition to DEGRADED if operational
            if agent_state and agent_state.is_operational():
                try:
                    await self.state_machine.transition_agent(
                        agent_id, AgentState.DEGRADED, "Health became UNHEALTHY"
                    )
                except InvalidTransitionError as e:
                    logger.warning(f"Could not transition to DEGRADED: {e}")

        # If agent recovers to healthy, record as success
        elif new_status == HealthStatus.HEALTHY and old_status in (HealthStatus.UNHEALTHY, HealthStatus.DEGRADED):
            await breaker.record_success()

            # Integration 2B: Transition back to READY if degraded
            if agent_state and agent_state.state == AgentState.DEGRADED:
                try:
                    await self.state_machine.transition_agent(
                        agent_id, AgentState.READY, "Health recovered to HEALTHY"
                    )
                except InvalidTransitionError as e:
                    logger.warning(f"Could not transition to READY: {e}")

    async def _on_state_change(self, agent_id: str, transition: StateTransition):
        """
        Callback for state machine changes (Integration 2B).
        Logs state transitions and updates readiness flag.
        """
        logger.info(
            f"State change: {agent_id} {transition.from_state.value} → {transition.to_state.value}"
            + (f" ({transition.reason})" if transition.reason else "")
        )

        # Update states_ready flag
        self._states_ready = self.state_machine.all_agents_ready()

    def record_heartbeat(self, agent_id: str) -> bool:
        """Record a heartbeat from an agent"""
        return self.health_monitor.record_heartbeat(agent_id)

    async def can_dispatch_to_agent(self, agent_id: str) -> bool:
        """
        Check if we can dispatch work to an agent.
        Considers health status, circuit breaker state, and lifecycle state.
        """
        if agent_id not in self.circuit_breakers:
            logger.warning(f"Unknown agent: {agent_id}")
            return False

        # Check circuit breaker first
        breaker = self.circuit_breakers[agent_id]
        if not await breaker.can_execute():
            logger.debug(f"Circuit breaker blocking dispatch to {agent_id}")
            return False

        # Check health status
        health = self.health_monitor.get_agent_health(agent_id)
        if health and health.status == HealthStatus.UNHEALTHY:
            logger.debug(f"Unhealthy agent blocking dispatch to {agent_id}")
            return False

        # Integration 2B: Check lifecycle state - must be operational (READY or DEGRADED)
        agent_state = self.state_machine.get_agent_state(agent_id)
        if agent_state and not agent_state.is_operational():
            logger.debug(f"Agent {agent_id} not operational (state: {agent_state.state.value})")
            return False

        return True

    async def record_agent_success(self, agent_id: str):
        """Record successful operation for an agent"""
        if agent_id in self.circuit_breakers:
            await self.circuit_breakers[agent_id].record_success()

    async def record_agent_failure(self, agent_id: str):
        """Record failed operation for an agent"""
        if agent_id in self.circuit_breakers:
            await self.circuit_breakers[agent_id].record_failure()

    def get_circuit_breaker(self, agent_id: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for a specific agent"""
        return self.circuit_breakers.get(agent_id)

    def get_all_circuit_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers"""
        return self.circuit_breakers

    def verify_circuit_breakers_closed(self) -> bool:
        """Verify all circuit breakers are in CLOSED state"""
        return all(
            breaker.state == CircuitState.CLOSED
            for breaker in self.circuit_breakers.values()
        )

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status for /health/agents endpoint.
        Includes health monitor, circuit breaker, state machine, and consensus status.
        """
        health_summary = self.health_monitor.get_health_summary()

        # Add circuit breaker status
        circuit_status = {}
        for agent_id, breaker in self.circuit_breakers.items():
            circuit_status[agent_id] = breaker.get_status()

        # Integration 2B: Add state machine status
        state_summary = self.state_machine.get_status_summary()

        # Integration 2C: Add consensus status
        consensus_status = None
        if self.consensus_manager:
            consensus_status = self.consensus_manager.get_status()

        # Calculate overall readiness
        all_circuits_closed = self.verify_circuit_breakers_closed()
        all_agents_healthy = health_summary["status_counts"].get("healthy", 0) == health_summary["total_agents"]
        all_agents_ready = self.state_machine.all_agents_ready()

        return {
            "ready": self._ready,
            "initialized": self._initialized,
            "states_ready": self._states_ready,  # Integration 2B
            "consensus_ready": self._consensus_ready,  # Integration 2C
            "health": health_summary,
            "circuit_breakers": circuit_status,
            "state_machine": state_summary,  # Integration 2B
            "consensus": consensus_status,  # Integration 2C
            "all_circuits_closed": all_circuits_closed,
            "all_agents_healthy": all_agents_healthy,
            "all_agents_ready": all_agents_ready,  # Integration 2B GATE
            "timestamp": datetime.utcnow().isoformat()
        }

    def readiness(self) -> bool:
        """
        GATE check: Returns True if orchestrator is fully ready.
        Integration 2A: All agents registered, heartbeats stabilized, circuits closed.
        Integration 2B: All agents in READY state.
        Integration 2C: BFT Consensus operational.
        """
        if not self._ready:
            return False

        if not self.health_monitor.readiness():
            return False

        if not self.verify_circuit_breakers_closed():
            return False

        # Integration 2B GATE: All agents must be in READY state
        if not self._states_ready:
            return False

        # Integration 2C GATE: Consensus must be operational
        if not self._consensus_ready:
            return False

        return True

    def get_registered_agent_count(self) -> int:
        """Get count of registered agents"""
        return len(self.health_monitor.agents)

    def get_agent_ids(self) -> List[str]:
        """Get list of all registered agent IDs"""
        return list(self.health_monitor.agents.keys())

    # =========================================================================
    # Integration 2B: State Machine Operations
    # =========================================================================

    async def simulate_agent_failure(self, agent_id: str, reason: str = "Simulated failure") -> bool:
        """
        Simulate a failure for an agent (Integration 2B).
        Transitions to FAILED state and opens circuit breaker.

        Returns:
            True if agent transitioned to FAILED
        """
        success = await self.state_machine.simulate_failure(agent_id, reason)
        if success:
            # Also record failure in circuit breaker
            if agent_id in self.circuit_breakers:
                await self.circuit_breakers[agent_id].record_failure()
            self._states_ready = self.state_machine.all_agents_ready()
        return success

    async def recover_agent(self, agent_id: str) -> bool:
        """
        Attempt to recover a failed agent (Integration 2B).
        Transitions through INIT → DISCOVERY → READY.

        Returns:
            True if agent recovered to READY state
        """
        success = await self.state_machine.recover_agent(agent_id)
        if success:
            # Record success in circuit breaker to help close it
            if agent_id in self.circuit_breakers:
                await self.circuit_breakers[agent_id].record_success()
            self._states_ready = self.state_machine.all_agents_ready()
        return success

    def get_operational_agents(self) -> List[str]:
        """Get list of agents that can accept work (READY or DEGRADED)"""
        return self.state_machine.get_operational_agents()

    def get_failed_agents(self) -> List[str]:
        """Get list of agents in FAILED state"""
        return self.state_machine.get_agents_in_state(AgentState.FAILED)

    def get_agent_lifecycle_state(self, agent_id: str):
        """Get the lifecycle state for a specific agent"""
        return self.state_machine.get_agent_state(agent_id)

    def can_degrade_gracefully(self) -> bool:
        """
        Check if system can continue with reduced agents (graceful degradation).
        Returns True if at least one agent is operational.
        """
        return len(self.get_operational_agents()) > 0

    def get_degradation_status(self) -> Dict[str, Any]:
        """
        Get graceful degradation status (Integration 2B).
        Shows operational capacity vs total agents.
        """
        total = len(self.state_machine.agents)
        operational = len(self.get_operational_agents())
        failed = len(self.get_failed_agents())

        return {
            "total_agents": total,
            "operational_agents": operational,
            "failed_agents": failed,
            "degradation_level": (total - operational) / total if total > 0 else 0,
            "can_operate": self.can_degrade_gracefully(),
            "operational_agent_ids": self.get_operational_agents(),
            "failed_agent_ids": self.get_failed_agents(),
        }

    # =========================================================================
    # Integration 2C: BFT Consensus Operations
    # =========================================================================

    def get_consensus_status(self) -> Optional[Dict[str, Any]]:
        """Get BFT consensus status (Integration 2C)"""
        if self.consensus_manager:
            return self.consensus_manager.get_status()
        return None

    def get_consensus_health(self) -> Dict[str, Any]:
        """Get consensus health for /health/consensus endpoint (Integration 2C)"""
        if self.consensus_manager:
            return self.consensus_manager.get_health_status()
        return {
            "ready": False,
            "phase": "not_initialized",
            "quorum_status": "not_initialized",
            "ready_agents": 0,
            "quorum_requirement": 5,
            "operational": False,
            "rounds_completed": 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def run_consensus_round(self, value: Any, leader_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Run a BFT consensus round (Integration 2C).

        Args:
            value: The value to reach consensus on
            leader_id: Optional leader (default: maestro)

        Returns:
            ConsensusResult as dict, or None if consensus not available
        """
        if not self.consensus_manager:
            logger.warning("Cannot run consensus: ConsensusManager not initialized")
            return None

        result = await self.consensus_manager.run_full_round(value, leader_id)
        return result.to_dict()

    def can_reach_consensus_quorum(self) -> bool:
        """Check if we have enough READY agents for consensus quorum"""
        if self.consensus_manager:
            return self.consensus_manager.can_reach_quorum()
        return False
