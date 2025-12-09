# src/agent_lifecycle/state_machine.py
"""
Agent State Machine - Lifecycle state transitions
Integration 2B: INIT → DISCOVERY → READY state management
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field

logger = logging.getLogger("aaos.lifecycle.state_machine")


class AgentState(str, Enum):
    """Agent lifecycle states"""
    INIT = "init"              # Initial state after registration
    DISCOVERY = "discovery"    # Agent discovering capabilities/environment
    READY = "ready"            # Agent ready for work
    BUSY = "busy"              # Agent processing a task
    DEGRADED = "degraded"      # Agent operational but impaired
    FAILED = "failed"          # Agent has failed
    SHUTDOWN = "shutdown"      # Agent shutting down


# Valid state transitions
VALID_TRANSITIONS: Dict[AgentState, Set[AgentState]] = {
    AgentState.INIT: {AgentState.DISCOVERY, AgentState.FAILED, AgentState.SHUTDOWN},
    AgentState.DISCOVERY: {AgentState.READY, AgentState.FAILED, AgentState.SHUTDOWN},
    AgentState.READY: {AgentState.BUSY, AgentState.DEGRADED, AgentState.FAILED, AgentState.SHUTDOWN},
    AgentState.BUSY: {AgentState.READY, AgentState.DEGRADED, AgentState.FAILED, AgentState.SHUTDOWN},
    AgentState.DEGRADED: {AgentState.READY, AgentState.FAILED, AgentState.SHUTDOWN},
    AgentState.FAILED: {AgentState.INIT, AgentState.SHUTDOWN},  # Can restart from INIT
    AgentState.SHUTDOWN: set(),  # Terminal state
}


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted"""
    def __init__(self, agent_id: str, from_state: AgentState, to_state: AgentState):
        self.agent_id = agent_id
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Invalid transition for {agent_id}: {from_state.value} → {to_state.value}"
        )


@dataclass
class StateTransition:
    """Record of a state transition"""
    agent_id: str
    from_state: AgentState
    to_state: AgentState
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "metadata": self.metadata
        }


@dataclass
class AgentLifecycleState:
    """Complete lifecycle state for an agent"""
    agent_id: str
    agent_type: str
    state: AgentState = field(default=AgentState.INIT)
    previous_state: Optional[AgentState] = None
    state_entered_at: datetime = field(default_factory=datetime.utcnow)
    transition_history: List[StateTransition] = field(default_factory=list)
    failure_count: int = field(default=0)
    recovery_count: int = field(default=0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def can_transition_to(self, new_state: AgentState) -> bool:
        """Check if transition to new_state is valid"""
        return new_state in VALID_TRANSITIONS.get(self.state, set())

    def transition_to(self, new_state: AgentState, reason: Optional[str] = None) -> StateTransition:
        """
        Transition to a new state.

        Raises:
            InvalidTransitionError: If transition is not valid
        """
        if not self.can_transition_to(new_state):
            raise InvalidTransitionError(self.agent_id, self.state, new_state)

        # Record transition
        transition = StateTransition(
            agent_id=self.agent_id,
            from_state=self.state,
            to_state=new_state,
            reason=reason
        )
        self.transition_history.append(transition)

        # Update state
        self.previous_state = self.state
        self.state = new_state
        self.state_entered_at = datetime.utcnow()

        # Track failures and recoveries
        if new_state == AgentState.FAILED:
            self.failure_count += 1
        elif self.previous_state == AgentState.FAILED and new_state == AgentState.INIT:
            self.recovery_count += 1

        logger.info(f"Agent {self.agent_id}: {self.previous_state.value} → {new_state.value}"
                   + (f" ({reason})" if reason else ""))

        return transition

    def is_ready(self) -> bool:
        """Check if agent is in READY state"""
        return self.state == AgentState.READY

    def is_operational(self) -> bool:
        """Check if agent can accept work (READY or DEGRADED)"""
        return self.state in (AgentState.READY, AgentState.DEGRADED)

    def is_failed(self) -> bool:
        """Check if agent is in FAILED state"""
        return self.state == AgentState.FAILED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "state_entered_at": self.state_entered_at.isoformat(),
            "failure_count": self.failure_count,
            "recovery_count": self.recovery_count,
            "transition_count": len(self.transition_history),
            "metadata": self.metadata
        }


@dataclass
class StateMachineConfig:
    """Configuration for the state machine"""
    discovery_timeout: float = 5.0      # Max seconds for discovery phase
    auto_discovery: bool = True          # Auto-advance through INIT → DISCOVERY → READY
    discovery_delay: float = 0.1         # Simulated discovery time
    max_failures_before_shutdown: int = 5  # Failures before forced shutdown


class AgentStateMachine:
    """
    Manages lifecycle states for all agents.
    Handles state transitions, validation, and callbacks.
    """

    def __init__(self, config: Optional[StateMachineConfig] = None):
        self.config = config or StateMachineConfig()
        self.agents: Dict[str, AgentLifecycleState] = {}
        self._state_callbacks: List[Callable] = []
        self._initialized = False
        logger.info("AgentStateMachine created")

    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentLifecycleState:
        """Register an agent with the state machine (starts in INIT)"""
        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered in state machine")
            return self.agents[agent_id]

        state = AgentLifecycleState(
            agent_id=agent_id,
            agent_type=agent_type,
            metadata=metadata or {}
        )
        self.agents[agent_id] = state
        logger.info(f"Registered agent {agent_id} in state machine (INIT)")
        return state

    def get_agent_state(self, agent_id: str) -> Optional[AgentLifecycleState]:
        """Get the lifecycle state for an agent"""
        return self.agents.get(agent_id)

    def get_all_states(self) -> Dict[str, AgentLifecycleState]:
        """Get all agent states"""
        return self.agents

    async def transition_agent(
        self,
        agent_id: str,
        new_state: AgentState,
        reason: Optional[str] = None
    ) -> StateTransition:
        """
        Transition an agent to a new state.

        Raises:
            KeyError: If agent not found
            InvalidTransitionError: If transition is invalid
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found in state machine")

        agent = self.agents[agent_id]
        transition = agent.transition_to(new_state, reason)

        # Notify callbacks
        for callback in self._state_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id, transition)
                else:
                    callback(agent_id, transition)
            except Exception as e:
                logger.error(f"State callback error: {e}")

        return transition

    async def advance_to_ready(self, agent_id: str) -> bool:
        """
        Advance an agent through INIT → DISCOVERY → READY.

        Returns:
            True if agent reached READY state
        """
        if agent_id not in self.agents:
            logger.error(f"Cannot advance unknown agent: {agent_id}")
            return False

        agent = self.agents[agent_id]

        try:
            # INIT → DISCOVERY
            if agent.state == AgentState.INIT:
                await self.transition_agent(agent_id, AgentState.DISCOVERY, "Starting discovery")

                # Simulate discovery phase
                if self.config.auto_discovery:
                    await asyncio.sleep(self.config.discovery_delay)

            # DISCOVERY → READY
            if agent.state == AgentState.DISCOVERY:
                await self.transition_agent(agent_id, AgentState.READY, "Discovery complete")

            return agent.state == AgentState.READY

        except InvalidTransitionError as e:
            logger.error(f"Failed to advance agent: {e}")
            return False

    async def advance_all_to_ready(self) -> Dict[str, bool]:
        """
        Advance all agents to READY state.

        Returns:
            Dict mapping agent_id to success status
        """
        results = {}
        for agent_id in self.agents:
            results[agent_id] = await self.advance_to_ready(agent_id)
        return results

    async def simulate_failure(self, agent_id: str, reason: str = "Simulated failure") -> bool:
        """
        Simulate a failure for testing circuit breaker behavior.

        Returns:
            True if agent transitioned to FAILED
        """
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]

        try:
            # Can fail from most states
            if agent.can_transition_to(AgentState.FAILED):
                await self.transition_agent(agent_id, AgentState.FAILED, reason)
                return True
            return False
        except InvalidTransitionError:
            return False

    async def recover_agent(self, agent_id: str) -> bool:
        """
        Attempt to recover a failed agent.

        Returns:
            True if agent recovered to READY state
        """
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]

        if agent.state != AgentState.FAILED:
            logger.warning(f"Agent {agent_id} not in FAILED state, cannot recover")
            return False

        try:
            # FAILED → INIT → DISCOVERY → READY
            await self.transition_agent(agent_id, AgentState.INIT, "Recovery initiated")
            return await self.advance_to_ready(agent_id)
        except InvalidTransitionError as e:
            logger.error(f"Recovery failed: {e}")
            return False

    def on_state_change(self, callback: Callable):
        """Register a callback for state changes"""
        self._state_callbacks.append(callback)

    def get_agents_in_state(self, state: AgentState) -> List[str]:
        """Get list of agent IDs in a specific state"""
        return [
            agent_id for agent_id, agent in self.agents.items()
            if agent.state == state
        ]

    def get_ready_count(self) -> int:
        """Get count of agents in READY state"""
        return len(self.get_agents_in_state(AgentState.READY))

    def get_failed_count(self) -> int:
        """Get count of agents in FAILED state"""
        return len(self.get_agents_in_state(AgentState.FAILED))

    def all_agents_ready(self) -> bool:
        """Check if all agents are in READY state (GATE condition)"""
        if not self.agents:
            return False
        return all(agent.is_ready() for agent in self.agents.values())

    def get_operational_agents(self) -> List[str]:
        """Get list of agents that can accept work (READY or DEGRADED)"""
        return [
            agent_id for agent_id, agent in self.agents.items()
            if agent.is_operational()
        ]

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of state machine status"""
        state_counts = {}
        for state in AgentState:
            state_counts[state.value] = len(self.get_agents_in_state(state))

        total = len(self.agents)
        ready = state_counts.get(AgentState.READY.value, 0)

        return {
            "total_agents": total,
            "ready_agents": ready,
            "all_ready": self.all_agents_ready(),
            "operational_count": len(self.get_operational_agents()),
            "failed_count": self.get_failed_count(),
            "state_counts": state_counts,
            "agents": {
                agent_id: agent.to_dict()
                for agent_id, agent in self.agents.items()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
