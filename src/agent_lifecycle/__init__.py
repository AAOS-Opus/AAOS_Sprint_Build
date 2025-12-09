# src/agent_lifecycle/__init__.py
"""
Agent Lifecycle Module - Health Monitoring, Circuit Breakers, State Machine, and Consensus
Integration 2A: maestro-orchestra/agent_lifecycle fork (Health Monitor + Circuit Breaker)
Integration 2B: Agent Lifecycle Orchestration (state transitions)
Integration 2C: BFT Consensus (Byzantine Fault Tolerance)
"""

from .health_monitor import HealthMonitor, AgentHealth, HealthStatus, HealthMonitorConfig
from .circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerConfig
from .state_machine import (
    AgentStateMachine,
    AgentState,
    AgentLifecycleState,
    StateMachineConfig,
    StateTransition,
    InvalidTransitionError,
    VALID_TRANSITIONS,
)
from .orchestrator import LifecycleOrchestrator

__all__ = [
    # Health Monitor (Integration 2A)
    "HealthMonitor",
    "HealthMonitorConfig",
    "AgentHealth",
    "HealthStatus",
    # Circuit Breaker (Integration 2A)
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    # State Machine (Integration 2B)
    "AgentStateMachine",
    "AgentState",
    "AgentLifecycleState",
    "StateMachineConfig",
    "StateTransition",
    "InvalidTransitionError",
    "VALID_TRANSITIONS",
    # Orchestrator (Integration 2A + 2B + 2C)
    "LifecycleOrchestrator",
]

# Note: BFT Consensus (Integration 2C) is in src/bft_consensus module
# Import from src.bft_consensus for consensus components
