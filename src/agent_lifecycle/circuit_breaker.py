# src/agent_lifecycle/circuit_breaker.py
"""
Circuit Breaker Pattern Implementation
Protects agents from cascading failures with state management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field

logger = logging.getLogger("aaos.lifecycle.circuit_breaker")


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 30.0      # Seconds before trying half-open
    success_threshold: int = 3          # Successes in half-open to close
    half_open_max_calls: int = 3        # Max concurrent calls in half-open


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for individual agent protection.
    Starts in CLOSED state per integration requirements.
    """
    agent_id: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # State tracking
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: Optional[datetime] = field(default=None)
    last_state_change: datetime = field(default_factory=datetime.utcnow)
    half_open_calls: int = field(default=0)

    # Lock for thread safety
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self):
        logger.info(f"Circuit breaker initialized for {self.agent_id} in {self.state.value} state")

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN

    async def can_execute(self) -> bool:
        """Check if a request can proceed through the circuit"""
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self.last_failure_time:
                    elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                    if elapsed >= self.config.recovery_timeout:
                        self._transition_to(CircuitState.HALF_OPEN)
                        return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self.half_open_calls < self.config.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self):
        """Record a successful operation"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    async def record_failure(self):
        """Record a failed operation"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()

            if self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                self._transition_to(CircuitState.OPEN)
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state with logging"""
        old_state = self.state
        self.state = new_state
        self.last_state_change = datetime.utcnow()

        # Reset counters based on new state
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.success_count = 0
            self.half_open_calls = 0
        elif new_state == CircuitState.OPEN:
            self.success_count = 0
            self.half_open_calls = 0

        logger.info(f"Circuit breaker {self.agent_id}: {old_state.value} -> {new_state.value}")

    async def force_open(self):
        """Manually open the circuit (for maintenance/emergency)"""
        async with self._lock:
            self._transition_to(CircuitState.OPEN)
            self.last_failure_time = datetime.utcnow()

    async def force_close(self):
        """Manually close the circuit (for recovery)"""
        async with self._lock:
            self._transition_to(CircuitState.CLOSED)

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold
            }
        }
