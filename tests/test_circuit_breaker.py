# tests/test_circuit_breaker.py
"""
Test suite for Fix #4: Circuit Breaker for Zero-Agents

Tests:
1. Circuit opens when no agents connected
2. 503 returned when circuit is open
3. Half-open state allows test task through
4. Circuit closes when agents reconnect (test task picked up)
5. Queue overflow triggers circuit open
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import redis

# Import the modules we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.orchestrator.core import (
    TaskCircuitBreaker,
    TaskCircuitState,
    task_circuit_breaker,
    CIRCUIT_MIN_AGENTS,
    CIRCUIT_MAX_QUEUE_SIZE,
    CIRCUIT_RECOVERY_TIMEOUT,
    CIRCUIT_TEST_TASK_TIMEOUT,
    REDIS_HOST,
    REDIS_PORT
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def circuit_breaker():
    """Create a fresh circuit breaker for testing"""
    return TaskCircuitBreaker(
        min_agents=1,
        max_queue_size=100,
        recovery_timeout=5,  # Short timeout for testing
        test_task_timeout=10,  # Short timeout for testing
        enabled=True
    )


@pytest.fixture
def redis_client():
    """Create a Redis client for testing"""
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    yield client
    # Cleanup
    client.delete("task_queue")


# =============================================================================
# Test: Circuit Breaker States
# =============================================================================

class TestCircuitBreakerStates:
    """Tests for circuit breaker state management"""

    def test_initial_state_is_closed(self, circuit_breaker):
        """Circuit breaker starts in CLOSED state"""
        assert circuit_breaker.state == TaskCircuitState.CLOSED

    def test_disabled_always_returns_closed(self):
        """Disabled circuit breaker always returns CLOSED"""
        cb = TaskCircuitBreaker(enabled=False)
        cb._state = TaskCircuitState.OPEN
        assert cb.state == TaskCircuitState.CLOSED

    def test_state_enum_values(self):
        """Verify state enum has correct values"""
        assert TaskCircuitState.CLOSED.value == "closed"
        assert TaskCircuitState.OPEN.value == "open"
        assert TaskCircuitState.HALF_OPEN.value == "half_open"


# =============================================================================
# Test: Circuit Opens When No Agents Connected
# =============================================================================

class TestCircuitOpensNoAgents:
    """Tests for circuit opening when no agents are available"""

    def test_opens_when_zero_agents(self, circuit_breaker):
        """Circuit opens when active_agents < min_agents"""
        # Check conditions with 0 agents
        allowed = circuit_breaker.check_conditions(active_agents=0, queue_size=0)

        assert allowed is False
        assert circuit_breaker.state == TaskCircuitState.OPEN
        assert "insufficient_agents" in circuit_breaker._open_reason

    def test_stays_closed_with_enough_agents(self, circuit_breaker):
        """Circuit stays closed when enough agents are available"""
        allowed = circuit_breaker.check_conditions(active_agents=1, queue_size=0)

        assert allowed is True
        assert circuit_breaker.state == TaskCircuitState.CLOSED

    def test_opens_with_queue_overflow(self, circuit_breaker):
        """Circuit opens when queue exceeds max size"""
        allowed = circuit_breaker.check_conditions(active_agents=5, queue_size=100)

        assert allowed is False
        assert circuit_breaker.state == TaskCircuitState.OPEN
        assert "queue_overflow" in circuit_breaker._open_reason


# =============================================================================
# Test: 503 Returned When Circuit Is Open
# =============================================================================

class TestCircuitRejectsTasks:
    """Tests for task rejection when circuit is open"""

    def test_reject_task_when_open(self, circuit_breaker):
        """Tasks are rejected when circuit is open"""
        # Open the circuit
        circuit_breaker.check_conditions(active_agents=0, queue_size=0)
        assert circuit_breaker.state == TaskCircuitState.OPEN

        # Try to allow a task
        task_id = str(uuid.uuid4())
        allowed = circuit_breaker.allow_task(task_id)

        assert allowed is False
        assert circuit_breaker._tasks_rejected == 1

    def test_retry_after_calculated_correctly(self, circuit_breaker):
        """Retry-after value is calculated correctly"""
        # Open the circuit
        circuit_breaker.check_conditions(active_agents=0, queue_size=0)

        # Immediately check retry_after (should be close to recovery_timeout)
        retry_after = circuit_breaker.get_retry_after()
        assert 1 <= retry_after <= circuit_breaker.recovery_timeout

    def test_multiple_rejections_tracked(self, circuit_breaker):
        """Multiple task rejections are tracked"""
        # Open the circuit
        circuit_breaker.check_conditions(active_agents=0, queue_size=0)

        # Try to submit multiple tasks
        for _ in range(5):
            circuit_breaker.allow_task(str(uuid.uuid4()))

        assert circuit_breaker._tasks_rejected == 5


# =============================================================================
# Test: Half-Open State Allows Test Task Through
# =============================================================================

class TestHalfOpenState:
    """Tests for half-open state behavior"""

    def test_transitions_to_half_open_after_timeout(self, circuit_breaker):
        """Circuit transitions to HALF_OPEN after recovery timeout"""
        # Open the circuit
        circuit_breaker.check_conditions(active_agents=0, queue_size=0)
        assert circuit_breaker.state == TaskCircuitState.OPEN

        # Simulate time passing (set opened_at to the past)
        circuit_breaker._opened_at = datetime.utcnow() - timedelta(seconds=circuit_breaker.recovery_timeout + 1)

        # Check state (should trigger transition)
        assert circuit_breaker.state == TaskCircuitState.HALF_OPEN

    def test_half_open_allows_one_test_task(self, circuit_breaker):
        """Half-open state allows exactly one test task"""
        # Open the circuit
        circuit_breaker.check_conditions(active_agents=0, queue_size=0)

        # Force transition to HALF_OPEN
        circuit_breaker._opened_at = datetime.utcnow() - timedelta(seconds=circuit_breaker.recovery_timeout + 1)
        _ = circuit_breaker.state  # Trigger transition

        # First task should be allowed
        task_id = str(uuid.uuid4())
        allowed = circuit_breaker.allow_task(task_id)
        assert allowed is True
        assert circuit_breaker._test_task_id == task_id

        # Second task should be rejected
        task_id_2 = str(uuid.uuid4())
        allowed_2 = circuit_breaker.allow_task(task_id_2)
        assert allowed_2 is False

    def test_check_conditions_allows_test_task_in_half_open(self, circuit_breaker):
        """check_conditions returns True in HALF_OPEN when no test task submitted"""
        # Force to HALF_OPEN state
        circuit_breaker._state = TaskCircuitState.HALF_OPEN
        circuit_breaker._test_task_id = None

        # Should allow (for test task)
        allowed = circuit_breaker.check_conditions(active_agents=1, queue_size=0)
        assert allowed is True

    def test_check_conditions_rejects_in_half_open_with_test_pending(self, circuit_breaker):
        """check_conditions returns False in HALF_OPEN when test task is pending"""
        # Force to HALF_OPEN state with pending test task
        circuit_breaker._state = TaskCircuitState.HALF_OPEN
        circuit_breaker._test_task_id = "existing-test-task"

        # Should reject (test task already pending)
        allowed = circuit_breaker.check_conditions(active_agents=1, queue_size=0)
        assert allowed is False


# =============================================================================
# Test: Circuit Closes When Agents Reconnect
# =============================================================================

class TestCircuitCloses:
    """Tests for circuit closing when test task is picked up"""

    def test_closes_when_test_task_assigned(self, circuit_breaker):
        """Circuit closes when test task is picked up by agent"""
        # Open circuit and transition to HALF_OPEN
        circuit_breaker.check_conditions(active_agents=0, queue_size=0)
        circuit_breaker._opened_at = datetime.utcnow() - timedelta(seconds=circuit_breaker.recovery_timeout + 1)
        _ = circuit_breaker.state  # Trigger transition to HALF_OPEN

        # Submit test task
        test_task_id = str(uuid.uuid4())
        circuit_breaker.allow_task(test_task_id)
        assert circuit_breaker._test_task_id == test_task_id

        # Simulate task being assigned
        circuit_breaker.on_task_assigned(test_task_id)

        # Circuit should now be CLOSED
        assert circuit_breaker.state == TaskCircuitState.CLOSED
        assert circuit_breaker._test_task_id is None

    def test_ignores_non_test_task_assignment(self, circuit_breaker):
        """Non-test task assignments don't affect circuit state"""
        # Force to HALF_OPEN with a test task
        circuit_breaker._state = TaskCircuitState.HALF_OPEN
        circuit_breaker._test_task_id = "test-task-123"

        # Assign a different task
        circuit_breaker.on_task_assigned("different-task-456")

        # Should still be HALF_OPEN
        assert circuit_breaker._state == TaskCircuitState.HALF_OPEN
        assert circuit_breaker._test_task_id == "test-task-123"

    def test_reopens_if_test_task_times_out(self, circuit_breaker):
        """Circuit reopens if test task is not picked up in time"""
        # Force to HALF_OPEN with test task
        circuit_breaker._state = TaskCircuitState.HALF_OPEN
        circuit_breaker._test_task_id = "test-task-123"
        circuit_breaker._test_task_submitted_at = datetime.utcnow() - timedelta(
            seconds=circuit_breaker.test_task_timeout + 1
        )

        # Check state (should trigger timeout)
        state = circuit_breaker.state

        # Should be back to OPEN
        assert state == TaskCircuitState.OPEN
        assert circuit_breaker._test_task_id is None


# =============================================================================
# Test: Force Check Method
# =============================================================================

class TestForceCheck:
    """Tests for the force_check method"""

    def test_force_check_opens_circuit_when_conditions_degrade(self, circuit_breaker):
        """force_check opens circuit when conditions degrade"""
        assert circuit_breaker.state == TaskCircuitState.CLOSED

        circuit_breaker.force_check(active_agents=0, queue_size=0)

        assert circuit_breaker.state == TaskCircuitState.OPEN

    def test_force_check_transitions_to_half_open_when_conditions_improve(self, circuit_breaker):
        """force_check transitions to HALF_OPEN when conditions improve after timeout"""
        # Open circuit
        circuit_breaker.force_check(active_agents=0, queue_size=0)
        assert circuit_breaker.state == TaskCircuitState.OPEN

        # Simulate time passing
        circuit_breaker._opened_at = datetime.utcnow() - timedelta(seconds=circuit_breaker.recovery_timeout + 1)

        # Force check with good conditions
        circuit_breaker.force_check(active_agents=5, queue_size=10)

        assert circuit_breaker.state == TaskCircuitState.HALF_OPEN


# =============================================================================
# Test: Circuit Breaker Status
# =============================================================================

class TestCircuitBreakerStatus:
    """Tests for get_status method"""

    def test_status_contains_required_fields(self, circuit_breaker):
        """Status response contains all required fields"""
        status = circuit_breaker.get_status()

        assert "enabled" in status
        assert "state" in status
        assert "open_reason" in status
        assert "last_state_change" in status
        assert "tasks_rejected" in status
        assert "state_changes" in status
        assert "config" in status

    def test_status_config_contains_settings(self, circuit_breaker):
        """Status config contains all settings"""
        status = circuit_breaker.get_status()
        config = status["config"]

        assert "min_agents" in config
        assert "max_queue_size" in config
        assert "recovery_timeout" in config
        assert "test_task_timeout" in config

    def test_status_tracks_state_changes(self, circuit_breaker):
        """Status tracks number of state changes"""
        assert circuit_breaker._state_changes == 0

        # Trigger state change
        circuit_breaker.check_conditions(active_agents=0, queue_size=0)

        status = circuit_breaker.get_status()
        assert status["state_changes"] >= 1


# =============================================================================
# Test: State Transitions
# =============================================================================

class TestStateTransitions:
    """Tests for valid state transitions"""

    def test_closed_to_open_transition(self, circuit_breaker):
        """CLOSED -> OPEN transition works"""
        assert circuit_breaker.state == TaskCircuitState.CLOSED

        circuit_breaker.check_conditions(active_agents=0, queue_size=0)

        assert circuit_breaker.state == TaskCircuitState.OPEN

    def test_open_to_half_open_transition(self, circuit_breaker):
        """OPEN -> HALF_OPEN transition works after timeout"""
        # Open circuit
        circuit_breaker.check_conditions(active_agents=0, queue_size=0)
        assert circuit_breaker.state == TaskCircuitState.OPEN

        # Simulate timeout
        circuit_breaker._opened_at = datetime.utcnow() - timedelta(seconds=circuit_breaker.recovery_timeout + 1)

        # Access state to trigger transition
        assert circuit_breaker.state == TaskCircuitState.HALF_OPEN

    def test_half_open_to_closed_transition(self, circuit_breaker):
        """HALF_OPEN -> CLOSED transition works when test succeeds"""
        # Force to HALF_OPEN
        circuit_breaker._state = TaskCircuitState.HALF_OPEN
        test_task_id = str(uuid.uuid4())
        circuit_breaker._test_task_id = test_task_id

        # Simulate test task success
        circuit_breaker.on_task_assigned(test_task_id)

        assert circuit_breaker.state == TaskCircuitState.CLOSED

    def test_half_open_to_open_transition(self, circuit_breaker):
        """HALF_OPEN -> OPEN transition works when test times out"""
        # Force to HALF_OPEN with timed out test task
        circuit_breaker._state = TaskCircuitState.HALF_OPEN
        circuit_breaker._test_task_id = "test-task"
        circuit_breaker._test_task_submitted_at = datetime.utcnow() - timedelta(
            seconds=circuit_breaker.test_task_timeout + 1
        )

        # Access state to trigger transition
        assert circuit_breaker.state == TaskCircuitState.OPEN


# =============================================================================
# Integration Test: Full Circuit Breaker Cycle
# =============================================================================

class TestFullCircuitBreakerCycle:
    """Integration tests for complete circuit breaker cycle"""

    def test_full_cycle_no_agents_to_recovery(self, circuit_breaker):
        """Test complete cycle: no agents -> open -> half-open -> test -> closed"""
        # Step 1: Start CLOSED
        assert circuit_breaker.state == TaskCircuitState.CLOSED

        # Step 2: No agents - circuit opens
        circuit_breaker.check_conditions(active_agents=0, queue_size=0)
        assert circuit_breaker.state == TaskCircuitState.OPEN

        # Step 3: Wait for recovery timeout
        circuit_breaker._opened_at = datetime.utcnow() - timedelta(seconds=circuit_breaker.recovery_timeout + 1)

        # Step 4: Check state - should transition to HALF_OPEN
        assert circuit_breaker.state == TaskCircuitState.HALF_OPEN

        # Step 5: Submit test task
        test_task_id = str(uuid.uuid4())
        allowed = circuit_breaker.allow_task(test_task_id)
        assert allowed is True
        assert circuit_breaker._test_task_id == test_task_id

        # Step 6: Agent picks up test task - circuit closes
        circuit_breaker.on_task_assigned(test_task_id)
        assert circuit_breaker.state == TaskCircuitState.CLOSED

        # Step 7: Normal operation resumes
        normal_task_id = str(uuid.uuid4())
        allowed_normal = circuit_breaker.allow_task(normal_task_id)
        assert allowed_normal is True


# =============================================================================
# Test: Disabled Circuit Breaker
# =============================================================================

class TestDisabledCircuitBreaker:
    """Tests for disabled circuit breaker"""

    def test_disabled_always_allows_tasks(self):
        """Disabled circuit breaker always allows tasks"""
        cb = TaskCircuitBreaker(enabled=False)

        # Even with bad conditions
        allowed = cb.check_conditions(active_agents=0, queue_size=10000)
        assert allowed is True

        task_allowed = cb.allow_task(str(uuid.uuid4()))
        assert task_allowed is True

    def test_disabled_force_check_does_nothing(self):
        """Disabled circuit breaker's force_check does nothing"""
        cb = TaskCircuitBreaker(enabled=False)
        cb._state = TaskCircuitState.CLOSED

        cb.force_check(active_agents=0, queue_size=10000)

        # Should still be CLOSED (force_check should have no effect)
        assert cb._state == TaskCircuitState.CLOSED


# =============================================================================
# Main entry point for running tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
