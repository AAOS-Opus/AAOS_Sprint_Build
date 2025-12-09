# tests/test_integration_2a.py
"""
Integration 2A Tests - Agent Lifecycle Health Monitor
Validates all requirements from layer-cake-integration-v1.md
"""

import pytest
import asyncio
from datetime import datetime

# Test the lifecycle module directly
from src.agent_lifecycle import (
    HealthMonitor,
    HealthMonitorConfig,
    HealthStatus,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    LifecycleOrchestrator,
)


class TestHealthMonitor:
    """Tests for HealthMonitor component"""

    def test_health_monitor_initialization(self):
        """Health monitor should initialize without error"""
        monitor = HealthMonitor()
        assert monitor is not None
        assert monitor._initialized is False

    @pytest.mark.asyncio
    async def test_health_monitor_async_init(self):
        """Health monitor async initialization"""
        monitor = HealthMonitor()
        await monitor.initialize()
        assert monitor._initialized is True

    def test_register_ensemble_agents(self):
        """Should register all 9 ensemble agents"""
        monitor = HealthMonitor()
        agents = monitor.register_ensemble_agents()

        assert len(agents) == 9

        expected_agents = [
            "maestro", "opus", "claude", "devzen",
            "frontend", "backend", "kimi", "scout", "dr-aeon"
        ]
        registered_ids = list(monitor.agents.keys())

        for agent_id in expected_agents:
            assert agent_id in registered_ids, f"Missing agent: {agent_id}"

    def test_record_heartbeat(self):
        """Should record heartbeats for registered agents"""
        monitor = HealthMonitor()
        monitor.register_agent("test-agent", "test")

        result = monitor.record_heartbeat("test-agent")
        assert result is True

        agent = monitor.get_agent_health("test-agent")
        assert agent is not None
        assert agent.heartbeat_count == 1
        assert agent.last_heartbeat is not None

    def test_unregistered_heartbeat(self):
        """Should reject heartbeats from unregistered agents"""
        monitor = HealthMonitor()
        result = monitor.record_heartbeat("unknown-agent")
        assert result is False


class TestCircuitBreaker:
    """Tests for CircuitBreaker component"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_starts_closed(self):
        """Circuit breakers should start in CLOSED state"""
        breaker = CircuitBreaker(agent_id="test-agent")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_allows_execution_when_closed(self):
        """CLOSED circuit should allow execution"""
        breaker = CircuitBreaker(agent_id="test-agent")
        can_execute = await breaker.can_execute()
        assert can_execute is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Circuit should OPEN after threshold failures"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(agent_id="test-agent", config=config)

        # Record failures up to threshold
        for _ in range(3):
            await breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_when_open(self):
        """OPEN circuit should block execution"""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60)
        breaker = CircuitBreaker(agent_id="test-agent", config=config)

        await breaker.record_failure()
        assert breaker.is_open is True

        can_execute = await breaker.can_execute()
        assert can_execute is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_status(self):
        """Should return complete status dict"""
        breaker = CircuitBreaker(agent_id="test-agent")
        status = breaker.get_status()

        assert status["agent_id"] == "test-agent"
        assert status["state"] == "closed"
        assert "config" in status


class TestLifecycleOrchestrator:
    """Tests for LifecycleOrchestrator - main integration"""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Orchestrator should initialize without error"""
        orchestrator = LifecycleOrchestrator()
        await orchestrator.initialize()

        assert orchestrator._initialized is True
        assert orchestrator.get_registered_agent_count() == 9

    @pytest.mark.asyncio
    async def test_all_agents_registered(self):
        """All 9 ensemble agents should be registered"""
        orchestrator = LifecycleOrchestrator()
        await orchestrator.initialize()

        agent_ids = orchestrator.get_agent_ids()
        expected = [
            "maestro", "opus", "claude", "devzen",
            "frontend", "backend", "kimi", "scout", "dr-aeon"
        ]

        assert len(agent_ids) == 9
        for agent_id in expected:
            assert agent_id in agent_ids

    @pytest.mark.asyncio
    async def test_all_circuit_breakers_closed(self):
        """All circuit breakers should be in CLOSED state"""
        orchestrator = LifecycleOrchestrator()
        await orchestrator.initialize()

        assert orchestrator.verify_circuit_breakers_closed() is True

        for agent_id, breaker in orchestrator.get_all_circuit_breakers().items():
            assert breaker.state == CircuitState.CLOSED, \
                f"Circuit breaker for {agent_id} not CLOSED"

    @pytest.mark.asyncio
    async def test_heartbeat_stabilization(self):
        """Should wait for heartbeat stabilization (3s minimum)"""
        config = HealthMonitorConfig(stabilization_time=0.1)  # Fast for testing
        orchestrator = LifecycleOrchestrator(health_config=config)
        await orchestrator.initialize()
        await orchestrator.start()

        # After start, should be ready
        assert orchestrator._ready is True

    @pytest.mark.asyncio
    async def test_readiness_gate(self):
        """GATE: health_monitor.readiness() == True"""
        config = HealthMonitorConfig(stabilization_time=0.1)
        orchestrator = LifecycleOrchestrator(health_config=config)
        await orchestrator.initialize()
        await orchestrator.start()

        # GATE CHECK
        assert orchestrator.readiness() is True

    @pytest.mark.asyncio
    async def test_health_status_response(self):
        """Health status should include all required fields"""
        config = HealthMonitorConfig(stabilization_time=0.1)
        orchestrator = LifecycleOrchestrator(health_config=config)
        await orchestrator.initialize()
        await orchestrator.start()

        status = orchestrator.get_health_status()

        # Required fields
        assert "ready" in status
        assert "initialized" in status
        assert "health" in status
        assert "circuit_breakers" in status
        assert "all_circuits_closed" in status
        assert "all_agents_healthy" in status
        assert "timestamp" in status

        # Should have 9 agents in health
        assert status["health"]["total_agents"] == 9

        # Should have 9 circuit breakers
        assert len(status["circuit_breakers"]) == 9


class TestIntegration2AValidation:
    """
    Final validation tests for Integration 2A requirements.
    These tests verify the complete integration per workflow spec.
    """

    @pytest.mark.asyncio
    async def test_validation_health_monitor_initializes(self):
        """VALIDATION: Health monitor initializes without error"""
        orchestrator = LifecycleOrchestrator()
        try:
            await orchestrator.initialize()
            assert True, "Health monitor initialized successfully"
        except Exception as e:
            pytest.fail(f"Health monitor initialization failed: {e}")

    @pytest.mark.asyncio
    async def test_validation_9_agents_registered(self):
        """VALIDATION: All 9 agents registered"""
        orchestrator = LifecycleOrchestrator()
        await orchestrator.initialize()

        count = orchestrator.get_registered_agent_count()
        assert count == 9, f"Expected 9 agents, got {count}"

    @pytest.mark.asyncio
    async def test_validation_circuit_breakers_closed(self):
        """VALIDATION: Circuit breakers in CLOSED state"""
        orchestrator = LifecycleOrchestrator()
        await orchestrator.initialize()

        assert orchestrator.verify_circuit_breakers_closed(), \
            "Not all circuit breakers in CLOSED state"

    @pytest.mark.asyncio
    async def test_validation_gate_passed(self):
        """GATE: health_monitor.readiness() == True"""
        config = HealthMonitorConfig(stabilization_time=0.1)
        orchestrator = LifecycleOrchestrator(health_config=config)
        await orchestrator.initialize()
        await orchestrator.start()

        assert orchestrator.readiness() is True, \
            "GATE FAILED: health_monitor.readiness() != True"

        print("\n" + "=" * 60)
        print("GATE PASSED: health_monitor.readiness() == True")
        print("=" * 60)
