# tests/test_integration_2b.py
"""
Integration 2B Tests - Agent Lifecycle Orchestration
Validates all requirements from layer-cake-integration-v1.md

STEPS:
1. Enable state transitions (INIT → DISCOVERY → READY)
2. Test state machine - verify all valid transitions work
3. Test circuit breaker activation (simulated failures)
4. Test graceful degradation (system continues with reduced agents)
5. Create validation endpoint or test suite

GATE: All agents in READY state
"""

import pytest
import asyncio
from datetime import datetime

from src.agent_lifecycle import (
    # Integration 2A components
    HealthMonitor,
    HealthMonitorConfig,
    HealthStatus,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    LifecycleOrchestrator,
    # Integration 2B components
    AgentStateMachine,
    AgentState,
    AgentLifecycleState,
    StateMachineConfig,
    StateTransition,
    InvalidTransitionError,
    VALID_TRANSITIONS,
)


class TestAgentStateMachine:
    """Tests for AgentStateMachine component (Integration 2B)"""

    def test_state_machine_initialization(self):
        """State machine should initialize without error"""
        sm = AgentStateMachine()
        assert sm is not None
        assert len(sm.agents) == 0

    def test_agent_registration_starts_in_init(self):
        """Registered agents should start in INIT state"""
        sm = AgentStateMachine()
        state = sm.register_agent("test-agent", "test")

        assert state.agent_id == "test-agent"
        assert state.agent_type == "test"
        assert state.state == AgentState.INIT
        assert len(state.transition_history) == 0

    def test_duplicate_registration_returns_existing(self):
        """Duplicate registration should return existing state"""
        sm = AgentStateMachine()
        state1 = sm.register_agent("test-agent", "test")
        state2 = sm.register_agent("test-agent", "test")

        assert state1 is state2


class TestStateTransitions:
    """Tests for state transition validation (Integration 2B Step 2)"""

    def test_valid_transitions_defined(self):
        """All states should have defined transitions"""
        for state in AgentState:
            assert state in VALID_TRANSITIONS

    def test_init_to_discovery_valid(self):
        """INIT → DISCOVERY should be valid"""
        state = AgentLifecycleState(agent_id="test", agent_type="test")
        assert state.can_transition_to(AgentState.DISCOVERY) is True

    def test_discovery_to_ready_valid(self):
        """DISCOVERY → READY should be valid"""
        state = AgentLifecycleState(
            agent_id="test",
            agent_type="test",
            state=AgentState.DISCOVERY
        )
        assert state.can_transition_to(AgentState.READY) is True

    def test_ready_to_busy_valid(self):
        """READY → BUSY should be valid"""
        state = AgentLifecycleState(
            agent_id="test",
            agent_type="test",
            state=AgentState.READY
        )
        assert state.can_transition_to(AgentState.BUSY) is True

    def test_ready_to_degraded_valid(self):
        """READY → DEGRADED should be valid"""
        state = AgentLifecycleState(
            agent_id="test",
            agent_type="test",
            state=AgentState.READY
        )
        assert state.can_transition_to(AgentState.DEGRADED) is True

    def test_degraded_to_ready_valid(self):
        """DEGRADED → READY should be valid (recovery)"""
        state = AgentLifecycleState(
            agent_id="test",
            agent_type="test",
            state=AgentState.DEGRADED
        )
        assert state.can_transition_to(AgentState.READY) is True

    def test_failed_to_init_valid(self):
        """FAILED → INIT should be valid (restart)"""
        state = AgentLifecycleState(
            agent_id="test",
            agent_type="test",
            state=AgentState.FAILED
        )
        assert state.can_transition_to(AgentState.INIT) is True

    def test_shutdown_is_terminal(self):
        """SHUTDOWN should be terminal (no valid transitions)"""
        state = AgentLifecycleState(
            agent_id="test",
            agent_type="test",
            state=AgentState.SHUTDOWN
        )
        assert len(VALID_TRANSITIONS[AgentState.SHUTDOWN]) == 0
        for target_state in AgentState:
            assert state.can_transition_to(target_state) is False

    def test_invalid_transition_raises_error(self):
        """Invalid transitions should raise InvalidTransitionError"""
        state = AgentLifecycleState(agent_id="test", agent_type="test")

        # INIT → READY is not valid (must go through DISCOVERY)
        with pytest.raises(InvalidTransitionError):
            state.transition_to(AgentState.READY)

    @pytest.mark.asyncio
    async def test_transition_history_recorded(self):
        """Transitions should be recorded in history"""
        sm = AgentStateMachine()
        sm.register_agent("test", "test")

        await sm.transition_agent("test", AgentState.DISCOVERY, "Test reason")

        state = sm.get_agent_state("test")
        assert len(state.transition_history) == 1
        assert state.transition_history[0].from_state == AgentState.INIT
        assert state.transition_history[0].to_state == AgentState.DISCOVERY
        assert state.transition_history[0].reason == "Test reason"


class TestAdvanceToReady:
    """Tests for INIT → DISCOVERY → READY advancement (Integration 2B Step 1)"""

    @pytest.mark.asyncio
    async def test_advance_to_ready_single_agent(self):
        """Single agent should advance through INIT → DISCOVERY → READY"""
        config = StateMachineConfig(discovery_delay=0.01)  # Fast for testing
        sm = AgentStateMachine(config=config)
        sm.register_agent("test", "test")

        success = await sm.advance_to_ready("test")

        assert success is True
        state = sm.get_agent_state("test")
        assert state.state == AgentState.READY
        assert len(state.transition_history) == 2  # INIT→DISCOVERY, DISCOVERY→READY

    @pytest.mark.asyncio
    async def test_advance_all_to_ready(self):
        """All agents should advance to READY state"""
        config = StateMachineConfig(discovery_delay=0.01)
        sm = AgentStateMachine(config=config)

        # Register multiple agents
        for i in range(5):
            sm.register_agent(f"agent-{i}", "test")

        results = await sm.advance_all_to_ready()

        # All should succeed
        assert all(results.values())
        assert sm.all_agents_ready() is True
        assert sm.get_ready_count() == 5


class TestCircuitBreakerWithFailure:
    """Tests for circuit breaker activation with simulated failures (Integration 2B Step 3)"""

    @pytest.mark.asyncio
    async def test_simulate_failure_transitions_to_failed(self):
        """Simulating failure should transition agent to FAILED state"""
        config = StateMachineConfig(discovery_delay=0.01)
        sm = AgentStateMachine(config=config)
        sm.register_agent("test", "test")
        await sm.advance_to_ready("test")

        success = await sm.simulate_failure("test", "Test failure")

        assert success is True
        state = sm.get_agent_state("test")
        assert state.state == AgentState.FAILED
        assert state.failure_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failure(self):
        """Circuit breaker should open when failures exceed threshold"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(agent_id="test", config=config)

        # Simulate failures
        for _ in range(3):
            await breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert await breaker.can_execute() is False

    @pytest.mark.asyncio
    async def test_orchestrator_failure_simulation(self):
        """Orchestrator should handle failure simulation correctly"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        # All agents should be READY initially
        assert orchestrator._states_ready is True

        # Simulate failure for one agent
        success = await orchestrator.simulate_agent_failure("maestro", "Test failure")
        assert success is True

        # States_ready should now be False
        assert orchestrator._states_ready is False

        # Get failed agents
        failed = orchestrator.get_failed_agents()
        assert "maestro" in failed


class TestGracefulDegradation:
    """Tests for graceful degradation (Integration 2B Step 4)"""

    @pytest.mark.asyncio
    async def test_system_operates_with_reduced_agents(self):
        """System should continue operating with reduced agents"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        # Fail one agent
        await orchestrator.simulate_agent_failure("maestro", "Test")

        # System should still be able to operate
        assert orchestrator.can_degrade_gracefully() is True

        # Should have 8 operational agents
        operational = orchestrator.get_operational_agents()
        assert len(operational) == 8
        assert "maestro" not in operational

    @pytest.mark.asyncio
    async def test_degradation_status_reporting(self):
        """Degradation status should be accurately reported"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        # Fail 2 agents
        await orchestrator.simulate_agent_failure("maestro", "Test")
        await orchestrator.simulate_agent_failure("opus", "Test")

        status = orchestrator.get_degradation_status()

        assert status["total_agents"] == 9
        assert status["operational_agents"] == 7
        assert status["failed_agents"] == 2
        assert status["can_operate"] is True
        # Degradation level should be 2/9 ≈ 0.222
        assert 0.2 < status["degradation_level"] < 0.3

    @pytest.mark.asyncio
    async def test_dispatch_blocked_to_failed_agents(self):
        """Dispatch should be blocked to failed agents"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        # Should be able to dispatch initially
        can_dispatch = await orchestrator.can_dispatch_to_agent("maestro")
        assert can_dispatch is True

        # Fail the agent
        await orchestrator.simulate_agent_failure("maestro", "Test")

        # Should not be able to dispatch now
        can_dispatch = await orchestrator.can_dispatch_to_agent("maestro")
        assert can_dispatch is False


class TestAgentRecovery:
    """Tests for agent recovery from FAILED state"""

    @pytest.mark.asyncio
    async def test_recover_agent_from_failed(self):
        """Agent should recover from FAILED → INIT → DISCOVERY → READY"""
        config = StateMachineConfig(discovery_delay=0.01)
        sm = AgentStateMachine(config=config)
        sm.register_agent("test", "test")

        # Advance to ready then fail
        await sm.advance_to_ready("test")
        await sm.simulate_failure("test", "Test")

        state = sm.get_agent_state("test")
        assert state.state == AgentState.FAILED

        # Recover
        success = await sm.recover_agent("test")

        assert success is True
        state = sm.get_agent_state("test")
        assert state.state == AgentState.READY
        assert state.recovery_count == 1

    @pytest.mark.asyncio
    async def test_orchestrator_recovery(self):
        """Orchestrator should handle agent recovery correctly"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        # Fail and recover
        await orchestrator.simulate_agent_failure("maestro", "Test")
        assert orchestrator._states_ready is False

        success = await orchestrator.recover_agent("maestro")
        assert success is True
        assert orchestrator._states_ready is True


class TestHealthStatusIntegration:
    """Tests for health status including state machine (Integration 2B)"""

    @pytest.mark.asyncio
    async def test_health_status_includes_state_machine(self):
        """Health status should include state machine information"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        status = orchestrator.get_health_status()

        # Integration 2B fields
        assert "states_ready" in status
        assert "state_machine" in status
        assert "all_agents_ready" in status

        assert status["states_ready"] is True
        assert status["all_agents_ready"] is True
        assert status["state_machine"]["all_ready"] is True
        assert status["state_machine"]["ready_agents"] == 9


class TestIntegration2BValidation:
    """
    Final validation tests for Integration 2B requirements.
    These tests verify the complete integration per workflow spec.
    """

    @pytest.mark.asyncio
    async def test_validation_state_transitions_enabled(self):
        """VALIDATION: State transitions (INIT → DISCOVERY → READY) work"""
        config = StateMachineConfig(discovery_delay=0.01)
        sm = AgentStateMachine(config=config)
        sm.register_agent("test", "test")

        # Start in INIT
        state = sm.get_agent_state("test")
        assert state.state == AgentState.INIT

        # Transition to DISCOVERY
        await sm.transition_agent("test", AgentState.DISCOVERY)
        assert state.state == AgentState.DISCOVERY

        # Transition to READY
        await sm.transition_agent("test", AgentState.READY)
        assert state.state == AgentState.READY

        print("\n✓ VALIDATION: State transitions enabled")

    @pytest.mark.asyncio
    async def test_validation_all_valid_transitions_work(self):
        """VALIDATION: All valid state transitions work"""
        test_transitions = [
            (AgentState.INIT, AgentState.DISCOVERY),
            (AgentState.DISCOVERY, AgentState.READY),
            (AgentState.READY, AgentState.BUSY),
            (AgentState.BUSY, AgentState.READY),
            (AgentState.READY, AgentState.DEGRADED),
            (AgentState.DEGRADED, AgentState.READY),
            (AgentState.READY, AgentState.FAILED),
            (AgentState.FAILED, AgentState.INIT),
        ]

        sm = AgentStateMachine()

        for from_state, to_state in test_transitions:
            # Create fresh state in from_state
            state = AgentLifecycleState(
                agent_id=f"test-{from_state.value}-{to_state.value}",
                agent_type="test",
                state=from_state
            )
            # Verify transition is valid
            assert state.can_transition_to(to_state), \
                f"Transition {from_state.value} → {to_state.value} should be valid"

        print("\n✓ VALIDATION: All valid transitions work")

    @pytest.mark.asyncio
    async def test_validation_circuit_breaker_activation(self):
        """VALIDATION: Circuit breaker trips on simulated failure"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        circuit_config = CircuitBreakerConfig(failure_threshold=1)  # Trip on first failure

        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config,
            circuit_config=circuit_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        # Simulate failure
        await orchestrator.simulate_agent_failure("maestro", "Test")

        # Verify circuit breaker tripped
        breaker = orchestrator.get_circuit_breaker("maestro")
        assert breaker.state == CircuitState.OPEN, "Circuit breaker should be OPEN"

        print("\n✓ VALIDATION: Circuit breaker trips on failure")

    @pytest.mark.asyncio
    async def test_validation_graceful_degradation(self):
        """VALIDATION: System degrades gracefully with reduced agents"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        # Fail multiple agents
        await orchestrator.simulate_agent_failure("maestro", "Test")
        await orchestrator.simulate_agent_failure("opus", "Test")
        await orchestrator.simulate_agent_failure("claude", "Test")

        # System should still operate
        assert orchestrator.can_degrade_gracefully() is True
        assert len(orchestrator.get_operational_agents()) == 6

        print("\n✓ VALIDATION: Graceful degradation works")

    @pytest.mark.asyncio
    async def test_validation_gate_all_agents_ready(self):
        """GATE: All agents in READY state"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        # Verify all agents reached READY
        assert orchestrator._states_ready is True, "All agents should be in READY state"
        assert orchestrator.state_machine.all_agents_ready() is True

        # Verify each agent is in READY state
        for agent_id in orchestrator.get_agent_ids():
            state = orchestrator.get_agent_lifecycle_state(agent_id)
            assert state.state == AgentState.READY, \
                f"Agent {agent_id} should be READY, got {state.state.value}"

        # Full readiness check (Integration 2A + 2B)
        assert orchestrator.readiness() is True, "Full readiness check should pass"

        print("\n" + "=" * 60)
        print("GATE PASSED: All agents in READY state")
        print("Integration 2B Complete")
        print("=" * 60)


class TestIntegration2BComplete:
    """
    Complete integration test running full lifecycle.
    Simulates real-world startup sequence.
    """

    @pytest.mark.asyncio
    async def test_full_lifecycle_sequence(self):
        """
        Full integration test:
        1. Initialize orchestrator
        2. Start services (health monitoring + state advancement)
        3. Verify all agents READY
        4. Simulate failures
        5. Verify graceful degradation
        6. Recover agents
        7. Verify full recovery
        """
        print("\n" + "=" * 60)
        print("Integration 2B - Full Lifecycle Test")
        print("=" * 60)

        # Setup
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        circuit_config = CircuitBreakerConfig(failure_threshold=3)

        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config,
            circuit_config=circuit_config
        )

        # Step 1: Initialize
        print("\n1. Initializing orchestrator...")
        await orchestrator.initialize()
        assert orchestrator._initialized is True
        print("   ✓ Orchestrator initialized")

        # Step 2: Start services
        print("\n2. Starting services (health + state machine)...")
        await orchestrator.start()
        assert orchestrator._ready is True
        assert orchestrator._states_ready is True
        print("   ✓ Services started")

        # Step 3: Verify all READY
        print("\n3. Verifying all agents in READY state...")
        status = orchestrator.get_health_status()
        assert status["all_agents_ready"] is True
        print(f"   ✓ {status['state_machine']['ready_agents']}/9 agents READY")

        # Step 4: Simulate failures
        print("\n4. Simulating agent failures...")
        await orchestrator.simulate_agent_failure("maestro", "Test failure 1")
        await orchestrator.simulate_agent_failure("opus", "Test failure 2")

        failed = orchestrator.get_failed_agents()
        print(f"   ✓ Failed agents: {failed}")
        assert len(failed) == 2

        # Step 5: Verify graceful degradation
        print("\n5. Verifying graceful degradation...")
        degradation = orchestrator.get_degradation_status()
        assert degradation["can_operate"] is True
        print(f"   ✓ Operational: {degradation['operational_agents']}/9")
        print(f"   ✓ Degradation level: {degradation['degradation_level']:.1%}")

        # Step 6: Recover agents
        print("\n6. Recovering failed agents...")
        await orchestrator.recover_agent("maestro")
        await orchestrator.recover_agent("opus")

        operational = orchestrator.get_operational_agents()
        print(f"   ✓ Recovered. Operational: {len(operational)}/9")

        # Step 7: Verify full recovery
        print("\n7. Verifying full recovery...")
        assert orchestrator._states_ready is True
        assert orchestrator.readiness() is True
        print("   ✓ Full readiness restored")

        print("\n" + "=" * 60)
        print("Integration 2B - Full Lifecycle Test PASSED")
        print("=" * 60)
