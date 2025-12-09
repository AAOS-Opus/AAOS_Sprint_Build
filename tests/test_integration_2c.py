# tests/test_integration_2c.py
"""
Integration 2C Tests - BFT Consensus
Validates all requirements from layer-cake-integration-v1.md

STEPS:
1. Wire consensus manager to lifecycle orchestrator
2. Verify only READY agents participate in consensus
3. Configure quorum: n=9, f=2, quorum=5
4. Test propose phase (leader can propose)
5. Test prepare phase (5+ prepare votes collected)
6. Test commit phase (5+ commit votes collected)
7. Test decide phase (consensus reached)
8. Create health probe: GET /health/consensus

GATE: Consensus operational - full round completes successfully
"""

import pytest
import asyncio
from datetime import datetime

from src.agent_lifecycle import (
    HealthMonitorConfig,
    StateMachineConfig,
    LifecycleOrchestrator,
    AgentState,
)
from src.bft_consensus import (
    ConsensusManager,
    ConsensusConfig,
    ConsensusPhase,
    ConsensusRound,
    ConsensusMessage,
    ConsensusResult,
    QuorumCalculator,
    QuorumStatus,
)


class TestQuorumCalculator:
    """Tests for QuorumCalculator (Integration 2C Step 3)"""

    def test_quorum_configuration(self):
        """Quorum should be configured: n=9, f=2, quorum=5"""
        quorum = QuorumCalculator(total_nodes=9, max_byzantine_faults=2)

        assert quorum.n == 9
        assert quorum.f == 2
        assert quorum.quorum_size == 5  # 2*2 + 1 = 5

    def test_quorum_formula(self):
        """Quorum = 2f + 1"""
        quorum = QuorumCalculator(total_nodes=9, max_byzantine_faults=2)

        # Formula: 2*f + 1 = 2*2 + 1 = 5
        expected = 2 * quorum.f + 1
        assert quorum.quorum_size == expected

    def test_bft_requirement_validation(self):
        """n >= 3f + 1 must be satisfied"""
        # Valid: 9 >= 3*2 + 1 = 7
        quorum = QuorumCalculator(total_nodes=9, max_byzantine_faults=2)
        assert quorum is not None

        # Invalid: 5 < 3*2 + 1 = 7
        with pytest.raises(ValueError, match="BFT requires n >= 3f \\+ 1"):
            QuorumCalculator(total_nodes=5, max_byzantine_faults=2)

    def test_has_quorum(self):
        """has_quorum should correctly identify when quorum is reached"""
        quorum = QuorumCalculator(total_nodes=9, max_byzantine_faults=2)

        assert quorum.has_quorum(4) is False
        assert quorum.has_quorum(5) is True
        assert quorum.has_quorum(6) is True
        assert quorum.has_quorum(9) is True

    def test_votes_needed(self):
        """votes_needed should calculate remaining votes correctly"""
        quorum = QuorumCalculator(total_nodes=9, max_byzantine_faults=2)

        assert quorum.votes_needed(0) == 5
        assert quorum.votes_needed(3) == 2
        assert quorum.votes_needed(5) == 0
        assert quorum.votes_needed(7) == 0

    def test_quorum_status(self):
        """Quorum status should track progress correctly"""
        quorum = QuorumCalculator(total_nodes=9, max_byzantine_faults=2)

        assert quorum.get_status(0, 9) == QuorumStatus.NOT_STARTED
        assert quorum.get_status(3, 9) == QuorumStatus.COLLECTING
        assert quorum.get_status(5, 9) == QuorumStatus.ACHIEVED
        assert quorum.get_status(2, 2) == QuorumStatus.FAILED  # Can't reach 5 with only 2 more


class TestConsensusManager:
    """Tests for ConsensusManager component (Integration 2C)"""

    def test_consensus_manager_initialization(self):
        """Consensus manager should initialize without error"""
        config = ConsensusConfig(total_agents=9, max_byzantine_faults=2)
        cm = ConsensusManager(config=config)

        assert cm is not None
        assert cm.quorum.quorum_size == 5

    @pytest.mark.asyncio
    async def test_initialize_with_agents(self):
        """Consensus manager should track registered agents"""
        cm = ConsensusManager()
        agents = ["maestro", "opus", "claude", "devzen", "frontend", "backend", "kimi", "scout", "dr-aeon"]
        await cm.initialize(agents)

        assert cm._initialized is True
        assert len(cm._registered_agents) == 9


class TestProposePhase:
    """Tests for propose phase (Integration 2C Step 4)"""

    @pytest.mark.asyncio
    async def test_leader_can_propose(self):
        """Leader should be able to start a consensus round"""
        cm = ConsensusManager()
        agents = ["maestro", "opus", "claude", "devzen", "frontend", "backend", "kimi", "scout", "dr-aeon"]
        await cm.initialize(agents)

        round = await cm.start_round(value="test_value", leader_id="maestro")

        assert round is not None
        assert round.phase == ConsensusPhase.PROPOSE
        assert round.proposed_value == "test_value"
        assert round.leader_id == "maestro"

    @pytest.mark.asyncio
    async def test_propose_creates_message(self):
        """Propose should record a message"""
        cm = ConsensusManager()
        await cm.initialize()

        round = await cm.start_round(value="test", leader_id="maestro")

        assert len(round.messages) == 1
        assert round.messages[0].phase == ConsensusPhase.PROPOSE
        assert round.messages[0].sender_id == "maestro"


class TestPreparePhase:
    """Tests for prepare phase (Integration 2C Step 5)"""

    @pytest.mark.asyncio
    async def test_prepare_votes_collected(self):
        """Prepare votes should be collected from agents"""
        cm = ConsensusManager()
        await cm.initialize()

        await cm.start_round(value="test", leader_id="maestro")

        # Submit prepare votes
        await cm.submit_prepare_vote("maestro")
        await cm.submit_prepare_vote("opus")
        await cm.submit_prepare_vote("claude")

        assert len(cm.current_round.prepare_votes) == 3
        assert "maestro" in cm.current_round.prepare_votes
        assert "opus" in cm.current_round.prepare_votes
        assert "claude" in cm.current_round.prepare_votes

    @pytest.mark.asyncio
    async def test_prepare_quorum_advances_to_commit(self):
        """5+ prepare votes should advance to COMMIT phase"""
        cm = ConsensusManager()
        await cm.initialize()

        await cm.start_round(value="test", leader_id="maestro")

        # Submit 5 prepare votes (quorum)
        for agent in ["maestro", "opus", "claude", "devzen", "frontend"]:
            await cm.submit_prepare_vote(agent)

        assert len(cm.current_round.prepare_votes) >= 5
        assert cm.current_round.phase == ConsensusPhase.COMMIT

    @pytest.mark.asyncio
    async def test_duplicate_prepare_vote_rejected(self):
        """Duplicate prepare votes should be rejected"""
        cm = ConsensusManager()
        await cm.initialize()

        await cm.start_round(value="test", leader_id="maestro")

        result1 = await cm.submit_prepare_vote("maestro")
        result2 = await cm.submit_prepare_vote("maestro")  # Duplicate

        assert result1 is True
        assert result2 is False
        assert len(cm.current_round.prepare_votes) == 1


class TestCommitPhase:
    """Tests for commit phase (Integration 2C Step 6)"""

    @pytest.mark.asyncio
    async def test_commit_votes_collected(self):
        """Commit votes should be collected from prepared agents"""
        cm = ConsensusManager()
        await cm.initialize()

        await cm.start_round(value="test", leader_id="maestro")

        # Prepare phase
        for agent in ["maestro", "opus", "claude", "devzen", "frontend"]:
            await cm.submit_prepare_vote(agent)

        # Commit phase
        await cm.submit_commit_vote("maestro")
        await cm.submit_commit_vote("opus")
        await cm.submit_commit_vote("claude")

        assert len(cm.current_round.commit_votes) == 3

    @pytest.mark.asyncio
    async def test_commit_requires_prepare(self):
        """Agent must prepare before committing"""
        cm = ConsensusManager()
        await cm.initialize()

        await cm.start_round(value="test", leader_id="maestro")

        # Try to commit without preparing
        result = await cm.submit_commit_vote("maestro")

        assert result is False

    @pytest.mark.asyncio
    async def test_commit_quorum_triggers_decide(self):
        """5+ commit votes should trigger DECIDE phase"""
        cm = ConsensusManager()
        await cm.initialize()

        await cm.start_round(value="test", leader_id="maestro")

        # Prepare phase - all 9 agents
        for agent in ["maestro", "opus", "claude", "devzen", "frontend", "backend", "kimi", "scout", "dr-aeon"]:
            await cm.submit_prepare_vote(agent)

        # Commit phase - 5 agents (quorum)
        for agent in ["maestro", "opus", "claude", "devzen", "frontend"]:
            await cm.submit_commit_vote(agent)

        assert cm.current_round.phase == ConsensusPhase.DECIDE
        assert len(cm.completed_rounds) == 1


class TestDecidePhase:
    """Tests for decide phase (Integration 2C Step 7)"""

    @pytest.mark.asyncio
    async def test_consensus_reached(self):
        """Consensus should be reached with full round"""
        cm = ConsensusManager()
        await cm.initialize()

        result = await cm.run_full_round(value="test_decision", leader_id="maestro")

        assert result.success is True
        assert result.value == "test_decision"
        assert result.phase_reached == ConsensusPhase.DECIDE
        assert result.prepare_votes >= 5
        assert result.commit_votes >= 5

    @pytest.mark.asyncio
    async def test_decision_recorded(self):
        """Decisions should be recorded in completed_rounds"""
        cm = ConsensusManager()
        await cm.initialize()

        await cm.run_full_round(value="decision_1", leader_id="maestro")
        await cm.run_full_round(value="decision_2", leader_id="maestro")

        assert len(cm.completed_rounds) == 2
        assert cm.completed_rounds[0].value == "decision_1"
        assert cm.completed_rounds[1].value == "decision_2"

    @pytest.mark.asyncio
    async def test_consensus_becomes_operational(self):
        """Consensus manager should become operational after first round"""
        cm = ConsensusManager()
        await cm.initialize()

        assert cm.is_operational() is False

        await cm.run_full_round(value="test", leader_id="maestro")

        assert cm.is_operational() is True


class TestOnlyReadyAgentsParticipate:
    """Tests for verifying only READY agents participate (Integration 2C Step 2)"""

    @pytest.mark.asyncio
    async def test_failed_agent_cannot_vote(self):
        """Failed agents should not be able to participate"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        # Fail an agent
        await orchestrator.simulate_agent_failure("maestro", "Test failure")

        # Get operational agents (should not include maestro)
        operational = orchestrator.get_operational_agents()
        assert "maestro" not in operational
        assert len(operational) == 8

        # Verify consensus manager only sees READY agents
        cm = orchestrator.consensus_manager
        ready = cm.get_ready_participants()
        assert "maestro" not in ready

    @pytest.mark.asyncio
    async def test_consensus_uses_ready_agents_only(self):
        """Consensus should only use READY agents for voting"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        # Fail 2 agents (still have 7 > 5 quorum)
        await orchestrator.simulate_agent_failure("kimi", "Test")
        await orchestrator.simulate_agent_failure("scout", "Test")

        # Run another consensus round
        result = await orchestrator.run_consensus_round(value="after_failures")

        assert result["success"] is True
        # Participants should not include failed agents
        assert "kimi" not in result["participants"]
        assert "scout" not in result["participants"]


class TestConsensusHealthProbe:
    """Tests for consensus health probe (Integration 2C Step 8)"""

    @pytest.mark.asyncio
    async def test_consensus_health_status(self):
        """get_consensus_health should return health status"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        health = orchestrator.get_consensus_health()

        assert "ready" in health
        assert "phase" in health
        assert "quorum_status" in health
        assert "ready_agents" in health
        assert "quorum_requirement" in health
        assert "operational" in health
        assert "rounds_completed" in health

        assert health["ready"] is True
        assert health["operational"] is True
        assert health["quorum_requirement"] == 5
        assert health["ready_agents"] == 9

    @pytest.mark.asyncio
    async def test_consensus_status_in_health_status(self):
        """Consensus should be included in overall health status"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        status = orchestrator.get_health_status()

        assert "consensus_ready" in status
        assert "consensus" in status
        assert status["consensus_ready"] is True
        assert status["consensus"]["operational"] is True


class TestIntegration2CValidation:
    """
    Final validation tests for Integration 2C requirements.
    These tests verify the complete integration per workflow spec.
    """

    @pytest.mark.asyncio
    async def test_validation_consensus_sees_9_agents(self):
        """VALIDATION: Consensus manager sees 9 agents"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        cm = orchestrator.consensus_manager
        count = cm.get_participant_count()

        assert count == 9, f"Expected 9 agents, got {count}"
        print("\n✓ VALIDATION: Consensus sees 9 agents")

    @pytest.mark.asyncio
    async def test_validation_only_ready_participate(self):
        """VALIDATION: Only READY agents participate"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        # Fail an agent
        await orchestrator.simulate_agent_failure("dr-aeon", "Test")

        # Check participants
        cm = orchestrator.consensus_manager
        participants = cm.get_ready_participants()

        assert "dr-aeon" not in participants, "Failed agent should not participate"
        assert len(participants) == 8, "Should have 8 READY participants"

        print("\n✓ VALIDATION: Only READY agents participate")

    @pytest.mark.asyncio
    async def test_validation_quorum_configuration(self):
        """VALIDATION: Quorum configured as n=9, f=2, quorum=5"""
        cm = ConsensusManager(ConsensusConfig(total_agents=9, max_byzantine_faults=2))

        assert cm.quorum.n == 9, "n should be 9"
        assert cm.quorum.f == 2, "f should be 2"
        assert cm.quorum.quorum_size == 5, "quorum should be 5"

        print("\n✓ VALIDATION: Quorum configured (n=9, f=2, quorum=5)")

    @pytest.mark.asyncio
    async def test_validation_full_round_completes(self):
        """VALIDATION: Full round completes (propose → prepare → commit → decide)"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        # Verify initial round completed during startup
        cm = orchestrator.consensus_manager
        assert len(cm.completed_rounds) >= 1

        # Run another round
        result = await orchestrator.run_consensus_round(value="validation_test")

        assert result["success"] is True
        assert result["phase_reached"] == "decide"
        assert result["prepare_votes"] >= 5
        assert result["commit_votes"] >= 5

        print("\n✓ VALIDATION: Full round completes (propose → prepare → commit → decide)")

    @pytest.mark.asyncio
    async def test_validation_gate_consensus_operational(self):
        """GATE: Consensus operational - full round completes successfully"""
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)
        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )
        await orchestrator.initialize()
        await orchestrator.start()

        # GATE CHECKS
        assert orchestrator._consensus_ready is True, "Consensus should be ready"
        assert orchestrator.consensus_manager.is_operational() is True, "Consensus should be operational"
        assert orchestrator.readiness() is True, "Full readiness should pass"

        # Verify round completed
        cm = orchestrator.consensus_manager
        assert len(cm.completed_rounds) >= 1, "At least one round should complete"
        assert cm.completed_rounds[0].success is True, "First round should succeed"

        print("\n" + "=" * 60)
        print("GATE PASSED: Consensus operational - full round completes")
        print("Integration 2C Complete")
        print("=" * 60)


class TestIntegration2CComplete:
    """
    Complete integration test running full consensus lifecycle.
    """

    @pytest.mark.asyncio
    async def test_full_consensus_lifecycle(self):
        """
        Full integration test:
        1. Initialize orchestrator (2A + 2B + 2C)
        2. Verify consensus initialized
        3. Run consensus rounds
        4. Test with agent failures
        5. Verify graceful degradation
        """
        print("\n" + "=" * 60)
        print("Integration 2C - Full Consensus Lifecycle Test")
        print("=" * 60)

        # Setup
        health_config = HealthMonitorConfig(stabilization_time=0.1)
        state_config = StateMachineConfig(discovery_delay=0.01)

        orchestrator = LifecycleOrchestrator(
            health_config=health_config,
            state_config=state_config
        )

        # Step 1: Initialize and start
        print("\n1. Initializing orchestrator with consensus...")
        await orchestrator.initialize()
        await orchestrator.start()

        assert orchestrator._initialized is True
        assert orchestrator._states_ready is True
        assert orchestrator._consensus_ready is True
        print("   ✓ Orchestrator initialized with consensus")

        # Step 2: Verify consensus configuration
        print("\n2. Verifying consensus configuration...")
        cm = orchestrator.consensus_manager
        assert cm.quorum.n == 9
        assert cm.quorum.f == 2
        assert cm.quorum.quorum_size == 5
        print(f"   ✓ Quorum: n={cm.quorum.n}, f={cm.quorum.f}, quorum={cm.quorum.quorum_size}")

        # Step 3: Verify initial round completed
        print("\n3. Verifying initial consensus round...")
        assert len(cm.completed_rounds) >= 1
        initial_round = cm.completed_rounds[0]
        assert initial_round.success is True
        print(f"   ✓ Initial round: prepare={initial_round.prepare_votes}, commit={initial_round.commit_votes}")

        # Step 4: Run additional consensus rounds
        print("\n4. Running additional consensus rounds...")
        for i in range(3):
            result = await orchestrator.run_consensus_round(value=f"round_{i+2}")
            assert result["success"] is True
            print(f"   ✓ Round {i+2}: prepare={result['prepare_votes']}, commit={result['commit_votes']}")

        # Step 5: Test with agent failures
        print("\n5. Testing consensus with agent failures...")
        await orchestrator.simulate_agent_failure("kimi", "Test failure")
        await orchestrator.simulate_agent_failure("scout", "Test failure")

        # Should still work with 7 agents (> 5 quorum)
        result = await orchestrator.run_consensus_round(value="after_failures")
        assert result["success"] is True
        assert "kimi" not in result["participants"]
        assert "scout" not in result["participants"]
        print(f"   ✓ Consensus works with {len(result['participants'])} agents (2 failed)")

        # Step 6: Verify health status
        print("\n6. Verifying health status...")
        health = orchestrator.get_consensus_health()
        assert health["operational"] is True
        assert health["ready_agents"] == 7  # 9 - 2 failed
        print(f"   ✓ Health status: operational={health['operational']}, ready={health['ready_agents']}")

        # Step 7: Full readiness check
        print("\n7. Checking full readiness...")
        # Note: readiness will be False because not all agents are in READY state
        # But consensus is still operational
        assert orchestrator.consensus_manager.is_operational() is True
        assert orchestrator.can_reach_consensus_quorum() is True
        print("   ✓ Consensus operational and can reach quorum")

        print("\n" + "=" * 60)
        print("Integration 2C - Full Consensus Lifecycle Test PASSED")
        print("=" * 60)
