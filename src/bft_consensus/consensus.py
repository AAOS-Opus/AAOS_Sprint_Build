# src/bft_consensus/consensus.py
"""
Consensus Manager - PBFT-style consensus implementation
Integration 2C: Byzantine Fault Tolerant consensus for AAOS ensemble

Simplified PBFT Protocol:
1. PROPOSE: Leader (maestro) proposes a value
2. PREPARE: Agents send prepare votes (need quorum)
3. COMMIT: Agents send commit votes (need quorum)
4. DECIDE: Consensus reached, decision finalized

Quorum: 5 out of 9 agents (tolerates f=2 Byzantine faults)
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Set, Callable
from dataclasses import dataclass, field
import uuid

from .quorum import QuorumCalculator, QuorumStatus

logger = logging.getLogger("aaos.bft.consensus")


class ConsensusPhase(str, Enum):
    """PBFT consensus phases"""
    IDLE = "idle"              # No active consensus round
    PROPOSE = "propose"        # Leader proposing value
    PREPARE = "prepare"        # Collecting prepare votes
    COMMIT = "commit"          # Collecting commit votes
    DECIDE = "decide"          # Decision finalized
    FAILED = "failed"          # Consensus failed


@dataclass
class ConsensusConfig:
    """Configuration for consensus manager"""
    total_agents: int = 9                    # n = total agents
    max_byzantine_faults: int = 2            # f = max Byzantine faults
    propose_timeout: float = 5.0             # Timeout for propose phase
    prepare_timeout: float = 10.0            # Timeout for prepare phase
    commit_timeout: float = 10.0             # Timeout for commit phase
    leader_id: str = "maestro"               # Default leader


@dataclass
class ConsensusMessage:
    """A message in the consensus protocol"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    round_id: str = ""
    phase: ConsensusPhase = ConsensusPhase.IDLE
    sender_id: str = ""
    value: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "round_id": self.round_id,
            "phase": self.phase.value,
            "sender_id": self.sender_id,
            "value": str(self.value) if self.value else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConsensusResult:
    """Result of a completed consensus round"""
    round_id: str
    success: bool
    value: Any
    phase_reached: ConsensusPhase
    prepare_votes: int
    commit_votes: int
    participants: List[str]
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "success": self.success,
            "value": str(self.value) if self.value else None,
            "phase_reached": self.phase_reached.value,
            "prepare_votes": self.prepare_votes,
            "commit_votes": self.commit_votes,
            "participants": self.participants,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConsensusRound:
    """State of a single consensus round"""
    round_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    phase: ConsensusPhase = ConsensusPhase.IDLE
    proposed_value: Any = None
    leader_id: str = ""

    # Vote tracking
    prepare_votes: Set[str] = field(default_factory=set)
    commit_votes: Set[str] = field(default_factory=set)

    # Timing
    started_at: Optional[datetime] = None
    phase_started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Message history
    messages: List[ConsensusMessage] = field(default_factory=list)

    def add_prepare_vote(self, agent_id: str) -> bool:
        """Add a prepare vote. Returns True if new vote."""
        if agent_id in self.prepare_votes:
            return False
        self.prepare_votes.add(agent_id)
        return True

    def add_commit_vote(self, agent_id: str) -> bool:
        """Add a commit vote. Returns True if new vote."""
        if agent_id in self.commit_votes:
            return False
        self.commit_votes.add(agent_id)
        return True

    def get_participants(self) -> List[str]:
        """Get all agents that participated in this round"""
        participants = set()
        participants.update(self.prepare_votes)
        participants.update(self.commit_votes)
        if self.leader_id:
            participants.add(self.leader_id)
        return sorted(participants)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "phase": self.phase.value,
            "proposed_value": str(self.proposed_value) if self.proposed_value else None,
            "leader_id": self.leader_id,
            "prepare_votes": len(self.prepare_votes),
            "commit_votes": len(self.commit_votes),
            "prepare_voters": sorted(self.prepare_votes),
            "commit_voters": sorted(self.commit_votes),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class ConsensusManager:
    """
    Manages PBFT-style consensus for the AAOS ensemble.

    Integrates with LifecycleOrchestrator to:
    - Only allow READY agents to participate
    - Use health status for Byzantine fault detection
    - Maintain quorum requirements (5 out of 9)
    """

    def __init__(
        self,
        config: Optional[ConsensusConfig] = None,
        lifecycle_orchestrator: Optional[Any] = None
    ):
        self.config = config or ConsensusConfig()
        self.lifecycle_orchestrator = lifecycle_orchestrator

        # Quorum calculator
        self.quorum = QuorumCalculator(
            total_nodes=self.config.total_agents,
            max_byzantine_faults=self.config.max_byzantine_faults
        )

        # State
        self.current_round: Optional[ConsensusRound] = None
        self.completed_rounds: List[ConsensusResult] = []
        self._initialized = False
        self._operational = False

        # Callbacks
        self._phase_callbacks: List[Callable] = []
        self._decision_callbacks: List[Callable] = []

        # Registered participants (set by lifecycle orchestrator)
        self._registered_agents: Set[str] = set()

        logger.info(
            f"ConsensusManager created: n={self.config.total_agents}, "
            f"f={self.config.max_byzantine_faults}, quorum={self.quorum.quorum_size}"
        )

    async def initialize(self, agent_ids: Optional[List[str]] = None):
        """Initialize the consensus manager with registered agents"""
        if self._initialized:
            return

        if agent_ids:
            self._registered_agents = set(agent_ids)
        elif self.lifecycle_orchestrator:
            # Get agents from lifecycle orchestrator
            self._registered_agents = set(self.lifecycle_orchestrator.get_agent_ids())
        else:
            # Default ensemble agents
            self._registered_agents = {
                "maestro", "opus", "claude", "devzen",
                "frontend", "backend", "kimi", "scout", "dr-aeon"
            }

        self._initialized = True
        logger.info(f"ConsensusManager initialized with {len(self._registered_agents)} agents")

    def get_ready_participants(self) -> List[str]:
        """
        Get agents that can participate in consensus (READY state only).
        Integration 2C requirement: Only READY agents participate.
        """
        if self.lifecycle_orchestrator:
            # Use lifecycle orchestrator to get operational agents
            return self.lifecycle_orchestrator.get_operational_agents()
        else:
            # Return all registered agents if no orchestrator
            return sorted(self._registered_agents)

    def get_participant_count(self) -> int:
        """Get count of eligible participants"""
        return len(self.get_ready_participants())

    def can_reach_quorum(self) -> bool:
        """Check if we have enough READY agents to reach quorum"""
        ready_count = self.get_participant_count()
        return ready_count >= self.quorum.quorum_size

    # =========================================================================
    # Consensus Protocol Implementation
    # =========================================================================

    async def start_round(self, value: Any, leader_id: Optional[str] = None) -> ConsensusRound:
        """
        Start a new consensus round (PROPOSE phase).

        Args:
            value: The value to reach consensus on
            leader_id: The leader proposing the value (default: maestro)

        Returns:
            The new ConsensusRound
        """
        if self.current_round and self.current_round.phase not in (
            ConsensusPhase.IDLE, ConsensusPhase.DECIDE, ConsensusPhase.FAILED
        ):
            raise RuntimeError("Cannot start new round while another is in progress")

        if not self.can_reach_quorum():
            raise RuntimeError(
                f"Cannot start consensus: only {self.get_participant_count()} READY agents, "
                f"need {self.quorum.quorum_size} for quorum"
            )

        leader = leader_id or self.config.leader_id

        # Verify leader is READY
        ready_participants = self.get_ready_participants()
        if leader not in ready_participants:
            raise RuntimeError(f"Leader {leader} is not in READY state")

        # Create new round
        self.current_round = ConsensusRound(
            phase=ConsensusPhase.PROPOSE,
            proposed_value=value,
            leader_id=leader,
            started_at=datetime.utcnow(),
            phase_started_at=datetime.utcnow(),
        )

        # Record propose message
        propose_msg = ConsensusMessage(
            round_id=self.current_round.round_id,
            phase=ConsensusPhase.PROPOSE,
            sender_id=leader,
            value=value,
        )
        self.current_round.messages.append(propose_msg)

        logger.info(
            f"Round {self.current_round.round_id} started: "
            f"leader={leader}, value={value}"
        )

        await self._notify_phase_change(ConsensusPhase.PROPOSE)
        return self.current_round

    async def submit_prepare_vote(self, agent_id: str) -> bool:
        """
        Submit a prepare vote for the current round.

        Args:
            agent_id: The agent submitting the vote

        Returns:
            True if vote was accepted
        """
        if not self.current_round:
            logger.warning(f"No active round for prepare vote from {agent_id}")
            return False

        if self.current_round.phase not in (ConsensusPhase.PROPOSE, ConsensusPhase.PREPARE):
            logger.warning(f"Cannot accept prepare vote in phase {self.current_round.phase.value}")
            return False

        # Verify agent is READY
        if agent_id not in self.get_ready_participants():
            logger.warning(f"Agent {agent_id} not READY, rejecting prepare vote")
            return False

        # Transition to PREPARE phase if still in PROPOSE
        if self.current_round.phase == ConsensusPhase.PROPOSE:
            self.current_round.phase = ConsensusPhase.PREPARE
            self.current_round.phase_started_at = datetime.utcnow()
            await self._notify_phase_change(ConsensusPhase.PREPARE)

        # Add vote
        if self.current_round.add_prepare_vote(agent_id):
            # Record prepare message
            prepare_msg = ConsensusMessage(
                round_id=self.current_round.round_id,
                phase=ConsensusPhase.PREPARE,
                sender_id=agent_id,
                value=self.current_round.proposed_value,
            )
            self.current_round.messages.append(prepare_msg)

            logger.info(
                f"Prepare vote from {agent_id}: "
                f"{len(self.current_round.prepare_votes)}/{self.quorum.quorum_size}"
            )

            # Check if we have prepare quorum
            if self.quorum.has_quorum(len(self.current_round.prepare_votes)):
                logger.info(
                    f"Prepare quorum achieved: {len(self.current_round.prepare_votes)} votes"
                )
                # Auto-advance to commit phase
                self.current_round.phase = ConsensusPhase.COMMIT
                self.current_round.phase_started_at = datetime.utcnow()
                await self._notify_phase_change(ConsensusPhase.COMMIT)

            return True

        return False

    async def submit_commit_vote(self, agent_id: str) -> bool:
        """
        Submit a commit vote for the current round.

        Args:
            agent_id: The agent submitting the vote

        Returns:
            True if vote was accepted
        """
        if not self.current_round:
            logger.warning(f"No active round for commit vote from {agent_id}")
            return False

        if self.current_round.phase not in (ConsensusPhase.PREPARE, ConsensusPhase.COMMIT):
            logger.warning(f"Cannot accept commit vote in phase {self.current_round.phase.value}")
            return False

        # Verify agent is READY
        if agent_id not in self.get_ready_participants():
            logger.warning(f"Agent {agent_id} not READY, rejecting commit vote")
            return False

        # Must have sent prepare vote first
        if agent_id not in self.current_round.prepare_votes:
            logger.warning(f"Agent {agent_id} must prepare before commit")
            return False

        # Transition to COMMIT phase if still in PREPARE
        if self.current_round.phase == ConsensusPhase.PREPARE:
            self.current_round.phase = ConsensusPhase.COMMIT
            self.current_round.phase_started_at = datetime.utcnow()
            await self._notify_phase_change(ConsensusPhase.COMMIT)

        # Add vote
        if self.current_round.add_commit_vote(agent_id):
            # Record commit message
            commit_msg = ConsensusMessage(
                round_id=self.current_round.round_id,
                phase=ConsensusPhase.COMMIT,
                sender_id=agent_id,
                value=self.current_round.proposed_value,
            )
            self.current_round.messages.append(commit_msg)

            logger.info(
                f"Commit vote from {agent_id}: "
                f"{len(self.current_round.commit_votes)}/{self.quorum.quorum_size}"
            )

            # Check if we have commit quorum
            if self.quorum.has_quorum(len(self.current_round.commit_votes)):
                logger.info(
                    f"Commit quorum achieved: {len(self.current_round.commit_votes)} votes"
                )
                # Finalize decision
                await self._finalize_decision()

            return True

        return False

    async def _finalize_decision(self):
        """Finalize the consensus decision (DECIDE phase)"""
        if not self.current_round:
            return

        self.current_round.phase = ConsensusPhase.DECIDE
        self.current_round.completed_at = datetime.utcnow()

        # Calculate duration
        duration_ms = 0
        if self.current_round.started_at:
            duration_ms = (
                self.current_round.completed_at - self.current_round.started_at
            ).total_seconds() * 1000

        # Create result
        result = ConsensusResult(
            round_id=self.current_round.round_id,
            success=True,
            value=self.current_round.proposed_value,
            phase_reached=ConsensusPhase.DECIDE,
            prepare_votes=len(self.current_round.prepare_votes),
            commit_votes=len(self.current_round.commit_votes),
            participants=self.current_round.get_participants(),
            duration_ms=duration_ms,
        )
        self.completed_rounds.append(result)

        logger.info(
            f"Round {self.current_round.round_id} DECIDED: "
            f"value={self.current_round.proposed_value}, "
            f"prepare={len(self.current_round.prepare_votes)}, "
            f"commit={len(self.current_round.commit_votes)}, "
            f"duration={duration_ms:.1f}ms"
        )

        self._operational = True
        await self._notify_phase_change(ConsensusPhase.DECIDE)
        await self._notify_decision(result)

    async def run_full_round(self, value: Any, leader_id: Optional[str] = None) -> ConsensusResult:
        """
        Run a complete consensus round with automatic voting.

        This simulates all READY agents voting in sequence.
        Used for testing and initialization.

        Args:
            value: The value to reach consensus on
            leader_id: The leader proposing the value

        Returns:
            ConsensusResult with the outcome
        """
        # Start round
        await self.start_round(value, leader_id)

        # Get ready participants
        participants = self.get_ready_participants()

        # All participants send prepare votes
        for agent_id in participants:
            await self.submit_prepare_vote(agent_id)
            if self.current_round.phase == ConsensusPhase.COMMIT:
                break  # Quorum reached

        # All prepared agents send commit votes
        for agent_id in sorted(self.current_round.prepare_votes):
            await self.submit_commit_vote(agent_id)
            if self.current_round.phase == ConsensusPhase.DECIDE:
                break  # Quorum reached

        # Return result
        if self.completed_rounds:
            return self.completed_rounds[-1]
        else:
            # Create failure result
            duration_ms = 0
            if self.current_round and self.current_round.started_at:
                duration_ms = (datetime.utcnow() - self.current_round.started_at).total_seconds() * 1000

            return ConsensusResult(
                round_id=self.current_round.round_id if self.current_round else "unknown",
                success=False,
                value=value,
                phase_reached=self.current_round.phase if self.current_round else ConsensusPhase.FAILED,
                prepare_votes=len(self.current_round.prepare_votes) if self.current_round else 0,
                commit_votes=len(self.current_round.commit_votes) if self.current_round else 0,
                participants=self.current_round.get_participants() if self.current_round else [],
                duration_ms=duration_ms,
            )

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_phase_change(self, callback: Callable):
        """Register callback for phase changes"""
        self._phase_callbacks.append(callback)

    def on_decision(self, callback: Callable):
        """Register callback for decisions"""
        self._decision_callbacks.append(callback)

    async def _notify_phase_change(self, phase: ConsensusPhase):
        """Notify callbacks of phase change"""
        for callback in self._phase_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.current_round, phase)
                else:
                    callback(self.current_round, phase)
            except Exception as e:
                logger.error(f"Phase callback error: {e}")

    async def _notify_decision(self, result: ConsensusResult):
        """Notify callbacks of decision"""
        for callback in self._decision_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Decision callback error: {e}")

    # =========================================================================
    # Status and Health
    # =========================================================================

    def is_operational(self) -> bool:
        """Check if consensus has completed at least one successful round"""
        return self._operational

    def readiness(self) -> bool:
        """
        GATE check: Returns True if consensus is operational.
        Requires at least one successful round to have completed.
        """
        if not self._initialized:
            return False
        if not self._operational:
            return False
        if not self.can_reach_quorum():
            return False
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive consensus status"""
        ready_participants = self.get_ready_participants()

        return {
            "initialized": self._initialized,
            "operational": self._operational,
            "can_reach_quorum": self.can_reach_quorum(),
            "quorum": self.quorum.to_dict(),
            "ready_participants": len(ready_participants),
            "ready_participant_ids": ready_participants,
            "current_round": self.current_round.to_dict() if self.current_round else None,
            "completed_rounds": len(self.completed_rounds),
            "last_decision": self.completed_rounds[-1].to_dict() if self.completed_rounds else None,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for /health/consensus endpoint"""
        ready_count = self.get_participant_count()
        quorum_status = self.quorum.get_status(ready_count, ready_count)

        return {
            "ready": self.readiness(),
            "phase": self.current_round.phase.value if self.current_round else "idle",
            "quorum_status": quorum_status.value,
            "ready_agents": ready_count,
            "quorum_requirement": self.quorum.quorum_size,
            "operational": self._operational,
            "rounds_completed": len(self.completed_rounds),
            "timestamp": datetime.utcnow().isoformat(),
        }
