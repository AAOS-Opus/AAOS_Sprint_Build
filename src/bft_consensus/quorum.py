# src/bft_consensus/quorum.py
"""
Quorum Calculator - BFT quorum calculation and verification
Integration 2C: Determines quorum requirements for Byzantine Fault Tolerance

BFT Formula:
- n = total nodes
- f = max Byzantine faults tolerated
- quorum = 2f + 1 (for safety)
- n >= 3f + 1 (for liveness)

For n=9 agents:
- f = floor((n-1)/3) = floor(8/3) = 2
- quorum = 2*2 + 1 = 5
"""

import logging
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger("aaos.bft.quorum")


class QuorumStatus(str, Enum):
    """Status of quorum achievement"""
    NOT_STARTED = "not_started"    # No votes collected yet
    COLLECTING = "collecting"       # Votes being collected
    ACHIEVED = "achieved"           # Quorum reached
    FAILED = "failed"               # Cannot reach quorum (too many failures)


@dataclass
class QuorumCalculator:
    """
    Calculates and tracks BFT quorum requirements.

    For n=9 agents with f=2 Byzantine tolerance:
    - quorum = 5 (majority of non-Byzantine nodes)
    """

    total_nodes: int = 9
    max_byzantine_faults: int = 2

    def __post_init__(self):
        # Validate BFT requirements: n >= 3f + 1
        min_nodes = 3 * self.max_byzantine_faults + 1
        if self.total_nodes < min_nodes:
            raise ValueError(
                f"BFT requires n >= 3f + 1. "
                f"For f={self.max_byzantine_faults}, need at least {min_nodes} nodes, got {self.total_nodes}"
            )

        # Calculate quorum: 2f + 1
        self._quorum_size = 2 * self.max_byzantine_faults + 1
        logger.info(
            f"QuorumCalculator initialized: n={self.total_nodes}, "
            f"f={self.max_byzantine_faults}, quorum={self._quorum_size}"
        )

    @property
    def quorum_size(self) -> int:
        """Minimum votes needed for quorum (2f + 1)"""
        return self._quorum_size

    @property
    def f(self) -> int:
        """Maximum Byzantine faults tolerated"""
        return self.max_byzantine_faults

    @property
    def n(self) -> int:
        """Total number of nodes"""
        return self.total_nodes

    def has_quorum(self, votes: int) -> bool:
        """Check if we have enough votes for quorum"""
        return votes >= self._quorum_size

    def votes_needed(self, current_votes: int) -> int:
        """Calculate how many more votes needed for quorum"""
        remaining = self._quorum_size - current_votes
        return max(0, remaining)

    def can_reach_quorum(self, current_votes: int, potential_voters: int) -> bool:
        """
        Check if quorum can still be reached.

        Args:
            current_votes: Votes already collected
            potential_voters: Remaining agents that could vote

        Returns:
            True if quorum is still achievable
        """
        max_possible = current_votes + potential_voters
        return max_possible >= self._quorum_size

    def get_status(self, current_votes: int, total_eligible: int) -> QuorumStatus:
        """
        Get current quorum status.

        Args:
            current_votes: Votes collected so far
            total_eligible: Total eligible voters (READY agents)

        Returns:
            QuorumStatus enum
        """
        if current_votes == 0:
            return QuorumStatus.NOT_STARTED

        if self.has_quorum(current_votes):
            return QuorumStatus.ACHIEVED

        # Calculate remaining potential voters
        remaining_voters = total_eligible - current_votes

        if self.can_reach_quorum(current_votes, remaining_voters):
            return QuorumStatus.COLLECTING
        else:
            return QuorumStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Return quorum configuration as dict"""
        return {
            "total_nodes": self.total_nodes,
            "max_byzantine_faults": self.max_byzantine_faults,
            "quorum_size": self._quorum_size,
            "formula": f"quorum = 2f + 1 = 2*{self.max_byzantine_faults} + 1 = {self._quorum_size}",
        }


# Default quorum calculator for 9 agents with f=2
DEFAULT_QUORUM = QuorumCalculator(total_nodes=9, max_byzantine_faults=2)
