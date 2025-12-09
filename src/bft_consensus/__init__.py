# src/bft_consensus/__init__.py
"""
BFT Consensus Module - Byzantine Fault Tolerant Consensus
Integration 2C: PBFT-style consensus for AAOS ensemble agents

Implements simplified PBFT with phases:
- PROPOSE: Leader proposes a value
- PREPARE: Agents send prepare votes
- COMMIT: Agents send commit votes
- DECIDE: Consensus reached when quorum achieved
"""

from .consensus import (
    ConsensusManager,
    ConsensusConfig,
    ConsensusPhase,
    ConsensusRound,
    ConsensusMessage,
    ConsensusResult,
)
from .quorum import QuorumCalculator, QuorumStatus

__all__ = [
    # Core consensus
    "ConsensusManager",
    "ConsensusConfig",
    "ConsensusPhase",
    "ConsensusRound",
    "ConsensusMessage",
    "ConsensusResult",
    # Quorum
    "QuorumCalculator",
    "QuorumStatus",
]
