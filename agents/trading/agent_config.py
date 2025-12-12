# agents/trading/agent_config.py
"""
Trading Agent Configuration

Centralized configuration for trading agents with defensive measure parameters.
Implements FM-TAW-004 (capability matching) and FM-TAW-006 (fallback distrust).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any


# =============================================================================
# Environment Configuration
# =============================================================================

# AAOS Orchestrator
AAOS_WS_URL = os.getenv("AAOS_WS_URL", "ws://localhost:8000/ws")
AAOS_API_KEY = os.getenv("AAOS_API_KEY", "")

# Sovereign LLM
SOVEREIGN_URL = os.getenv("SOVEREIGN_URL", "http://localhost:11434/v1")
SOVEREIGN_TIMEOUT = int(os.getenv("SOVEREIGN_TIMEOUT", "120"))
SOVEREIGN_MODEL_PRIMARY = os.getenv("SOVEREIGN_MODEL_PRIMARY", "qwen2.5-coder:32b")
SOVEREIGN_MODEL_FALLBACK = os.getenv("SOVEREIGN_MODEL_FALLBACK", "qwen2.5-coder:7b")

# Redis (for processing_tasks set - FM-TAW-007)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


# =============================================================================
# Capability Definitions (FM-TAW-004)
# =============================================================================

# Trading Analyst: Can call Sovereign for analysis
TRADING_ANALYST_CAPABILITIES: Dict[str, bool] = {
    "sovereign_access": True,
    "trading_analysis": True,
}

# Signal Validator: Can call Sovereign AND validate against Hidden Hand
SIGNAL_VALIDATOR_CAPABILITIES: Dict[str, bool] = {
    "sovereign_access": True,
    "trading_signal": True,
    "hidden_hand": True,
}


# =============================================================================
# Task Type Claims
# =============================================================================

# Which task types each agent claims
TRADING_ANALYST_TASK_TYPES: List[str] = ["trading_analysis"]
SIGNAL_VALIDATOR_TASK_TYPES: List[str] = ["trading_signal"]


# =============================================================================
# Defensive Measure Parameters
# =============================================================================

# FM-TAW-006: Fallback distrust - reduce confidence when 7B model responds
FALLBACK_CONFIDENCE_MULTIPLIER = 0.8

# FM-TAW-007: Processing set tracking
PROCESSING_TASKS_KEY = "processing_tasks"

# FM-TAW-008: Result delivery ACK (future implementation)
TASK_ACK_TIMEOUT = 10  # seconds
TASK_ACK_RETRIES = 3

# Heartbeat configuration
HEARTBEAT_INTERVAL = 15  # seconds
HEARTBEAT_TIMEOUT = 45   # miss 3 = disconnect


# =============================================================================
# Dataclass Configuration
# =============================================================================

@dataclass
class TradingAgentConfig:
    """
    Configuration for trading agents.

    Loads from environment variables with sensible defaults.
    Used by both TradingAnalystAgent and SignalValidatorAgent.
    """

    # Agent identity
    agent_id: str = field(default="")
    agent_type: str = field(default="trading")

    # AAOS connection
    ws_url: str = field(default_factory=lambda: AAOS_WS_URL)
    api_key: str = field(default_factory=lambda: AAOS_API_KEY)

    # Sovereign LLM
    sovereign_url: str = field(default_factory=lambda: SOVEREIGN_URL)
    sovereign_timeout: int = field(default_factory=lambda: SOVEREIGN_TIMEOUT)

    # Redis
    redis_host: str = field(default_factory=lambda: REDIS_HOST)
    redis_port: int = field(default_factory=lambda: REDIS_PORT)

    # Heartbeat
    heartbeat_interval: int = HEARTBEAT_INTERVAL
    heartbeat_timeout: int = HEARTBEAT_TIMEOUT

    # Task handling
    task_ack_timeout: int = TASK_ACK_TIMEOUT
    task_ack_retries: int = TASK_ACK_RETRIES

    # Defensive measures
    fallback_confidence_multiplier: float = FALLBACK_CONFIDENCE_MULTIPLIER

    # Capabilities (set by subclass)
    capabilities: Dict[str, bool] = field(default_factory=dict)

    # Task types to claim (set by subclass)
    task_types: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Generate agent_id if not provided"""
        if not self.agent_id:
            import uuid
            self.agent_id = f"{self.agent_type}_{uuid.uuid4().hex[:8]}"


@dataclass
class TradingAnalystConfig(TradingAgentConfig):
    """Configuration specific to Trading Analyst agent"""

    agent_type: str = field(default="trading_analyst")
    capabilities: Dict[str, bool] = field(
        default_factory=lambda: TRADING_ANALYST_CAPABILITIES.copy()
    )
    task_types: List[str] = field(
        default_factory=lambda: TRADING_ANALYST_TASK_TYPES.copy()
    )


@dataclass
class SignalValidatorConfig(TradingAgentConfig):
    """Configuration specific to Signal Validator agent"""

    agent_type: str = field(default="signal_validator")
    capabilities: Dict[str, bool] = field(
        default_factory=lambda: SIGNAL_VALIDATOR_CAPABILITIES.copy()
    )
    task_types: List[str] = field(
        default_factory=lambda: SIGNAL_VALIDATOR_TASK_TYPES.copy()
    )


# =============================================================================
# Wyckoff Phase Definitions (FM-TAW-012)
# =============================================================================

class WyckoffPhase:
    """Wyckoff market phases for Hidden Hand validation"""
    ACCUMULATION = "accumulation"
    MARKUP = "markup"
    DISTRIBUTION = "distribution"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class VolumeProfile:
    """Volume trend indicators"""
    RISING = "rising"
    DECLINING = "declining"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


# Phase-signal conflict multipliers (FM-TAW-012)
PHASE_SIGNAL_MULTIPLIERS = {
    # (phase, signal) -> (is_suspicious, confidence_multiplier, reason)
    (WyckoffPhase.DISTRIBUTION, "BUY"): {
        VolumeProfile.DECLINING: (True, 0.5, "BUY during distribution with declining volume - likely upthrust trap"),
        VolumeProfile.RISING: (True, 0.8, "BUY during distribution with rising volume - cautious proceed"),
        VolumeProfile.NEUTRAL: (True, 0.7, "BUY during distribution - monitor for upthrust"),
        VolumeProfile.UNKNOWN: (True, 0.6, "BUY during distribution - insufficient volume data"),
    },
    (WyckoffPhase.ACCUMULATION, "SELL"): {
        VolumeProfile.DECLINING: (True, 0.5, "SELL during accumulation with declining volume - likely spring trap"),
        VolumeProfile.RISING: (True, 0.8, "SELL during accumulation with rising volume - cautious proceed"),
        VolumeProfile.NEUTRAL: (True, 0.7, "SELL during accumulation - monitor for spring"),
        VolumeProfile.UNKNOWN: (True, 0.6, "SELL during accumulation - insufficient volume data"),
    },
}
