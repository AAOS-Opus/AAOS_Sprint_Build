# agents/trading/__init__.py
"""
AAOS Trading Agents for Aurora TA Integration

Agents:
  - TradingAnalystAgent: Claims trading_analysis tasks, calls Sovereign for analysis
  - SignalValidatorAgent: Claims trading_signal tasks, validates against Hidden Hand methodology

Defensive Measures Implemented:
  - FM-TAW-004: Capability-based task matching (sovereign_access, hidden_hand)
  - FM-TAW-006: Fallback distrust enforcement (0.8x confidence multiplier)
  - FM-TAW-012: Hidden Hand signal validation (Wyckoff phase cross-check)
"""

from .agent_config import (
    TradingAgentConfig,
    TRADING_ANALYST_CAPABILITIES,
    SIGNAL_VALIDATOR_CAPABILITIES,
    FALLBACK_CONFIDENCE_MULTIPLIER,
)
from .trading_analyst import TradingAnalystAgent
from .signal_validator import SignalValidatorAgent

__all__ = [
    "TradingAgentConfig",
    "TradingAnalystAgent",
    "SignalValidatorAgent",
    "TRADING_ANALYST_CAPABILITIES",
    "SIGNAL_VALIDATOR_CAPABILITIES",
    "FALLBACK_CONFIDENCE_MULTIPLIER",
]
