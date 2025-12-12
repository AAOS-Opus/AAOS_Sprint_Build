# agents/trading/signal_validator.py
"""
Signal Validator Agent

Claims trading_signal tasks from AAOS and validates signals against Hidden Hand methodology.
Cross-checks signals with Wyckoff market phases to detect institutional manipulation patterns.

Defensive Measures:
  - FM-TAW-004: Registers sovereign_access + hidden_hand capabilities on connect
  - FM-TAW-006: Applies 0.8x confidence multiplier when fallback model used
  - FM-TAW-012: Validates signals against Wyckoff phase (distribution+BUY, accumulation+SELL)
"""

import os
import sys
import json
import signal
import asyncio
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import websockets
import redis.asyncio as aioredis

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.integrations.sovereign_client import SovereignClient, SovereignClientError
from agents.trading.agent_config import (
    SignalValidatorConfig,
    FALLBACK_CONFIDENCE_MULTIPLIER,
    PROCESSING_TASKS_KEY,
    WyckoffPhase,
    VolumeProfile,
    PHASE_SIGNAL_MULTIPLIERS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("aaos.signal_validator")


class SignalValidatorAgent:
    """
    Signal Validator Agent for AAOS.

    Validates trading signals against Hidden Hand methodology.
    Cross-checks signals with Wyckoff market phases to detect
    institutional manipulation patterns (springs, upthrusts).
    """

    def __init__(self, config: Optional[SignalValidatorConfig] = None):
        self.config = config or SignalValidatorConfig()
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.sovereign: Optional[SovereignClient] = None
        self.redis: Optional[aioredis.Redis] = None
        self.running = False
        self.current_task_id: Optional[str] = None

        logger.info(
            f"SignalValidatorAgent initialized: agent_id={self.config.agent_id}, "
            f"capabilities={self.config.capabilities}"
        )

    async def connect(self) -> bool:
        """
        Connect to AAOS orchestrator via WebSocket.

        Returns True if connection and registration successful.
        """
        try:
            # Build headers with API key authentication
            headers = {}
            if self.config.api_key:
                headers["X-API-Key"] = self.config.api_key

            # Connect to WebSocket
            self.ws = await websockets.connect(
                self.config.ws_url,
                extra_headers=headers
            )

            logger.info(f"Connected to AAOS: {self.config.ws_url}")

            # Register with capabilities (FM-TAW-004)
            # Signal validator has both sovereign_access AND hidden_hand
            register_msg = {
                "type": "register",
                "agent_id": self.config.agent_id,
                "agent_type": self.config.agent_type,
                "capabilities": self.config.capabilities,
            }
            await self.ws.send(json.dumps(register_msg))

            # Wait for registration acknowledgment
            response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            ack = json.loads(response)

            if ack.get("type") == "registration_ack":
                logger.info(
                    f"Registered with AAOS: agent_id={self.config.agent_id}, "
                    f"capabilities={list(self.config.capabilities.keys())}"
                )
                return True
            else:
                logger.error(f"Unexpected registration response: {ack}")
                return False

        except asyncio.TimeoutError:
            logger.error("Registration timeout - no acknowledgment received")
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def initialize_services(self) -> bool:
        """Initialize Sovereign client and Redis connection"""
        try:
            # Initialize Sovereign client
            self.sovereign = SovereignClient()
            health = await self.sovereign.health_check()
            if health["status"] != "healthy":
                logger.warning(f"Sovereign health check warning: {health}")

            # Initialize Redis for processing_tasks tracking (FM-TAW-007)
            self.redis = aioredis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True
            )
            await self.redis.ping()

            logger.info("Services initialized: Sovereign and Redis connected")
            return True

        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            return False

    async def send_heartbeat(self):
        """Send periodic heartbeat to orchestrator"""
        while self.running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                if self.ws and self.running:
                    heartbeat = {
                        "type": "heartbeat",
                        "agent_id": self.config.agent_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await self.ws.send(json.dumps(heartbeat))
                    logger.debug(f"Heartbeat sent: {self.config.agent_id}")
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
                if not self.running:
                    break

    def validate_against_hidden_hand(
        self,
        signal: str,
        wyckoff_phase: str,
        volume_trend: str
    ) -> Tuple[bool, float, str]:
        """
        FM-TAW-012: Validate signal against Hidden Hand methodology.

        Cross-checks the trading signal against the current Wyckoff phase
        to detect potential institutional manipulation patterns.

        Args:
            signal: Trading signal (BUY, SELL, HOLD)
            wyckoff_phase: Current Wyckoff phase (accumulation, markup, distribution, markdown)
            volume_trend: Volume trend (rising, declining, neutral)

        Returns:
            Tuple of (is_valid, confidence_multiplier, reason)
            - is_valid: Whether the signal is valid (always True, but may need review)
            - confidence_multiplier: Factor to apply to confidence (0.5-1.0)
            - reason: Explanation of the validation result
        """
        signal = signal.upper()
        wyckoff_phase = wyckoff_phase.lower() if wyckoff_phase else WyckoffPhase.UNKNOWN
        volume_trend = volume_trend.lower() if volume_trend else VolumeProfile.UNKNOWN

        # Check for phase-signal conflicts in the lookup table
        phase_signal_key = (wyckoff_phase, signal)

        if phase_signal_key in PHASE_SIGNAL_MULTIPLIERS:
            volume_data = PHASE_SIGNAL_MULTIPLIERS[phase_signal_key]

            if volume_trend in volume_data:
                is_suspicious, multiplier, reason = volume_data[volume_trend]
                logger.warning(
                    f"Hidden Hand conflict detected: {signal} during {wyckoff_phase} "
                    f"with {volume_trend} volume -> multiplier={multiplier}"
                )
                return (True, multiplier, reason)

        # Check for specific known dangerous patterns
        if wyckoff_phase == WyckoffPhase.DISTRIBUTION and signal == "BUY":
            # Distribution phase + BUY = potential upthrust trap
            if volume_trend == VolumeProfile.DECLINING:
                return (
                    True,
                    0.5,
                    "BUY during distribution with declining volume - "
                    "likely upthrust trap, institutional selling detected"
                )
            else:
                return (
                    True,
                    0.7,
                    "BUY during distribution - exercise caution, "
                    "monitor for upthrust pattern confirmation"
                )

        if wyckoff_phase == WyckoffPhase.ACCUMULATION and signal == "SELL":
            # Accumulation phase + SELL = potential spring trap
            if volume_trend == VolumeProfile.DECLINING:
                return (
                    True,
                    0.5,
                    "SELL during accumulation with declining volume - "
                    "likely spring trap, institutional buying detected"
                )
            else:
                return (
                    True,
                    0.7,
                    "SELL during accumulation - exercise caution, "
                    "monitor for spring pattern confirmation"
                )

        # Signal aligned with phase
        if wyckoff_phase == WyckoffPhase.MARKUP and signal == "BUY":
            return (True, 1.0, "BUY aligned with markup phase - bullish continuation")

        if wyckoff_phase == WyckoffPhase.MARKDOWN and signal == "SELL":
            return (True, 1.0, "SELL aligned with markdown phase - bearish continuation")

        if signal == "HOLD":
            return (True, 1.0, "HOLD signal - no phase conflict")

        # Default: signal doesn't conflict with known patterns
        return (True, 1.0, "Signal consistent with Hidden Hand analysis")

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a trading_signal validation task.

        Args:
            task: Task assignment with signal data and market context

        Returns:
            Validation result with is_valid, confidence_multiplier, and reasoning
        """
        task_id = task["task_id"]
        self.current_task_id = task_id

        logger.info(f"Processing signal validation task: {task_id}")

        # FM-TAW-007: Mark task as processing in Redis
        if self.redis:
            await self.redis.sadd(PROCESSING_TASKS_KEY, task_id)
            logger.debug(f"Task {task_id} added to processing_tasks set")

        try:
            # Extract signal and market context from task
            metadata = task.get("metadata", {})
            signal_data = metadata.get("signal_data", {})
            market_context = metadata.get("market_context", {})

            # Extract signal details
            signal = signal_data.get("signal", "HOLD")
            original_confidence = signal_data.get("confidence", 0.5)
            original_reasoning = signal_data.get("reasoning", "")
            risk_score = signal_data.get("risk_score", 0.5)

            # Extract market context
            symbol = market_context.get("symbol", "UNKNOWN")
            timeframe = market_context.get("timeframe", "1H")
            wyckoff_phase = market_context.get("wyckoff_phase", WyckoffPhase.UNKNOWN)
            volume_trend = market_context.get("volume_trend", VolumeProfile.UNKNOWN)

            # FM-TAW-012: Validate against Hidden Hand methodology
            is_valid, phase_multiplier, phase_reason = self.validate_against_hidden_hand(
                signal=signal,
                wyckoff_phase=wyckoff_phase,
                volume_trend=volume_trend
            )

            # Optionally call Sovereign for deeper analysis
            sovereign_analysis = None
            fallback_used = False

            if metadata.get("require_sovereign_analysis", False):
                try:
                    prompt = self._build_validation_prompt(
                        signal=signal,
                        confidence=original_confidence,
                        reasoning=original_reasoning,
                        wyckoff_phase=wyckoff_phase,
                        volume_trend=volume_trend,
                        symbol=symbol,
                        timeframe=timeframe
                    )

                    sovereign_result = await self.sovereign.analysis_completion(
                        prompt,
                        "You are a signal validation expert specializing in Hidden Hand methodology."
                    )
                    sovereign_analysis = sovereign_result
                    fallback_used = sovereign_result.get("fallback_used", False)

                except SovereignClientError as e:
                    logger.warning(f"Sovereign analysis failed for task {task_id}: {e.message}")
                    sovereign_analysis = {"error": e.message}

            # Calculate final confidence
            adjusted_confidence = original_confidence * phase_multiplier

            # FM-TAW-006: Apply fallback penalty if Sovereign was used and fell back
            if fallback_used:
                pre_fallback_confidence = adjusted_confidence
                adjusted_confidence *= FALLBACK_CONFIDENCE_MULTIPLIER
                logger.warning(
                    f"Task {task_id} used fallback model: "
                    f"confidence {pre_fallback_confidence:.2f} -> {adjusted_confidence:.2f}"
                )

            # Build result
            result = {
                "task_id": task_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "original_signal": signal,
                "original_confidence": original_confidence,
                "adjusted_confidence": adjusted_confidence,
                "phase_multiplier": phase_multiplier,
                "validation_reason": phase_reason,
                "wyckoff_phase": wyckoff_phase,
                "volume_trend": volume_trend,
                "is_valid": is_valid,
                "phase_signal_conflict": phase_multiplier < 1.0,
                "risk_score": risk_score,
                "validated_at": datetime.utcnow().isoformat(),
            }

            # Add fallback warning if applicable
            if fallback_used:
                result["degraded_inference"] = True
                result["fallback_multiplier"] = FALLBACK_CONFIDENCE_MULTIPLIER
                result["degradation_reason"] = "fallback_model_used"

            # Add Hidden Hand warning if phase conflict detected
            if phase_multiplier < 1.0:
                result["hidden_hand_warning"] = phase_reason
                logger.warning(
                    f"Task {task_id} phase conflict: {signal} during {wyckoff_phase} "
                    f"-> confidence reduced from {original_confidence:.2f} to {adjusted_confidence:.2f}"
                )

            # Add Sovereign analysis if performed
            if sovereign_analysis:
                result["sovereign_analysis"] = sovereign_analysis

            logger.info(
                f"Task {task_id} validation complete: signal={signal}, "
                f"phase={wyckoff_phase}, conflict={phase_multiplier < 1.0}, "
                f"adjusted_confidence={adjusted_confidence:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Unexpected error for task {task_id}: {e}")
            return {
                "task_id": task_id,
                "error": True,
                "error_code": "VALIDATOR_ERROR",
                "error_message": str(e),
                "is_valid": False,
                "adjusted_confidence": 0.0,
                "validation_reason": f"Validation error: {str(e)}",
            }

        finally:
            # FM-TAW-007: Remove task from processing set
            if self.redis:
                await self.redis.srem(PROCESSING_TASKS_KEY, task_id)
                logger.debug(f"Task {task_id} removed from processing_tasks set")
            self.current_task_id = None

    def _build_validation_prompt(
        self,
        signal: str,
        confidence: float,
        reasoning: str,
        wyckoff_phase: str,
        volume_trend: str,
        symbol: str,
        timeframe: str
    ) -> str:
        """Build validation prompt for Sovereign analysis"""

        prompt = f"""Validate the following trading signal against Hidden Hand methodology:

SYMBOL: {symbol}
TIMEFRAME: {timeframe}

SIGNAL: {signal}
ORIGINAL CONFIDENCE: {confidence:.2f}
ORIGINAL REASONING: {reasoning}

MARKET CONTEXT:
  Wyckoff Phase: {wyckoff_phase}
  Volume Trend: {volume_trend}

VALIDATION TASK:
1. Assess whether this signal aligns with the current Wyckoff phase
2. Check for potential institutional manipulation patterns (springs, upthrusts)
3. Evaluate if the signal might be a trap set by smart money
4. Consider volume confirmation or divergence

Hidden Hand Principles to Apply:
- Distribution phase + BUY signal = potential upthrust trap
- Accumulation phase + SELL signal = potential spring trap
- Volume declining on breakout = likely false move
- Volume rising on reversal = likely genuine move

Return your validation as JSON with:
- is_valid: boolean
- confidence_adjustment: float (multiplier 0.0-1.0)
- reasoning: string explaining your validation
- risk_score: float (0.0-1.0)
"""
        return prompt

    async def send_task_complete(self, result: Dict[str, Any]):
        """Send task completion to orchestrator"""
        if not self.ws:
            logger.error("Cannot send task_complete: WebSocket not connected")
            return

        try:
            complete_msg = {
                "type": "task_complete",
                "agent_id": self.config.agent_id,
                "task_id": result["task_id"],
                "result": result,
                "reasoning_steps": [
                    f"Original Signal: {result.get('original_signal', 'N/A')}",
                    f"Wyckoff Phase: {result.get('wyckoff_phase', 'N/A')}",
                    f"Phase Conflict: {result.get('phase_signal_conflict', False)}",
                    f"Original Confidence: {result.get('original_confidence', 0):.2f}",
                    f"Adjusted Confidence: {result.get('adjusted_confidence', 0):.2f}",
                    f"Validation: {result.get('validation_reason', 'N/A')}",
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.ws.send(json.dumps(complete_msg))
            logger.info(f"Task complete sent: {result['task_id']}")

        except Exception as e:
            logger.error(f"Failed to send task_complete: {e}")

    async def run(self):
        """Main agent loop"""
        self.running = True

        # Connect to AAOS
        if not await self.connect():
            logger.error("Failed to connect to AAOS - exiting")
            return

        # Initialize services
        if not await self.initialize_services():
            logger.error("Failed to initialize services - exiting")
            return

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self.send_heartbeat())

        logger.info(f"Agent {self.config.agent_id} running - waiting for tasks")

        try:
            while self.running:
                try:
                    # Wait for messages from orchestrator
                    message = await asyncio.wait_for(
                        self.ws.recv(),
                        timeout=self.config.heartbeat_interval * 2
                    )
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "task_assignment":
                        # Process assigned task
                        logger.info(f"Received task assignment: {data.get('task_id')}")
                        result = await self.process_task(data)
                        await self.send_task_complete(result)

                    elif msg_type == "heartbeat_ack":
                        logger.debug("Heartbeat acknowledged")

                    elif msg_type == "shutdown":
                        logger.info("Received shutdown command")
                        self.running = False

                    else:
                        logger.debug(f"Received message type: {msg_type}")

                except asyncio.TimeoutError:
                    # No message received - continue loop
                    continue

                except websockets.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    self.running = False

        except Exception as e:
            logger.error(f"Agent loop error: {e}")

        finally:
            # Cleanup
            self.running = False
            heartbeat_task.cancel()
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"Shutting down agent: {self.config.agent_id}")

        self.running = False

        # Close Sovereign client
        if self.sovereign:
            await self.sovereign.close()

        # Close Redis
        if self.redis:
            await self.redis.close()

        # Close WebSocket
        if self.ws:
            await self.ws.close()

        logger.info("Agent shutdown complete")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point with signal handling"""

    # Create agent
    config = SignalValidatorConfig()
    agent = SignalValidatorAgent(config)

    # Setup signal handlers for graceful shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig} - initiating shutdown")
        agent.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        loop.run_until_complete(agent.run())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        loop.run_until_complete(agent.shutdown())
        loop.close()


if __name__ == "__main__":
    main()
