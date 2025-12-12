# agents/trading/trading_analyst.py
"""
Trading Analyst Agent

Claims trading_analysis tasks from AAOS and calls Sovereign for AI-powered analysis.
Implements defensive measures from ensemble resilience review.

Defensive Measures:
  - FM-TAW-004: Registers sovereign_access capability on connect
  - FM-TAW-006: Applies 0.8x confidence multiplier when fallback model used
  - FM-TAW-007: Tracks task in Redis processing_tasks set during Sovereign call
"""

import os
import sys
import json
import signal
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

import websockets
import redis.asyncio as aioredis

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.integrations.sovereign_client import SovereignClient, SovereignClientError
from agents.trading.agent_config import (
    TradingAnalystConfig,
    FALLBACK_CONFIDENCE_MULTIPLIER,
    PROCESSING_TASKS_KEY,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("aaos.trading_analyst")


class TradingAnalystAgent:
    """
    Trading Analyst Agent for AAOS.

    Connects to orchestrator via WebSocket, claims trading_analysis tasks,
    calls Sovereign for AI analysis, and returns structured results.
    """

    def __init__(self, config: Optional[TradingAnalystConfig] = None):
        self.config = config or TradingAnalystConfig()
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.sovereign: Optional[SovereignClient] = None
        self.redis: Optional[aioredis.Redis] = None
        self.running = False
        self.current_task_id: Optional[str] = None

        logger.info(
            f"TradingAnalystAgent initialized: agent_id={self.config.agent_id}, "
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

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a trading_analysis task.

        Args:
            task: Task assignment with task_id, description, metadata

        Returns:
            Analysis result with signal, confidence, reasoning, risk_score
        """
        task_id = task["task_id"]
        self.current_task_id = task_id

        logger.info(f"Processing task: {task_id}")

        # FM-TAW-007: Mark task as processing in Redis
        if self.redis:
            await self.redis.sadd(PROCESSING_TASKS_KEY, task_id)
            logger.debug(f"Task {task_id} added to processing_tasks set")

        try:
            # Extract trading context from task
            metadata = task.get("metadata", {})
            symbol = metadata.get("symbol", "UNKNOWN")
            timeframe = metadata.get("timeframe", "1H")
            indicators = metadata.get("indicators", {})
            price_data = metadata.get("price_data", {})
            context = metadata.get("context", "")

            # Build analysis prompt
            prompt = self._build_analysis_prompt(
                symbol=symbol,
                timeframe=timeframe,
                indicators=indicators,
                price_data=price_data,
                description=task.get("description", ""),
                context=context
            )

            # Call Sovereign for analysis
            system_role = (
                "You are an expert trading analyst specializing in technical analysis "
                "and market structure. Analyze the provided data using Wyckoff methodology "
                "and Hidden Hand principles. Identify institutional activity patterns."
            )

            result = await self.sovereign.analysis_completion(prompt, system_role)

            # FM-TAW-006: Apply fallback distrust penalty
            if result.get("fallback_used", False):
                original_confidence = result["confidence"]
                result["confidence"] *= FALLBACK_CONFIDENCE_MULTIPLIER
                result["degraded_inference"] = True
                result["original_confidence"] = original_confidence
                result["degradation_reason"] = "fallback_model_used"

                logger.warning(
                    f"Task {task_id} used fallback model: "
                    f"confidence {original_confidence:.2f} -> {result['confidence']:.2f}"
                )

            # Add task context to result
            result["task_id"] = task_id
            result["symbol"] = symbol
            result["timeframe"] = timeframe
            result["analyzed_at"] = datetime.utcnow().isoformat()

            logger.info(
                f"Task {task_id} analysis complete: signal={result['signal']}, "
                f"confidence={result['confidence']:.2f}, risk_score={result['risk_score']:.2f}"
            )

            return result

        except SovereignClientError as e:
            logger.error(f"Sovereign error for task {task_id}: {e.message}")
            return {
                "task_id": task_id,
                "error": True,
                "error_code": e.error_code,
                "error_message": e.message,
                "signal": "ERROR",
                "confidence": 0.0,
                "risk_score": 1.0,
                "reasoning": f"Analysis failed: {e.message}",
            }

        except Exception as e:
            logger.error(f"Unexpected error for task {task_id}: {e}")
            return {
                "task_id": task_id,
                "error": True,
                "error_code": "AGENT_ERROR",
                "error_message": str(e),
                "signal": "ERROR",
                "confidence": 0.0,
                "risk_score": 1.0,
                "reasoning": f"Agent error: {str(e)}",
            }

        finally:
            # FM-TAW-007: Remove task from processing set
            if self.redis:
                await self.redis.srem(PROCESSING_TASKS_KEY, task_id)
                logger.debug(f"Task {task_id} removed from processing_tasks set")
            self.current_task_id = None

    def _build_analysis_prompt(
        self,
        symbol: str,
        timeframe: str,
        indicators: Dict[str, Any],
        price_data: Dict[str, Any],
        description: str,
        context: str = ""
    ) -> str:
        """Build analysis prompt from task data"""

        # Format indicators
        indicator_text = ""
        if indicators:
            indicator_lines = []
            for name, value in indicators.items():
                if isinstance(value, dict):
                    indicator_lines.append(f"  {name}: {json.dumps(value)}")
                else:
                    indicator_lines.append(f"  {name}: {value}")
            indicator_text = "\n".join(indicator_lines)

        # Format price data
        price_text = ""
        if price_data:
            price_text = (
                f"  Current: {price_data.get('current', 'N/A')}\n"
                f"  Open: {price_data.get('open', 'N/A')}\n"
                f"  High: {price_data.get('high', 'N/A')}\n"
                f"  Low: {price_data.get('low', 'N/A')}\n"
                f"  Volume: {price_data.get('volume', 'N/A')}"
            )

        prompt = f"""Analyze the following trading setup:

SYMBOL: {symbol}
TIMEFRAME: {timeframe}

TASK: {description}

TECHNICAL INDICATORS:
{indicator_text if indicator_text else "  No indicator data provided"}

PRICE DATA:
{price_text if price_text else "  No price data provided"}

{f"ADDITIONAL CONTEXT: {context}" if context else ""}

Provide your analysis focusing on:
1. Current market phase (accumulation, markup, distribution, markdown)
2. Institutional activity signals (Hidden Hand patterns)
3. Key support/resistance levels
4. Risk assessment

Return your analysis as JSON with signal, confidence, reasoning, and risk_score.
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
                    f"Signal: {result.get('signal', 'N/A')}",
                    f"Confidence: {result.get('confidence', 0):.2f}",
                    f"Risk Score: {result.get('risk_score', 0):.2f}",
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
    config = TradingAnalystConfig()
    agent = TradingAnalystAgent(config)

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
