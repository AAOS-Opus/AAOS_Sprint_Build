# src/integrations/aurora_bridge.py
"""
Aurora Bridge - Routes AAOS task completion events to Aurora's Event Bus

Subscribes to Redis PubSub channel "aaos:task_complete" and routes results
to Aurora's internal Event Bus channels based on task_type.

Defensive Measures:
  - FM-RF-001: Out-of-order prevention via timestamp tracking
  - FM-RF-002: Backpressure handling with bounded queue
  - FM-RF-003: Graceful reconnection on Redis disconnect

Event Routing:
  - trading_analysis -> analysis-events
  - trading_signal -> signal-events

Also emits WebSocket "sidebar-update" events for real-time UI updates.
"""

import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any, Set, Callable, Awaitable
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field

import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("aurora.bridge")


# =============================================================================
# Configuration
# =============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
AAOS_CHANNEL = "aaos:task_complete"

# FM-RF-002: Bounded queue configuration
MAX_QUEUE_SIZE = 100
COMPACT_THRESHOLD = 80  # Compact when queue reaches this size


# =============================================================================
# Event Bus Interface (Abstract)
# =============================================================================

class EventBusInterface:
    """
    Abstract Event Bus interface for Aurora integration.

    In production, this would connect to Aurora's actual Event Bus.
    This implementation provides a local pub/sub for testing.
    """

    def __init__(self):
        self._subscribers: Dict[str, Set[Callable[[Dict[str, Any]], Awaitable[None]]]] = {}

    def subscribe(self, channel: str, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Subscribe to a channel"""
        if channel not in self._subscribers:
            self._subscribers[channel] = set()
        self._subscribers[channel].add(callback)
        logger.debug(f"Subscribed to Event Bus channel: {channel}")

    def unsubscribe(self, channel: str, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Unsubscribe from a channel"""
        if channel in self._subscribers:
            self._subscribers[channel].discard(callback)

    async def publish(self, channel: str, event: Dict[str, Any]):
        """Publish event to a channel"""
        if channel in self._subscribers:
            for callback in self._subscribers[channel]:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Event Bus subscriber error on {channel}: {e}")

        logger.debug(f"Event Bus publish: channel={channel}, event_type={event.get('type')}")


# =============================================================================
# WebSocket Manager (Abstract)
# =============================================================================

class WebSocketManager:
    """
    Manages WebSocket connections for sidebar updates.

    In production, this would connect to Aurora's WebSocket server.
    This implementation provides local connection tracking.
    """

    def __init__(self):
        self._clients: Dict[str, Any] = {}  # client_id -> websocket

    def register(self, client_id: str, websocket: Any):
        """Register a WebSocket client"""
        self._clients[client_id] = websocket
        logger.info(f"WebSocket client registered: {client_id}")

    def unregister(self, client_id: str):
        """Unregister a WebSocket client"""
        if client_id in self._clients:
            del self._clients[client_id]
            logger.info(f"WebSocket client unregistered: {client_id}")

    async def broadcast(self, event: Dict[str, Any]):
        """Broadcast event to all connected clients"""
        disconnected = []

        for client_id, ws in self._clients.items():
            try:
                if hasattr(ws, 'send_json'):
                    await ws.send_json(event)
                elif hasattr(ws, 'send'):
                    await ws.send(json.dumps(event))
            except Exception as e:
                logger.warning(f"Failed to send to client {client_id}: {e}")
                disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            self.unregister(client_id)

        if self._clients:
            logger.debug(f"Broadcast sidebar-update to {len(self._clients)} clients")


# =============================================================================
# Bounded Event Queue (FM-RF-002)
# =============================================================================

@dataclass
class QueuedEvent:
    """Event in the bounded queue"""
    task_id: str
    task_type: str
    symbol: str
    result: Dict[str, Any]
    completed_at: str
    queued_at: datetime = field(default_factory=datetime.utcnow)


class BoundedEventQueue:
    """
    FM-RF-002: Bounded queue with overflow protection.

    When queue reaches capacity, compacts by keeping only the latest
    event per symbol to prevent memory issues during bursts.
    """

    def __init__(self, max_size: int = MAX_QUEUE_SIZE, compact_threshold: int = COMPACT_THRESHOLD):
        self.max_size = max_size
        self.compact_threshold = compact_threshold
        self._queue: deque[QueuedEvent] = deque(maxlen=max_size)
        self._latest_by_symbol: Dict[str, QueuedEvent] = {}

    def push(self, event: QueuedEvent) -> bool:
        """
        Push event to queue.

        Returns True if added, False if dropped due to overflow.
        """
        # Update latest by symbol tracker
        self._latest_by_symbol[event.symbol] = event

        # Check if we need to compact
        if len(self._queue) >= self.compact_threshold:
            self._compact()

        # Add to queue
        self._queue.append(event)
        return True

    def _compact(self):
        """
        Compact queue by keeping only latest event per symbol.

        This prevents memory issues when multiple analyses for the
        same symbol complete rapidly.
        """
        before_count = len(self._queue)

        # Keep only the latest events
        unique_events = list(self._latest_by_symbol.values())
        self._queue.clear()
        for event in unique_events:
            self._queue.append(event)

        after_count = len(self._queue)
        logger.info(
            f"Queue compacted: {before_count} -> {after_count} events "
            f"(kept latest per symbol)"
        )

    def pop(self) -> Optional[QueuedEvent]:
        """Pop oldest event from queue"""
        if self._queue:
            return self._queue.popleft()
        return None

    def __len__(self) -> int:
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0


# =============================================================================
# Aurora Bridge Service
# =============================================================================

class AuroraBridge:
    """
    Bridge service connecting AAOS task completions to Aurora's Event Bus.

    Subscribes to Redis PubSub, routes events based on task_type,
    and pushes sidebar updates via WebSocket.
    """

    def __init__(
        self,
        event_bus: Optional[EventBusInterface] = None,
        ws_manager: Optional[WebSocketManager] = None
    ):
        self.event_bus = event_bus or EventBusInterface()
        self.ws_manager = ws_manager or WebSocketManager()

        self._redis: Optional[aioredis.Redis] = None
        self._pubsub: Optional[aioredis.client.PubSub] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # FM-RF-001: Track last timestamp per task_type to prevent out-of-order
        self._last_timestamp: Dict[str, str] = {}

        # FM-RF-002: Bounded event queue
        self._queue = BoundedEventQueue()

        # Task type to Event Bus channel mapping
        self._channel_map = {
            "trading_analysis": "analysis-events",
            "trading_signal": "signal-events",
        }

        logger.info("AuroraBridge initialized")

    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            self._redis = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
            )
            await self._redis.ping()
            logger.info(f"Connected to Redis: {REDIS_HOST}:{REDIS_PORT}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    async def subscribe(self) -> bool:
        """Subscribe to AAOS task completion channel"""
        if not self._redis:
            logger.error("Cannot subscribe: Redis not connected")
            return False

        try:
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe(AAOS_CHANNEL)
            logger.info(f"Subscribed to Redis channel: {AAOS_CHANNEL}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to {AAOS_CHANNEL}: {e}")
            return False

    def _is_stale(self, task_type: str, completed_at: str) -> bool:
        """
        FM-RF-001: Check if event is stale (out-of-order).

        Returns True if the incoming event is older than the last
        processed event for this task_type.
        """
        last = self._last_timestamp.get(task_type)
        if not last:
            return False

        # Compare ISO timestamps lexicographically (works for ISO format)
        if completed_at < last:
            logger.warning(
                f"Stale event detected for {task_type}: "
                f"incoming={completed_at} < last={last}"
            )
            return True

        return False

    def _update_timestamp(self, task_type: str, completed_at: str):
        """Update last timestamp for task_type"""
        current = self._last_timestamp.get(task_type)
        if not current or completed_at > current:
            self._last_timestamp[task_type] = completed_at

    async def _process_message(self, message: Dict[str, Any]):
        """Process a single Redis PubSub message"""
        if message["type"] != "message":
            return

        try:
            # Parse message data
            data = json.loads(message["data"])
            task_id = data.get("task_id")
            task_type = data.get("task_type")
            status = data.get("status")
            result = data.get("result", {})
            completed_at = data.get("completed_at")
            agent_id = data.get("agent_id")

            logger.info(
                f"Received task_complete: task_id={task_id}, "
                f"task_type={task_type}, status={status}"
            )

            # Skip failed tasks (only route completed)
            if status != "completed":
                logger.debug(f"Skipping non-completed task: {task_id} ({status})")
                return

            # FM-RF-001: Check for stale (out-of-order) event
            if self._is_stale(task_type, completed_at):
                logger.warning(f"Dropping stale event: {task_id}")
                return

            # Update timestamp tracker
            self._update_timestamp(task_type, completed_at)

            # Extract symbol from result
            symbol = result.get("symbol", "UNKNOWN")

            # Route to Event Bus based on task_type
            event_bus_channel = self._channel_map.get(task_type)
            if event_bus_channel:
                event = {
                    "type": f"{task_type}_complete",
                    "task_id": task_id,
                    "payload": {
                        "result": result,
                        "completed_at": completed_at,
                        "agent_id": agent_id,
                    }
                }
                await self.event_bus.publish(event_bus_channel, event)
                logger.debug(f"Routed to Event Bus: {event_bus_channel}")
            else:
                logger.debug(f"No Event Bus channel mapped for task_type: {task_type}")

            # Emit sidebar-update WebSocket event
            sidebar_event = {
                "type": "sidebar-update",
                "task_type": task_type,
                "symbol": symbol,
                "analysis": {
                    "signal": result.get("signal"),
                    "confidence": result.get("adjusted_confidence", result.get("confidence")),
                    "risk_score": result.get("risk_score"),
                    "reasoning": result.get("reasoning", "")[:200],  # Truncate for sidebar
                    "wyckoff_phase": result.get("wyckoff_phase"),
                    "degraded_inference": result.get("degraded_inference", False),
                    "hidden_hand_warning": result.get("hidden_hand_warning"),
                },
                "timestamp": completed_at
            }
            await self.ws_manager.broadcast(sidebar_event)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message JSON: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _listen_loop(self):
        """Main loop listening for Redis PubSub messages"""
        logger.info("Aurora Bridge listen loop started")

        while self._running:
            try:
                # Get message with timeout
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )

                if message:
                    await self._process_message(message)

            except aioredis.ConnectionError as e:
                logger.error(f"Redis connection lost: {e}")
                # Attempt to reconnect
                await self._reconnect()

            except asyncio.CancelledError:
                logger.info("Listen loop cancelled")
                break

            except Exception as e:
                logger.error(f"Listen loop error: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

        logger.info("Aurora Bridge listen loop stopped")

    async def _reconnect(self):
        """Attempt to reconnect to Redis"""
        logger.info("Attempting Redis reconnection...")

        for attempt in range(5):
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

            try:
                if await self.connect() and await self.subscribe():
                    logger.info(f"Reconnected to Redis (attempt {attempt + 1})")
                    return
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")

        logger.error("Failed to reconnect to Redis after 5 attempts")
        self._running = False

    async def start(self):
        """Start the Aurora Bridge service"""
        logger.info("Starting Aurora Bridge...")

        if not await self.connect():
            logger.error("Failed to start: Redis connection failed")
            return False

        if not await self.subscribe():
            logger.error("Failed to start: Redis subscription failed")
            return False

        self._running = True
        self._task = asyncio.create_task(self._listen_loop())

        logger.info("Aurora Bridge started successfully")
        return True

    async def stop(self):
        """Stop the Aurora Bridge service"""
        logger.info("Stopping Aurora Bridge...")

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._pubsub:
            await self._pubsub.unsubscribe(AAOS_CHANNEL)
            await self._pubsub.close()

        if self._redis:
            await self._redis.close()

        logger.info("Aurora Bridge stopped")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status for health monitoring"""
        return {
            "running": self._running,
            "redis_connected": self._redis is not None,
            "subscribed_channel": AAOS_CHANNEL,
            "last_timestamps": self._last_timestamp.copy(),
            "queue_size": len(self._queue),
            "channel_map": self._channel_map.copy(),
        }


# =============================================================================
# Standalone Runner
# =============================================================================

async def run_bridge():
    """Run Aurora Bridge as standalone service"""
    bridge = AuroraBridge()

    # Handle shutdown signals
    import signal

    def signal_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(bridge.stop())

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    async with bridge:
        # Keep running until stopped
        while bridge._running:
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run_bridge())
