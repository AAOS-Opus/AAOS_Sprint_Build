# Kimi K2 Resilience Analysis: Trading Agent Workers Design

**Analysis Date**: 2025-12-11
**Analyst**: Kimi K2 (Moonshot) via Resilience Engineer Protocol
**ISC Compliance**: Full adherence to Inter-System Communication standards
**Status**: PRE-IMPLEMENTATION REVIEW

---

## Executive Summary

The Trading Agent Workers design introduces **critical integration boundary risks** that don't throw exceptions but fail silently. This analysis identifies **12 failure modes** across WebSocket lifecycle, task claiming, Sovereign parsing, and result delivery. The Sovereign client has excellent defensive measures (FM-001 through FM-006) - the gap is in the **agent-to-orchestrator integration layer**.

**Overall Risk Assessment**: MEDIUM-HIGH
**Recommendation**: Implement defensive measures before production deployment

---

## Component Analysis

### 1. sovereign_client.py (Existing - Well Defended)

The Sovereign client implements robust defensive measures:
- **FM-001**: Semantic validation (risk_score range 0-1) ✅
- **FM-002**: Context cap at 20K tokens ✅
- **FM-003**: Fallback distrust flag when 7B responds ✅
- **FM-004**: Per-request timeouts ✅
- **FM-005**: Per-model circuit breaker ✅
- **FM-006**: Standardized error envelope ✅

**Gap Identified**: The client is defensive, but **callers must handle the distrust flag**.

### 2. orchestrator/core.py (Existing - Task Flow Analyzed)

The orchestrator has strong fundamentals:
- Two-phase commit for task creation ✅
- Task circuit breaker for overload protection ✅
- Timeout/retry with exponential backoff ✅
- WebSocket authentication via API key ✅

**Gap Identified**: Task types `trading_analysis` and `trading_signal` exist in enum but **no capability matching logic** ensures trading agents claim trading tasks.

---

## Failure Mode Catalog

### FM-TAW-001: WebSocket Connection Race on Startup

**Symptom**: Agent connects, sends register, but orchestrator hasn't added to `active_connections` yet.

**Silent Failure Pattern**:
```python
# Agent sends:
{"type": "register", "agent_id": "trading_analyst_01", ...}

# Orchestrator receives but hasn't called manager.active_connections[agent_id] = websocket
# Next: task_assignment_loop calls manager.get_idle_agent() -> returns None
# Result: Tasks queue indefinitely despite agent being "registered"
```

**Detection**: Agent receives `registration_ack` but never gets tasks.

**Mitigation**: Add connection readiness gate - don't mark agent as `idle` until bidirectional message confirmed.

### FM-TAW-002: Heartbeat Timeout Without Task Reassignment

**Symptom**: Agent stops sending heartbeats, orchestrator doesn't detect for extended period.

**Current Gap**: No heartbeat timeout monitoring in `ConnectionManager`. The `update_heartbeat()` method records timestamps but nothing **checks** them.

**Silent Failure Pattern**:
```python
# Agent freezes (Sovereign call hangs beyond httpx timeout)
# Heartbeat updates stop
# Orchestrator never notices until WebSocket TCP keepalive fails (could be minutes)
# Task remains "assigned" indefinitely
```

**Mitigation**: Add heartbeat watchdog in `task_assignment_loop` or dedicated coroutine.

### FM-TAW-003: Task Claiming Race Condition

**Symptom**: Two trading agents both see same task in queue, both attempt to claim.

**Current Protection**: FIFO peek-then-pop pattern (`lindex` + `rpop`) is safe for single consumer but **not for multiple agents of same type**.

**Silent Failure Pattern**:
```python
# Agent A: lindex("task_queue", -1) -> "task_123"
# Agent B: lindex("task_queue", -1) -> "task_123" (same task!)
# Agent A: finds idle, rpop() -> "task_123"
# Agent B: finds idle, rpop() -> None (race lost) OR different task
# Result: Agent B may get wrong task or spin unnecessarily
```

**Mitigation**: Use Redis BRPOPLPUSH to atomically move task to "processing" list per agent, OR use distributed lock.

### FM-TAW-004: Capability Mismatch - Trading Task to Non-Trading Agent

**Symptom**: `trading_analysis` task assigned to generic agent that can't call Sovereign.

**Current Gap**: `get_idle_agent()` returns **any** idle agent, ignoring capabilities.

**Silent Failure Pattern**:
```python
# Task: {"task_type": "trading_analysis", "description": "Analyze BTCUSD"}
# Generic agent claims it, has no Sovereign client
# Agent either crashes or returns garbage result
# Task marked "completed" with invalid analysis
```

**Mitigation**: Add capability matching in `get_idle_agent()`:
```python
def get_idle_agent(self, required_capabilities: List[str] = None) -> Optional[str]:
    for agent_id, info in self.agent_info.items():
        if info["status"] == "idle" and agent_id in self.active_connections:
            if required_capabilities:
                agent_caps = info.get("capabilities", {})
                if all(cap in agent_caps for cap in required_capabilities):
                    return agent_id
            else:
                return agent_id
    return None
```

### FM-TAW-005: Sovereign Response Parsing Failure

**Symptom**: Sovereign returns valid JSON but with unexpected structure.

**Current Protection**: `analysis_completion()` validates required fields. But...

**Silent Failure Pattern**:
```python
# Sovereign responds with:
{"signal": "HOLD", "confidence": 0.7, "reasoning": "...", "risk_score": 0.3,
 "unexpected_field": {"nested": "data"}}

# Agent receives, passes validation
# Agent's prompt extraction assumes specific field presence
# Agent sends partial result to orchestrator
# Task "completes" with incomplete analysis
```

**Mitigation**: Define explicit TypedDict/dataclass for trading response, validate on agent side too.

### FM-TAW-006: Fallback Distrust Flag Ignored

**Symptom**: 7B model responds (primary circuit open), agent doesn't downgrade confidence.

**Current Gap**: `analysis_completion()` returns `fallback_used: bool` but **nothing enforces action**.

**Silent Failure Pattern**:
```python
result = await sovereign_client.analysis_completion(prompt)
# result["fallback_used"] = True (7B model used)
# Agent ignores this flag
# Trading decision made with lower-quality analysis
# No audit trail of degraded inference
```

**Mitigation**: Agent MUST check `fallback_used` and either:
1. Set `"degraded_inference": true` in result metadata
2. Reduce confidence score by 20%
3. Require human review for signals from fallback

### FM-TAW-007: WebSocket Disconnect During Sovereign Call

**Symptom**: Agent disconnects mid-inference, task orphaned.

**Current Protection**: `task_assignment_loop` handles disconnect by requeuing. But...

**Silent Failure Pattern**:
```python
# Agent assigned task "task_456"
# Agent calls sovereign_client.analysis_completion() - takes 60+ seconds
# WebSocket heartbeat timeout triggers disconnect detection
# Orchestrator requeues task to Redis
# Sovereign COMPLETES, agent tries to send task_complete
# WebSocket already closed - message lost
# Task now in queue AGAIN - will be double-processed
```

**Mitigation**: Use Redis "processing_tasks" set (already referenced in timeout detection):
1. Before Sovereign call: `SADD processing_tasks task_456`
2. After Sovereign call: `SREM processing_tasks task_456`
3. On reconnect: Check if task was in processing, resume or abandon

### FM-TAW-008: Result Delivery Guarantee Failure

**Symptom**: Agent sends `task_complete` but WebSocket buffer fills, message dropped.

**Current Gap**: No acknowledgment pattern for `task_complete` messages.

**Silent Failure Pattern**:
```python
# Agent completes analysis
await websocket.send_json({"type": "task_complete", "task_id": "...", "result": {...}})
# WebSocket send succeeds (from agent perspective)
# But: Network glitch, orchestrator never receives
# Agent thinks it's done, goes idle
# Task remains "assigned" in DB forever (until timeout detection)
```

**Mitigation**: Add ACK pattern:
1. Agent sends `task_complete`
2. Orchestrator responds `{"type": "task_complete_ack", "task_id": "..."}`
3. Agent retries if no ACK within 10s
4. After 3 retries, agent marks task as "delivery_failed" locally

### FM-TAW-009: Context Overflow on Complex Trading Prompts

**Symptom**: Trading context with full indicator data exceeds 20K token limit.

**Current Protection**: `_check_context_size()` raises `ContextTooLargeError`. But...

**Silent Failure Pattern**:
```python
# Trading analyst builds prompt:
prompt = f"""
Analyze {symbol} on {timeframe}:
{full_candlestick_history}  # 500 candles = ~15K tokens
{14_indicator_values}       # 5K tokens
{hidden_hand_context}       # 3K tokens
"""
# Total: 23K tokens > 20K limit
# ContextTooLargeError raised
# Agent has no fallback - crashes or returns empty
```

**Mitigation**: Implement tiered context compression:
1. Full context attempt
2. On ContextTooLargeError: Summarize candlestick history to last 100 candles
3. Still too large: Use only most recent indicator values
4. Log context reduction for audit

### FM-TAW-010: Circuit Breaker Cascade - Sovereign + Task

**Symptom**: Sovereign primary circuit opens, task circuit ALSO opens, total system halt.

**Interaction Pattern**:
```python
# Sovereign primary (32B) circuit opens after 3 failures
# All tasks now use fallback (7B)
# 7B is slower, tasks timeout more frequently
# Task timeout detection marks tasks as failed
# Retry count exhausted rapidly
# Task circuit sees queue growing + agents failing
# Task circuit opens -> 503 for all new tasks
# System completely halted despite 7B being available
```

**Mitigation**: Separate circuit breaker domains:
1. Sovereign circuit: Controls model selection only
2. Task circuit: Should NOT count Sovereign timeouts as task failures
3. Add "Sovereign failure" as distinct failure reason in task metadata

### FM-TAW-011: Stale Task State on Agent Reconnect

**Symptom**: Agent reconnects with same ID, gets duplicate task assignment.

**Current Gap**: `register` message re-registers agent but doesn't check for in-flight tasks.

**Silent Failure Pattern**:
```python
# Agent "trading_analyst_01" assigned task_789
# Network blip, WebSocket disconnects
# Orchestrator requeues task_789
# Agent reconnects with same ID within 1 second
# Agent sends register, gets registration_ack
# Task assignment loop sends task_789 AGAIN
# Agent now has duplicate task context
```

**Mitigation**: On reconnect with existing agent_id:
1. Check `agent_tasks` mapping for in-flight task
2. If exists, send `task_resume` message instead of new assignment
3. Or: Reject reconnect until previous session fully cleaned up

### FM-TAW-012: Hidden Hand Methodology Violation - Signal Inversion

**Symptom**: Agent returns BUY signal when Hidden Hand analysis indicates distribution.

**This is DOMAIN-SPECIFIC but critical for Aurora TA integration.**

**Silent Failure Pattern**:
```python
# Sovereign analyzes and returns:
{"signal": "BUY", "reasoning": "Price broke resistance", ...}

# But Hidden Hand context indicated:
# - Volume declining on rally (distribution sign)
# - Upthrust pattern forming
# - Institutional selling detected

# Agent trusts Sovereign signal without Hidden Hand cross-check
# Trade executes against institutional flow
# Financial loss
```

**Mitigation**: Signal Validator agent MUST:
1. Parse Wyckoff phase from context
2. Check signal against phase expectations:
   - Distribution phase: BUY signals require extra validation
   - Accumulation phase: SELL signals require extra validation
3. Require consensus if signal contradicts phase
4. Log all overrides with reasoning

---

## Integration Boundary Stress Points

### WebSocket <-> Agent

| Boundary Point | Current State | Risk | Recommendation |
|----------------|--------------|------|----------------|
| Connection establishment | Auth before accept | LOW | ✅ Solid |
| Registration handshake | ACK sent | MEDIUM | Add readiness confirmation |
| Heartbeat monitoring | Timestamp recorded | HIGH | Add watchdog coroutine |
| Task assignment | Fire-and-forget | HIGH | Add assignment ACK |
| Task completion | Fire-and-forget | HIGH | Add completion ACK |
| Disconnect handling | Requeue on disconnect | MEDIUM | Check processing set |

### Agent <-> Sovereign

| Boundary Point | Current State | Risk | Recommendation |
|----------------|--------------|------|
| Connection pooling | httpx.AsyncClient | LOW | ✅ Solid |
| Request timeout | Per-request 120s | LOW | ✅ Solid |
| Response validation | JSON + field check | MEDIUM | Add TypedDict |
| Fallback handling | Flag returned | HIGH | **Enforce action** |
| Circuit breaker | Per-model tracking | LOW | ✅ Solid |
| Context limits | 20K token cap | MEDIUM | Add compression |

### Agent <-> Redis

| Boundary Point | Current State | Risk | Recommendation |
|----------------|--------------|------|
| Processing set | Referenced but unused | HIGH | **Implement fully** |
| Idempotency | Checked in timeout loop | MEDIUM | Check on reconnect too |
| Task claiming | FIFO peek-pop | MEDIUM | Consider BRPOPLPUSH |

---

## Recommended Defensive Measures for agents/trading/

### trading_analyst.py

```python
# Required defensive patterns:

class TradingAnalystAgent:
    async def process_task(self, task: dict) -> dict:
        task_id = task["task_id"]

        # FM-TAW-007: Mark processing
        await self.redis.sadd("processing_tasks", task_id)

        try:
            # FM-TAW-009: Context compression
            prompt = self._build_analysis_prompt(task, max_tokens=18000)

            result = await self.sovereign.analysis_completion(prompt)

            # FM-TAW-006: Handle fallback
            if result.get("fallback_used"):
                result["confidence"] *= 0.8  # 20% reduction
                result["degraded_inference"] = True
                logger.warning(f"Task {task_id} used fallback model")

            return result

        finally:
            # FM-TAW-007: Unmark processing
            await self.redis.srem("processing_tasks", task_id)
```

### signal_validator.py

```python
# FM-TAW-012: Hidden Hand validation

class SignalValidatorAgent:
    def validate_against_hidden_hand(
        self,
        signal: str,
        wyckoff_phase: str,
        volume_trend: str
    ) -> tuple[bool, str]:
        """
        Returns (is_valid, reason)
        """
        # Distribution phase + BUY = suspicious
        if wyckoff_phase == "distribution" and signal == "BUY":
            if volume_trend == "declining":
                return False, "BUY signal during distribution with declining volume - likely upthrust trap"

        # Accumulation phase + SELL = suspicious
        if wyckoff_phase == "accumulation" and signal == "SELL":
            if volume_trend == "declining":
                return False, "SELL signal during accumulation with declining volume - likely spring trap"

        return True, "Signal consistent with Hidden Hand analysis"
```

### agent_config.py

```python
# Defensive configuration

@dataclass
class TradingAgentConfig:
    # Connection
    ws_url: str = "ws://localhost:8000/ws"
    api_key: str = field(default_factory=lambda: os.getenv("AAOS_API_KEY", ""))

    # Heartbeat
    heartbeat_interval: int = 15  # seconds
    heartbeat_timeout: int = 45   # miss 3 = disconnect

    # Task handling
    task_ack_timeout: int = 10    # FM-TAW-008
    task_ack_retries: int = 3     # FM-TAW-008

    # Sovereign
    sovereign_timeout: int = 120
    max_context_tokens: int = 18000  # Leave 2K buffer from 20K limit

    # Capabilities (for FM-TAW-004)
    capabilities: dict = field(default_factory=lambda: {
        "trading_analysis": True,
        "trading_signal": True,
        "sovereign_access": True,
        "hidden_hand_validation": True
    })
```

---

## Risk Matrix

| ID | Failure Mode | Severity | Likelihood | Detection | Priority |
|----|-------------|----------|------------|-----------|----------|
| FM-TAW-001 | Connection race | Medium | Low | Low | P3 |
| FM-TAW-002 | Heartbeat timeout | High | Medium | Low | **P1** |
| FM-TAW-003 | Task claiming race | Medium | Low | Medium | P3 |
| FM-TAW-004 | Capability mismatch | High | Medium | Low | **P1** |
| FM-TAW-005 | Response parsing | Medium | Low | High | P3 |
| FM-TAW-006 | Fallback ignored | High | High | None | **P0** |
| FM-TAW-007 | Disconnect mid-call | High | Medium | Low | **P1** |
| FM-TAW-008 | Result delivery | High | Low | Low | P2 |
| FM-TAW-009 | Context overflow | Medium | Medium | High | P2 |
| FM-TAW-010 | Circuit cascade | Critical | Low | Low | **P1** |
| FM-TAW-011 | Stale reconnect | Medium | Low | Medium | P3 |
| FM-TAW-012 | Signal inversion | Critical | Medium | None | **P0** |

---

## Verdict

**PROCEED WITH CAUTION**

The Trading Agent Workers design is architecturally sound but has **integration boundary gaps** that will cause silent failures in production. Before implementation:

**Must Fix (P0)**:
1. FM-TAW-006: Enforce fallback distrust action
2. FM-TAW-012: Implement Hidden Hand signal validation

**Should Fix (P1)**:
1. FM-TAW-002: Add heartbeat watchdog
2. FM-TAW-004: Add capability matching
3. FM-TAW-007: Implement processing set fully
4. FM-TAW-010: Separate circuit breaker domains

The Sovereign client is well-defended. Focus defensive effort on the **agent implementation layer**.

---

*Analysis complete. Ready for DevZen validation.*
