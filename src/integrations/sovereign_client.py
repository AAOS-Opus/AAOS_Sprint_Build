# src/integrations/sovereign_client.py
"""
Sovereign Client for AAOS Trading Agents

Provides async inference capabilities via Sovereign (OpenAI-compatible API).
Implements defensive measures from ensemble resilience review (K2/DevZen).

Defensive Measures:
  - FM-001: Semantic validation for analysis responses (risk_score range check)
  - FM-002: Context cap at 20K tokens with warning logs
  - FM-003: Fallback distrust flag when 7B model responds
  - FM-004: Per-request timeouts (not connection-level)
  - FM-005: Per-model circuit breaker (3 failures in 60s opens circuit)
  - FM-006: Standardized error envelope via SovereignClientError
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass, field
from collections import deque

import httpx

# Configure logger
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SovereignConfig:
    """Sovereign client configuration loaded from environment"""
    url: str = field(default_factory=lambda: os.getenv("SOVEREIGN_URL", "http://localhost:11434/v1"))
    timeout: int = field(default_factory=lambda: int(os.getenv("SOVEREIGN_TIMEOUT", "120")))
    model_primary: str = field(default_factory=lambda: os.getenv("SOVEREIGN_MODEL_PRIMARY", "qwen2.5-coder:32b"))
    model_fallback: str = field(default_factory=lambda: os.getenv("SOVEREIGN_MODEL_FALLBACK", "qwen2.5-coder:7b"))

    # Defensive measure thresholds
    max_context_tokens: int = 20000  # FM-002: Context cap
    circuit_failure_threshold: int = 3  # FM-005: Failures before circuit opens
    circuit_window_seconds: int = 60  # FM-005: Window for failure counting


# =============================================================================
# Exceptions
# =============================================================================

class SovereignClientError(Exception):
    """
    Standardized error envelope for Sovereign client failures (FM-006).

    Wraps all exceptions with context for consistent error propagation.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "SOVEREIGN_ERROR",
        model: Optional[str] = None,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.model = model
        self.original_error = original_error
        self.context = context or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standardized error dict for API responses"""
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "model": self.model,
            "context": self.context
        }


class CircuitOpenError(SovereignClientError):
    """Circuit breaker is open for the requested model"""

    def __init__(self, model: str, failures: int, window_seconds: int):
        super().__init__(
            message=f"Circuit breaker open for model '{model}': {failures} failures in {window_seconds}s",
            error_code="CIRCUIT_OPEN",
            model=model,
            context={"failures": failures, "window_seconds": window_seconds}
        )


class ContextTooLargeError(SovereignClientError):
    """Request exceeds context token limit"""

    def __init__(self, token_count: int, max_tokens: int):
        super().__init__(
            message=f"Context too large: {token_count} tokens exceeds {max_tokens} limit",
            error_code="CONTEXT_TOO_LARGE",
            context={"token_count": token_count, "max_tokens": max_tokens}
        )


class ValidationError(SovereignClientError):
    """Analysis response failed semantic validation"""

    def __init__(self, message: str, field: str, value: Any):
        super().__init__(
            message=message,
            error_code="VALIDATION_FAILED",
            context={"field": field, "value": value}
        )


# =============================================================================
# Circuit Breaker (Per-Model)
# =============================================================================

class CircuitBreaker:
    """
    Per-model circuit breaker (FM-005).

    Tracks failures per model separately. Opens circuit after threshold
    failures within the time window. Prevents cascade failures from
    permanently blacklisting capable models based on transient issues.
    """

    def __init__(self, failure_threshold: int = 3, window_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        # Track failures per model: model -> deque of failure timestamps
        self._failures: Dict[str, deque] = {}
        # Track circuit state per model
        self._open_until: Dict[str, float] = {}

    def _get_failures(self, model: str) -> deque:
        """Get or create failure deque for model"""
        if model not in self._failures:
            self._failures[model] = deque()
        return self._failures[model]

    def _prune_old_failures(self, model: str) -> None:
        """Remove failures outside the time window"""
        failures = self._get_failures(model)
        cutoff = time.time() - self.window_seconds
        while failures and failures[0] < cutoff:
            failures.popleft()

    def is_open(self, model: str) -> bool:
        """Check if circuit is open for model"""
        # Check if in explicit open state
        if model in self._open_until:
            if time.time() < self._open_until[model]:
                return True
            else:
                # Recovery window passed, reset
                del self._open_until[model]
                self._failures[model] = deque()
                logger.info(f"Circuit breaker reset for model '{model}'")

        return False

    def record_failure(self, model: str) -> None:
        """Record a failure for the model"""
        self._prune_old_failures(model)
        failures = self._get_failures(model)
        failures.append(time.time())

        logger.warning(
            f"Sovereign failure for model '{model}': "
            f"{len(failures)}/{self.failure_threshold} in {self.window_seconds}s window"
        )

        # Open circuit if threshold reached
        if len(failures) >= self.failure_threshold:
            self._open_until[model] = time.time() + self.window_seconds
            logger.error(
                f"Circuit breaker OPEN for model '{model}': "
                f"{len(failures)} failures in {self.window_seconds}s"
            )

    def record_success(self, model: str) -> None:
        """Record a success - clears failure history for model"""
        if model in self._failures:
            self._failures[model].clear()
        if model in self._open_until:
            del self._open_until[model]

    def get_failure_count(self, model: str) -> int:
        """Get current failure count for model"""
        self._prune_old_failures(model)
        return len(self._get_failures(model))


# =============================================================================
# Sovereign Client
# =============================================================================

class SovereignClient:
    """
    Async client for AAOS trading agents to call Sovereign for inference.

    Provides:
      - chat_completion: Simple prompt/response
      - analysis_completion: Structured trading analysis with validation
      - stream_completion: Async generator for longer analyses

    All methods implement defensive measures from ensemble review.
    """

    def __init__(self, config: Optional[SovereignConfig] = None):
        self.config = config or SovereignConfig()
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_failure_threshold,
            window_seconds=self.config.circuit_window_seconds
        )
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(
            f"SovereignClient initialized: url={self.config.url}, "
            f"primary={self.config.model_primary}, fallback={self.config.model_fallback}"
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx async client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.url,
                # No default timeout - we use per-request timeouts (FM-004)
            )
        return self._client

    async def close(self) -> None:
        """Close the httpx client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.debug("SovereignClient closed")

    async def __aenter__(self) -> "SovereignClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # -------------------------------------------------------------------------
    # Token Estimation (FM-002)
    # -------------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Approximate token count for context cap enforcement.

        Uses ~4 chars per token heuristic (conservative for English text).
        This is intentionally conservative to avoid FM-002 context saturation.
        """
        return len(text) // 4

    def _check_context_size(self, prompt: str, system_role: Optional[str] = None) -> None:
        """
        Enforce context cap (FM-002).

        Raises ContextTooLargeError if combined prompt + system exceeds limit.
        """
        combined = prompt
        if system_role:
            combined = system_role + prompt

        token_count = self.estimate_tokens(combined)

        if token_count > self.config.max_context_tokens:
            logger.warning(
                f"Context too large: {token_count} tokens exceeds "
                f"{self.config.max_context_tokens} limit"
            )
            raise ContextTooLargeError(token_count, self.config.max_context_tokens)

    # -------------------------------------------------------------------------
    # Role Reinforcement (FM-002 mitigation)
    # -------------------------------------------------------------------------

    @staticmethod
    def _reinforce_role(system_role: str) -> str:
        """
        Prepend role reinforcement reminder to system prompt.

        Mitigates FM-002 (context saturation causing role amnesia) by
        explicitly restating role expectations at the start.
        """
        reinforcement = (
            "IMPORTANT: You are operating as a trading analysis assistant. "
            "Maintain this role throughout your response. "
            "Provide structured, accurate analysis.\n\n"
        )
        return reinforcement + system_role

    # -------------------------------------------------------------------------
    # Model Selection with Fallback
    # -------------------------------------------------------------------------

    def _select_model(self, prefer_primary: bool = True) -> tuple[str, bool]:
        """
        Select model with circuit breaker awareness.

        Returns:
            tuple: (model_name, is_fallback)
        """
        primary = self.config.model_primary
        fallback = self.config.model_fallback

        # Try primary first if circuit is closed
        if prefer_primary and not self._circuit_breaker.is_open(primary):
            return primary, False

        # Check fallback circuit
        if self._circuit_breaker.is_open(fallback):
            # Both circuits open - raise error
            raise CircuitOpenError(
                model="all",
                failures=self._circuit_breaker.get_failure_count(primary),
                window_seconds=self.config.circuit_window_seconds
            )

        logger.warning(f"Primary model '{primary}' circuit open, using fallback '{fallback}'")
        return fallback, True

    # -------------------------------------------------------------------------
    # Chat Completion
    # -------------------------------------------------------------------------

    async def chat_completion(
        self,
        prompt: str,
        system_role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simple prompt/response chat completion.

        Args:
            prompt: User prompt
            system_role: Optional system role (will have reinforcement prepended)

        Returns:
            Dict with keys: content, model, fallback_used, tokens_used

        Raises:
            SovereignClientError: On any failure
        """
        # FM-002: Check context size
        self._check_context_size(prompt, system_role)

        # Build messages
        messages = []
        if system_role:
            messages.append({
                "role": "system",
                "content": self._reinforce_role(system_role)
            })
        messages.append({"role": "user", "content": prompt})

        # Select model with circuit breaker
        model, is_fallback = self._select_model()

        logger.debug(f"chat_completion: model={model}, fallback={is_fallback}")

        try:
            client = await self._get_client()

            # FM-004: Per-request timeout
            response = await client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False
                },
                timeout=httpx.Timeout(self.config.timeout)
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Record success
            self._circuit_breaker.record_success(model)

            result = {
                "content": content,
                "model": model,
                "fallback_used": is_fallback,  # FM-003: Flag for caller awareness
                "tokens_used": data.get("usage", {}).get("total_tokens", 0)
            }

            logger.info(
                f"chat_completion success: model={model}, "
                f"tokens={result['tokens_used']}, fallback={is_fallback}"
            )

            return result

        except httpx.TimeoutException as e:
            self._circuit_breaker.record_failure(model)
            raise SovereignClientError(
                message=f"Timeout after {self.config.timeout}s",
                error_code="TIMEOUT",
                model=model,
                original_error=e
            )
        except httpx.HTTPStatusError as e:
            self._circuit_breaker.record_failure(model)
            raise SovereignClientError(
                message=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                error_code=f"HTTP_{e.response.status_code}",
                model=model,
                original_error=e
            )
        except Exception as e:
            self._circuit_breaker.record_failure(model)
            raise SovereignClientError(
                message=str(e),
                error_code="UNKNOWN",
                model=model,
                original_error=e
            )

    # -------------------------------------------------------------------------
    # Analysis Completion (Structured Output)
    # -------------------------------------------------------------------------

    async def analysis_completion(
        self,
        prompt: str,
        system_role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Structured trading analysis with semantic validation.

        Returns dict with required fields:
          - signal: str (BUY, SELL, HOLD, or custom)
          - confidence: float (0-1)
          - reasoning: str
          - risk_score: float (0-1, validated per FM-001)

        Plus metadata:
          - model: str
          - fallback_used: bool (FM-003)
          - raw_content: str (original response for debugging)

        Raises:
            ValidationError: If response missing required fields or invalid values
            SovereignClientError: On any other failure
        """
        # Enhance system role for structured output
        analysis_system = system_role or ""
        analysis_system = (
            "You are a trading analysis system. "
            "You MUST respond with valid JSON containing these exact fields:\n"
            '{"signal": "BUY|SELL|HOLD", "confidence": 0.0-1.0, '
            '"reasoning": "your analysis", "risk_score": 0.0-1.0}\n\n'
            + analysis_system
        )

        # Get raw completion
        result = await self.chat_completion(prompt, analysis_system)
        raw_content = result["content"]

        # Parse JSON from response
        try:
            # Try to extract JSON from response (handle markdown code blocks)
            content = raw_content.strip()
            if content.startswith("```"):
                # Extract from code block
                lines = content.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```") and not in_block:
                        in_block = True
                        continue
                    elif line.startswith("```") and in_block:
                        break
                    elif in_block:
                        json_lines.append(line)
                content = "\n".join(json_lines)

            analysis = json.loads(content)

        except json.JSONDecodeError as e:
            raise ValidationError(
                message=f"Invalid JSON in response: {str(e)}",
                field="response",
                value=raw_content[:200]
            )

        # FM-001: Semantic validation - required fields
        required_fields = ["signal", "confidence", "reasoning", "risk_score"]
        for field in required_fields:
            if field not in analysis:
                raise ValidationError(
                    message=f"Missing required field: {field}",
                    field=field,
                    value=None
                )

        # FM-001: Semantic validation - risk_score range
        risk_score = analysis["risk_score"]
        try:
            risk_score = float(risk_score)
        except (TypeError, ValueError):
            raise ValidationError(
                message=f"risk_score must be numeric, got: {type(risk_score).__name__}",
                field="risk_score",
                value=risk_score
            )

        if not (0.0 <= risk_score <= 1.0):
            raise ValidationError(
                message=f"risk_score must be in range [0, 1], got: {risk_score}",
                field="risk_score",
                value=risk_score
            )

        # FM-001: Validate confidence range too
        confidence = analysis["confidence"]
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            raise ValidationError(
                message=f"confidence must be numeric, got: {type(confidence).__name__}",
                field="confidence",
                value=confidence
            )

        if not (0.0 <= confidence <= 1.0):
            raise ValidationError(
                message=f"confidence must be in range [0, 1], got: {confidence}",
                field="confidence",
                value=confidence
            )

        # Build validated result
        validated_result = {
            "signal": str(analysis["signal"]).upper(),
            "confidence": confidence,
            "reasoning": str(analysis["reasoning"]),
            "risk_score": risk_score,
            "model": result["model"],
            "fallback_used": result["fallback_used"],
            "raw_content": raw_content
        }

        logger.info(
            f"analysis_completion success: signal={validated_result['signal']}, "
            f"confidence={confidence:.2f}, risk_score={risk_score:.2f}, "
            f"model={result['model']}, fallback={result['fallback_used']}"
        )

        return validated_result

    # -------------------------------------------------------------------------
    # Stream Completion (Async Generator)
    # -------------------------------------------------------------------------

    async def stream_completion(
        self,
        prompt: str,
        system_role: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming completion for longer analyses.

        Yields dicts with keys:
          - chunk: str (content delta)
          - done: bool
          - model: str
          - fallback_used: bool (FM-003)

        Final yield includes:
          - full_content: str (accumulated content)
          - tokens_used: int (if available)

        Raises:
            SovereignClientError: On any failure
        """
        # FM-002: Check context size
        self._check_context_size(prompt, system_role)

        # Build messages
        messages = []
        if system_role:
            messages.append({
                "role": "system",
                "content": self._reinforce_role(system_role)
            })
        messages.append({"role": "user", "content": prompt})

        # Select model with circuit breaker
        model, is_fallback = self._select_model()

        logger.debug(f"stream_completion: model={model}, fallback={is_fallback}")

        accumulated_content = ""

        try:
            client = await self._get_client()

            # FM-004: Per-request timeout for initial connection
            async with client.stream(
                "POST",
                "/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True
                },
                timeout=httpx.Timeout(self.config.timeout)
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # SSE format: "data: {...}"
                    if line.startswith("data: "):
                        data_str = line[6:]

                        if data_str.strip() == "[DONE]":
                            # Final message
                            yield {
                                "chunk": "",
                                "done": True,
                                "model": model,
                                "fallback_used": is_fallback,
                                "full_content": accumulated_content
                            }
                            break

                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                accumulated_content += content
                                yield {
                                    "chunk": content,
                                    "done": False,
                                    "model": model,
                                    "fallback_used": is_fallback
                                }
                        except json.JSONDecodeError:
                            # Skip malformed SSE lines
                            logger.debug(f"Skipping malformed SSE line: {line[:100]}")
                            continue

                # Record success
                self._circuit_breaker.record_success(model)

                logger.info(
                    f"stream_completion success: model={model}, "
                    f"chars={len(accumulated_content)}, fallback={is_fallback}"
                )

        except httpx.TimeoutException as e:
            self._circuit_breaker.record_failure(model)
            raise SovereignClientError(
                message=f"Stream timeout after {self.config.timeout}s",
                error_code="TIMEOUT",
                model=model,
                original_error=e
            )
        except httpx.HTTPStatusError as e:
            self._circuit_breaker.record_failure(model)
            raise SovereignClientError(
                message=f"HTTP {e.response.status_code}",
                error_code=f"HTTP_{e.response.status_code}",
                model=model,
                original_error=e
            )
        except Exception as e:
            self._circuit_breaker.record_failure(model)
            raise SovereignClientError(
                message=str(e),
                error_code="UNKNOWN",
                model=model,
                original_error=e
            )

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Sovereign connectivity and model availability.

        Returns:
            Dict with status, models, and circuit breaker states
        """
        client = await self._get_client()

        result = {
            "status": "unknown",
            "url": self.config.url,
            "models": {
                "primary": {
                    "name": self.config.model_primary,
                    "circuit_open": self._circuit_breaker.is_open(self.config.model_primary),
                    "failure_count": self._circuit_breaker.get_failure_count(self.config.model_primary)
                },
                "fallback": {
                    "name": self.config.model_fallback,
                    "circuit_open": self._circuit_breaker.is_open(self.config.model_fallback),
                    "failure_count": self._circuit_breaker.get_failure_count(self.config.model_fallback)
                }
            }
        }

        try:
            response = await client.get(
                "/models",
                timeout=httpx.Timeout(10.0)
            )
            response.raise_for_status()

            models_data = response.json()
            available_models = [m.get("id", m.get("name", "unknown")) for m in models_data.get("data", [])]

            result["status"] = "healthy"
            result["available_models"] = available_models

            logger.info(f"Sovereign health check passed: {len(available_models)} models available")

        except Exception as e:
            result["status"] = "unhealthy"
            result["error"] = str(e)
            logger.error(f"Sovereign health check failed: {e}")

        return result


# =============================================================================
# Connection Test
# =============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    async def test_connection():
        """Test Sovereign connection and basic functionality"""
        print("=" * 60)
        print("Sovereign Client Connection Test")
        print("=" * 60)

        async with SovereignClient() as client:
            # Health check
            print("\n[1/4] Health Check...")
            health = await client.health_check()
            print(f"  Status: {health['status']}")
            print(f"  URL: {health['url']}")
            print(f"  Primary model: {health['models']['primary']['name']}")
            print(f"  Fallback model: {health['models']['fallback']['name']}")

            if health["status"] != "healthy":
                print(f"\n  ERROR: {health.get('error', 'Unknown error')}")
                print("\n  Check that Sovereign is running at the configured URL.")
                return

            print(f"  Available models: {health.get('available_models', [])}")

            # Chat completion test
            print("\n[2/4] Chat Completion Test...")
            try:
                result = await client.chat_completion(
                    prompt="What is 2 + 2? Reply with just the number.",
                    system_role="You are a helpful math assistant."
                )
                print(f"  Response: {result['content'][:100]}")
                print(f"  Model: {result['model']}")
                print(f"  Fallback used: {result['fallback_used']}")
                print(f"  Tokens: {result['tokens_used']}")
            except SovereignClientError as e:
                print(f"  ERROR: {e.message}")

            # Analysis completion test
            print("\n[3/4] Analysis Completion Test...")
            try:
                result = await client.analysis_completion(
                    prompt="Analyze BTC/USD at $50,000. Provide a trading signal.",
                    system_role="You are a cryptocurrency trading analyst."
                )
                print(f"  Signal: {result['signal']}")
                print(f"  Confidence: {result['confidence']:.2f}")
                print(f"  Risk Score: {result['risk_score']:.2f}")
                print(f"  Reasoning: {result['reasoning'][:100]}...")
                print(f"  Model: {result['model']}")
                print(f"  Fallback used: {result['fallback_used']}")
            except SovereignClientError as e:
                print(f"  ERROR: {e.message}")

            # Stream completion test
            print("\n[4/4] Stream Completion Test...")
            try:
                chunks = []
                async for chunk in client.stream_completion(
                    prompt="Count from 1 to 5.",
                    system_role="You are a helpful assistant."
                ):
                    if chunk["chunk"]:
                        chunks.append(chunk["chunk"])
                    if chunk["done"]:
                        print(f"  Full response: {chunk['full_content'][:100]}")
                        print(f"  Model: {chunk['model']}")
                        print(f"  Fallback used: {chunk['fallback_used']}")
            except SovereignClientError as e:
                print(f"  ERROR: {e.message}")

        print("\n" + "=" * 60)
        print("Connection test complete")
        print("=" * 60)

    asyncio.run(test_connection())
