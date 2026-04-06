from __future__ import annotations


class TeamTradeError(Exception):
    """Base exception for all TeamTrade errors."""


# ── Configuration ──────────────────────────────────────────────────────────────
class ConfigurationError(TeamTradeError):
    """Missing or invalid configuration."""


# ── Data / validation ─────────────────────────────────────────────────────────
class DataValidationError(TeamTradeError):
    """Incoming data failed Pydantic validation."""


class StaleDataError(TeamTradeError):
    """Data is too old to be used safely."""


# ── Infrastructure ─────────────────────────────────────────────────────────────
class DatabaseError(TeamTradeError):
    """PostgreSQL / TimescaleDB error."""


class RedisError(TeamTradeError):
    """Redis Streams error."""


class QdrantError(TeamTradeError):
    """Qdrant vector DB error."""


# ── External APIs ──────────────────────────────────────────────────────────────
class AlpacaError(TeamTradeError):
    """Alpaca Markets API error."""


class AlpacaWebSocketError(AlpacaError):
    """Alpaca WebSocket disconnection or stream error."""


class FireworksError(TeamTradeError):
    """Fireworks.ai LLM API error."""


class FireworksBudgetExceededError(FireworksError):
    """Daily Fireworks.ai token budget exceeded."""


class RSSError(TeamTradeError):
    """RSS feed or RSSHub error."""


# ── Circuit breaker ────────────────────────────────────────────────────────────
class CircuitOpenError(TeamTradeError):
    """Circuit breaker is open; operation not allowed."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Circuit breaker open: {reason}")


# ── Trading pipeline ───────────────────────────────────────────────────────────
class PipelineError(TeamTradeError):
    """Generic pipeline processing error."""


class RegimeFilterError(PipelineError):
    """Trade rejected by regime filter (hard rule)."""


class RiskRejectionError(PipelineError):
    """Trade rejected by Risk Management Agent."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Risk rejection: {reason}")


class ExecutionError(PipelineError):
    """Order placement or management error."""


class InsufficientDataError(PipelineError):
    """Not enough market data to compute indicators."""
