from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from core.models.trade import Direction


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


class Position(BaseModel):
    """An open trading position."""

    trade_id: str
    asset: str
    direction: Direction
    entry_price: float
    current_size_usd: float
    original_size_usd: float
    unrealized_pnl: float = 0.0
    entry_timestamp_ms: int = Field(default_factory=_now_ms)
    stop_loss: float = 0.0
    take_profit: list[float] = Field(default_factory=list)
    max_hold_until_ms: int | None = None
    tp1_hit: bool = False                      # 50% already closed


class PortfolioState(BaseModel):
    """Current portfolio snapshot, updated after each trade event."""

    capital_usd: float
    open_positions: list[Position] = Field(default_factory=list)
    daily_pnl_usd: float = 0.0
    daily_pnl_pct: float = 0.0
    total_realized_pnl_usd: float = 0.0
    circuit_breaker_active: bool = False
    circuit_breaker_reason: str | None = None
    trades_today: int = 0
    winning_trades_today: int = 0

    @property
    def total_risk_usd(self) -> float:
        """Sum of risk across all open positions."""
        return sum(
            abs(p.entry_price - p.stop_loss) / p.entry_price * p.current_size_usd
            for p in self.open_positions
            if p.stop_loss > 0
        )

    @property
    def total_exposure_pct(self) -> float:
        """Total open exposure as % of capital."""
        total = sum(p.current_size_usd for p in self.open_positions)
        return total / self.capital_usd if self.capital_usd > 0 else 0.0

    @property
    def win_rate_today(self) -> float:
        if self.trades_today == 0:
            return 0.0
        return self.winning_trades_today / self.trades_today


class SessionMetadata(BaseModel):
    """Per-pipeline-run metadata — reset for each EventTrigger."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_mode: str = "FULL"             # FULL | LIGHT | TECHNICAL_ONLY
    started_at_ms: int = Field(default_factory=_now_ms)
    fireworks_tokens_used: int = 0
    fireworks_cost_usd: float = 0.0
    agent_latencies_ms: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    degraded_agents: list[str] = Field(default_factory=list)

    @property
    def elapsed_ms(self) -> int:
        return _now_ms() - self.started_at_ms

    def record_agent_latency(self, agent: str, latency_ms: int) -> None:
        self.agent_latencies_ms[agent] = latency_ms

    def record_llm_usage(self, tokens: int, cost_usd: float) -> None:
        self.fireworks_tokens_used += tokens
        self.fireworks_cost_usd += cost_usd
