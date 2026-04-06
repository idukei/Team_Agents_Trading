from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


class Direction(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class TimeHorizon(StrEnum):
    IMMEDIATE = "IMMEDIATE"     # seconds
    FIVE_MIN = "5MIN"
    THIRTY_MIN = "30MIN"
    ONE_HOUR = "1H"


class OrderType(StrEnum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class StrategyType(StrEnum):
    POLITICAL_SCALP = "POLITICAL_SCALP"
    ECONOMIC_EVENT_SCALP = "ECONOMIC_EVENT_SCALP"
    ANOMALY_BREAKOUT = "ANOMALY_BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"


class RiskDecisionStatus(StrEnum):
    APPROVED = "APPROVED"
    APPROVED_REDUCED = "APPROVED_REDUCED"
    REJECTED = "REJECTED"


class TradeProposal(BaseModel):
    """Proposed trade from Strategy Agent, validated by Risk Agent."""

    trade_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str | None = None
    asset: str
    direction: Direction
    entry_type: OrderType = OrderType.LIMIT
    entry_price: float = Field(gt=0)
    position_size_usd: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: list[float] = Field(min_length=1, max_length=3)
    max_hold_seconds: int = Field(gt=0, default=600)
    exit_triggers: list[str] = Field(default_factory=list)
    cts_score: float = Field(ge=0.0, le=1.0)
    strategy_type: StrategyType = StrategyType.POLITICAL_SCALP
    reasoning_summary: str = ""
    created_at_ms: int = Field(default_factory=_now_ms)

    @model_validator(mode="after")
    def validate_sl_tp_direction(self) -> "TradeProposal":
        if self.direction == Direction.LONG:
            if self.stop_loss >= self.entry_price:
                raise ValueError(f"LONG SL ({self.stop_loss}) must be below entry ({self.entry_price})")
            if not all(tp > self.entry_price for tp in self.take_profit):
                raise ValueError("All LONG TPs must be above entry price")
        elif self.direction == Direction.SHORT:
            if self.stop_loss <= self.entry_price:
                raise ValueError(f"SHORT SL ({self.stop_loss}) must be above entry ({self.entry_price})")
            if not all(tp < self.entry_price for tp in self.take_profit):
                raise ValueError("All SHORT TPs must be below entry price")
        return self

    @property
    def risk_usd(self) -> float:
        """Dollar risk based on entry vs stop-loss."""
        return abs(self.entry_price - self.stop_loss) / self.entry_price * self.position_size_usd

    @property
    def primary_rr(self) -> float:
        """Risk:Reward ratio to first TP."""
        reward = abs(self.take_profit[0] - self.entry_price)
        risk = abs(self.stop_loss - self.entry_price)
        return reward / risk if risk > 0 else 0.0


class RiskDecision(BaseModel):
    """Risk Management Agent decision on a TradeProposal."""

    trade_id: str
    status: RiskDecisionStatus
    approved_size_usd: float = 0.0
    rejection_reason: str | None = None
    adjustments: dict[str, Any] = Field(default_factory=dict)
    var_95: float | None = None
    portfolio_correlation: float | None = None
    layer_results: dict[str, bool] = Field(default_factory=dict)  # layer1/2/3 pass/fail


class ExecutionResult(BaseModel):
    """Order execution outcome from Execution Agent."""

    trade_id: str
    order_id: str
    filled_price: float
    filled_qty: float
    fill_timestamp_ms: int = Field(default_factory=_now_ms)
    slippage_bps: float = 0.0
    # Trade lifecycle flags
    tp1_hit: bool = False
    tp2_hit: bool = False
    sl_hit: bool = False
    timeout_exit: bool = False
    counter_signal_exit: bool = False
    exit_price: float | None = None
    exit_timestamp_ms: int | None = None
    pnl_usd: float | None = None
    pnl_pct: float | None = None

    @property
    def is_closed(self) -> bool:
        return any([self.tp1_hit and self.tp2_hit, self.sl_hit, self.timeout_exit, self.counter_signal_exit])

    @property
    def duration_seconds(self) -> float | None:
        if self.exit_timestamp_ms:
            return (self.exit_timestamp_ms - self.fill_timestamp_ms) / 1000
        return None
