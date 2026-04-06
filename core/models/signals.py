from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from core.models.trade import Direction, TimeHorizon


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


class SentimentSignal(BaseModel):
    """Output of Sentiment Agent — direction + confidence from LLM + RAG analysis."""

    direction: Direction
    primary_asset: str
    secondary_assets: list[str] = Field(default_factory=list)
    magnitude: float = Field(ge=0.0, le=1.0)           # how strong is the signal
    confidence: float = Field(ge=0.0, le=1.0)           # how sure is the LLM
    time_horizon: TimeHorizon
    expected_volatility_spike: bool = False
    precedent_summary: str | None = None                # best RAG match summary
    precedent_outcome_pct: float | None = None          # historical return of best match
    llm_reasoning: str = ""
    fireworks_model_used: str = ""
    latency_ms: int = 0
    conflicted: bool = False                            # set by coherence validator


class MarketContext(BaseModel):
    """Output of Market Data Agent — technical snapshot for an asset."""

    asset: str
    price: float
    spread_bps: float
    obi: float = Field(ge=-1.0, le=1.0)                # order book imbalance
    atr14: float                                        # ATR(14) for SL sizing
    rsi14: float = Field(ge=0.0, le=100.0)
    vwap: float
    trend: str                                          # BULLISH / BEARISH / SIDEWAYS / *_WEAKENING
    volatility_regime: str                              # LOW / NORMAL / HIGH / EXTREME
    key_levels: dict[str, list[float]] = Field(
        default_factory=lambda: {"support": [], "resistance": [], "liquidity_pools": []}
    )
    volume_spike_ratio: float = 1.0                    # current_volume / 20-period MA
    timestamp_ms: int = Field(default_factory=_now_ms)

    @property
    def is_spread_acceptable(self, max_spread_bps: float = 5.0) -> bool:
        return self.spread_bps <= max_spread_bps

    @property
    def trend_score(self) -> float:
        """Normalized trend strength for CTS calculation."""
        mapping = {
            "BULLISH": 1.0,
            "BULLISH_WEAKENING": 0.6,
            "SIDEWAYS": 0.3,
            "BEARISH_WEAKENING": 0.6,
            "BEARISH": 1.0,
        }
        return mapping.get(self.trend, 0.3)


class AnomalyType(StrEnum):
    VOLUME_ACCUMULATION_SILENT = "VOLUME_ACCUMULATION_SILENT"
    PRICE_SPIKE_SUDDEN = "PRICE_SPIKE_SUDDEN"
    SPREAD_WIDENING = "SPREAD_WIDENING"
    OBI_IMBALANCE_EXTREME = "OBI_IMBALANCE_EXTREME"
    MULTIVARIATE_OUTLIER = "MULTIVARIATE_OUTLIER"
    LSTM_RECONSTRUCTION_ERROR = "LSTM_RECONSTRUCTION_ERROR"


class AnomalyClassification(StrEnum):
    INSTITUTIONAL_ACCUMULATION = "INSTITUTIONAL_ACCUMULATION"
    INSTITUTIONAL_DISTRIBUTION = "INSTITUTIONAL_DISTRIBUTION"
    NEWS_OVERREACTION = "NEWS_OVERREACTION"
    BULL_TRAP = "BULL_TRAP"
    BEAR_TRAP = "BEAR_TRAP"
    DATA_GLITCH = "DATA_GLITCH"
    GENUINE_BREAKOUT = "GENUINE_BREAKOUT"
    UNKNOWN = "UNKNOWN"


class AnomalyAlert(BaseModel):
    """Output of Anomaly Detection Agent."""

    anomaly_type: AnomalyType
    asset: str
    severity: str                                       # LOW / MEDIUM / HIGH / CRITICAL
    ml_confidence: float = Field(ge=0.0, le=1.0)
    llm_classification: AnomalyClassification = AnomalyClassification.UNKNOWN
    expected_move: dict[str, Any] = Field(
        default_factory=lambda: {"direction": "UNKNOWN", "magnitude_pct": 0.0, "window_seconds": 60}
    )
    false_positive_risk: str = "MEDIUM"                 # LOW / MEDIUM / HIGH
    models_agreed: int = Field(ge=0, le=3)              # how many of 3 models voted
    z_score: float | None = None
    isolation_score: float | None = None
    lstm_reconstruction_error: float | None = None
    timestamp_ms: int = Field(default_factory=_now_ms)
