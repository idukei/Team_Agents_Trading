from __future__ import annotations

import structlog

from core.config import settings
from core.models.signals import AnomalyAlert, MarketContext, SentimentSignal
from core.models.trade import Direction

log = structlog.get_logger(__name__)

# Default CTS weights (updated daily by post_trade_analyst)
DEFAULT_WEIGHTS = {
    "sentiment": 0.32,
    "anomaly": 0.28,
    "market": 0.20,
    "precedent": 0.12,
    "obi": 0.08,
}


def compute_cts(
    sentiment: SentimentSignal | None,
    anomaly: AnomalyAlert | None,
    market_ctx: MarketContext | None,
    trade_direction: Direction,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict]:
    """
    Compute Composite Trade Score (CTS).

    Returns (cts_score 0-1, component_breakdown dict).
    """
    w = weights or DEFAULT_WEIGHTS

    components = {}

    # Component 1: Sentiment signal (magnitude × confidence)
    if sentiment and not sentiment.conflicted:
        # Direction alignment bonus
        direction_match = (
            (sentiment.direction == Direction.LONG and trade_direction == Direction.LONG) or
            (sentiment.direction == Direction.SHORT and trade_direction == Direction.SHORT)
        )
        sent_score = sentiment.magnitude * sentiment.confidence if direction_match else 0.0
        components["sentiment"] = round(sent_score, 4)
    else:
        components["sentiment"] = 0.0
        if sentiment and sentiment.conflicted:
            log.debug("cts_sentiment_conflicted_zeroed")

    # Component 2: Anomaly signal (ML confidence × direction alignment)
    if anomaly:
        expected_dir = anomaly.expected_move.get("direction", "UNKNOWN")
        anomaly_match = (
            (expected_dir == "UP" and trade_direction == Direction.LONG) or
            (expected_dir == "DOWN" and trade_direction == Direction.SHORT)
        )
        anomaly_score = anomaly.ml_confidence if anomaly_match else anomaly.ml_confidence * 0.3
        # Reduce score for high false positive risk
        if anomaly.false_positive_risk == "HIGH":
            anomaly_score *= 0.5
        components["anomaly"] = round(anomaly_score, 4)
    else:
        components["anomaly"] = 0.0

    # Component 3: Market technical context
    if market_ctx:
        trend_score = market_ctx.trend_score
        # RSI confirmation: avoid chasing overbought longs or oversold shorts
        rsi_penalty = 0.0
        if trade_direction == Direction.LONG and market_ctx.rsi14 > 75:
            rsi_penalty = 0.2
        elif trade_direction == Direction.SHORT and market_ctx.rsi14 < 25:
            rsi_penalty = 0.2
        market_score = max(0.0, trend_score - rsi_penalty)
        components["market"] = round(market_score, 4)
    else:
        components["market"] = 0.0

    # Component 4: Historical precedent outcome
    if sentiment and sentiment.precedent_outcome_pct is not None:
        pct = sentiment.precedent_outcome_pct
        # Normalize: 3% return → 1.0, 0% → 0.5, negative → penalize
        normalized = max(0.0, min(1.0, (pct + 3.0) / 6.0))
        components["precedent"] = round(normalized, 4)
    else:
        components["precedent"] = 0.5   # neutral when no precedent

    # Component 5: OBI alignment
    if market_ctx:
        obi = market_ctx.obi
        obi_match = (
            (obi > 0.2 and trade_direction == Direction.LONG) or
            (obi < -0.2 and trade_direction == Direction.SHORT)
        )
        obi_score = min(1.0, abs(obi) * 2) if obi_match else 0.1
        components["obi"] = round(obi_score, 4)
    else:
        components["obi"] = 0.3   # neutral

    # Weighted sum
    cts = sum(components[k] * w.get(k, 0) for k in components)
    cts = round(max(0.0, min(1.0, cts)), 4)

    log.debug("cts_computed", cts=cts, components=components)
    return cts, components


def cts_to_size_multiplier(cts: float) -> float:
    """Map CTS score to position size multiplier."""
    if cts >= settings.cts_execute:
        return 1.0
    elif cts >= settings.cts_half_size:
        return 0.5
    return 0.0
