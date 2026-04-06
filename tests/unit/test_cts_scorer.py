import pytest
from agents.strategy.cts_scorer import compute_cts, cts_to_size_multiplier
from core.models.signals import AnomalyAlert, AnomalyType, MarketContext, SentimentSignal
from core.models.trade import Direction, TimeHorizon


def make_sentiment(direction=Direction.SHORT, magnitude=0.8, confidence=0.75, conflicted=False):
    return SentimentSignal(
        direction=direction,
        primary_asset="NVDA",
        magnitude=magnitude,
        confidence=confidence,
        time_horizon=TimeHorizon.IMMEDIATE,
        llm_reasoning="test",
        precedent_outcome_pct=-2.5,
    )


def make_market(trend="BEARISH", rsi=62.0, obi=-0.3):
    return MarketContext(
        asset="NVDA",
        price=876.0,
        spread_bps=1.2,
        obi=obi,
        atr14=15.0,
        rsi14=rsi,
        vwap=868.0,
        trend=trend,
        volatility_regime="HIGH",
        volume_spike_ratio=2.1,
    )


def test_cts_full_signals_high_score():
    sentiment = make_sentiment(Direction.SHORT)
    market = make_market("BEARISH", rsi=62.0, obi=-0.3)
    cts, components = compute_cts(sentiment, None, market, Direction.SHORT)
    assert cts >= 0.52, f"Expected CTS >= 0.52, got {cts}"
    assert "sentiment" in components
    assert "market" in components


def test_cts_conflicted_sentiment_zeroed():
    sentiment = make_sentiment(conflicted=True)
    market = make_market()
    cts, components = compute_cts(sentiment, None, market, Direction.SHORT)
    assert components["sentiment"] == 0.0


def test_cts_direction_mismatch_lowers_score():
    sentiment_long = make_sentiment(Direction.LONG, magnitude=0.9, confidence=0.9)
    cts_long_on_short, _ = compute_cts(sentiment_long, None, None, Direction.SHORT)
    cts_long_on_long, _ = compute_cts(sentiment_long, None, None, Direction.LONG)
    assert cts_long_on_long > cts_long_on_short


def test_cts_size_multiplier():
    assert cts_to_size_multiplier(0.75) == 1.0
    assert cts_to_size_multiplier(0.60) == 0.5
    assert cts_to_size_multiplier(0.40) == 0.0


def test_cts_rsi_overbought_penalty():
    market_overbought = make_market("BULLISH", rsi=80.0, obi=0.3)
    market_normal = make_market("BULLISH", rsi=55.0, obi=0.3)
    cts_over, _ = compute_cts(None, None, market_overbought, Direction.LONG)
    cts_normal, _ = compute_cts(None, None, market_normal, Direction.LONG)
    assert cts_normal >= cts_over
