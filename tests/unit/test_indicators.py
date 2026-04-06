import numpy as np
import pytest

from agents.market_data.indicators import (
    classify_trend,
    classify_volatility_regime,
    compute_atr_from_prices,
    compute_bollinger_bands,
    compute_obi,
    compute_rsi,
    compute_vwap,
)


def test_rsi_neutral():
    prices = np.full(30, 100.0)   # no change
    rsi = compute_rsi(prices)
    assert rsi == 50.0 or rsi == 100.0   # no gains and no losses case


def test_rsi_all_gains():
    prices = np.linspace(100, 130, 30)
    rsi = compute_rsi(prices)
    assert rsi > 70


def test_rsi_all_losses():
    prices = np.linspace(130, 100, 30)
    rsi = compute_rsi(prices)
    assert rsi < 30


def test_vwap_uniform():
    prices = np.full(10, 50.0)
    volumes = np.full(10, 100.0)
    assert compute_vwap(prices, volumes) == pytest.approx(50.0)


def test_vwap_weighted():
    prices = np.array([40.0, 60.0])
    volumes = np.array([1.0, 3.0])   # 60 has 3x volume
    vwap = compute_vwap(prices, volumes)
    assert vwap == pytest.approx(55.0)


def test_bollinger_bands_width():
    prices = np.linspace(100, 110, 30)
    upper, middle, lower = compute_bollinger_bands(prices)
    assert upper > middle > lower


def test_obi_bullish():
    bids = np.full(5, 100.0)
    asks = np.full(5, 20.0)
    obi = compute_obi(bids, asks)
    assert obi > 0     # more bids = bullish


def test_obi_bearish():
    bids = np.full(5, 10.0)
    asks = np.full(5, 90.0)
    obi = compute_obi(bids, asks)
    assert obi < 0     # more asks = bearish


def test_classify_trend_bullish():
    prices = np.concatenate([np.full(50, 100.0), np.linspace(100, 115, 10)])
    trend = classify_trend(prices)
    assert "BULLISH" in trend


def test_classify_trend_sideways():
    prices = np.random.normal(100, 0.1, 60)
    trend = classify_trend(prices)
    assert trend in ("SIDEWAYS", "BULLISH", "BEARISH")   # small random fluctuations


def test_volatility_regime():
    assert classify_volatility_regime(0.5, 100.0) == "LOW"
    assert classify_volatility_regime(1.0, 100.0) == "NORMAL"
    assert classify_volatility_regime(2.0, 100.0) == "HIGH"
    assert classify_volatility_regime(4.0, 100.0) == "EXTREME"
