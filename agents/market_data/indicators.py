from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import settings


def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """RSI using Wilder's smoothing. Returns 0-100."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Wilder's smoothing for remaining periods
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1 + rs)), 3)


def compute_vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Volume-Weighted Average Price (intraday fair value reference)."""
    total_vol = np.sum(volumes)
    if total_vol == 0:
        return float(prices[-1]) if len(prices) > 0 else 0.0
    return float(np.sum(prices * volumes) / total_vol)


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Average True Range using Wilder's smoothing."""
    if len(closes) < 2:
        return float(highs[-1] - lows[-1]) if len(highs) > 0 else 0.0

    tr_list = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)

    tr_arr = np.array(tr_list)
    if len(tr_arr) < period:
        return float(np.mean(tr_arr))

    atr = float(np.mean(tr_arr[:period]))
    for i in range(period, len(tr_arr)):
        atr = (atr * (period - 1) + tr_arr[i]) / period
    return round(atr, 6)


def compute_atr_from_prices(prices: np.ndarray, period: int = 14) -> float:
    """Approximate ATR from price array only (tick data, no OHLC)."""
    if len(prices) < 2:
        return 0.0
    returns = np.abs(np.diff(prices))
    window = min(period, len(returns))
    return float(np.mean(returns[-window:]))


def compute_bollinger_bands(
    prices: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[float, float, float]:
    """Returns (upper, middle, lower) Bollinger Bands."""
    if len(prices) < period:
        p = float(prices[-1]) if len(prices) > 0 else 0.0
        return p, p, p
    window = prices[-period:]
    middle = float(np.mean(window))
    std = float(np.std(window, ddof=1))
    return (
        round(middle + std_dev * std, 6),
        round(middle, 6),
        round(middle - std_dev * std, 6),
    )


def compute_obi(bids: np.ndarray, asks: np.ndarray, periods: int = 5) -> float:
    """
    Order Book Imbalance (OBI) averaged over last N ticks.
    Range: -1 (full ask pressure) to +1 (full bid pressure).
    """
    if len(bids) == 0 or len(asks) == 0:
        return 0.0
    n = min(periods, len(bids), len(asks))
    b = bids[-n:]
    a = asks[-n:]
    total = b + a
    with np.errstate(divide="ignore", invalid="ignore"):
        obi_vals = np.where(total > 0, (b - a) / total, 0.0)
    return round(float(np.mean(obi_vals)), 4)


def classify_trend(prices: np.ndarray, short_period: int = 10, long_period: int = 50) -> str:
    """Classify trend as BULLISH / BEARISH / SIDEWAYS with _WEAKENING suffix."""
    if len(prices) < long_period:
        return "SIDEWAYS"

    short_ma = float(np.mean(prices[-short_period:]))
    long_ma = float(np.mean(prices[-long_period:]))

    # Slope of short MA
    if len(prices) >= short_period + 5:
        slope = float(np.mean(prices[-short_period:]) - np.mean(prices[-(short_period + 5):-5]))
    else:
        slope = 0.0

    pct_diff = (short_ma - long_ma) / long_ma * 100

    if pct_diff > 0.3:
        return "BULLISH_WEAKENING" if slope < 0 else "BULLISH"
    elif pct_diff < -0.3:
        return "BEARISH_WEAKENING" if slope > 0 else "BEARISH"
    return "SIDEWAYS"


def classify_volatility_regime(atr: float, price: float, atr_pct_thresholds: tuple = (0.5, 1.5, 3.0)) -> str:
    """Classify volatility as LOW / NORMAL / HIGH / EXTREME."""
    if price <= 0:
        return "NORMAL"
    atr_pct = (atr / price) * 100
    low_t, high_t, extreme_t = atr_pct_thresholds
    if atr_pct < low_t:
        return "LOW"
    elif atr_pct < high_t:
        return "NORMAL"
    elif atr_pct < extreme_t:
        return "HIGH"
    return "EXTREME"
