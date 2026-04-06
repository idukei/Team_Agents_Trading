from __future__ import annotations

import numpy as np
import pandas as pd

import structlog

log = structlog.get_logger(__name__)


def williams_fractal_pivots(
    highs: np.ndarray,
    lows: np.ndarray,
    window: int = 2,
) -> tuple[list[float], list[float]]:
    """
    Williams Fractal algorithm: identifies pivot highs and lows.
    A pivot high = local max surrounded by `window` lower highs on each side.
    A pivot low = local min surrounded by `window` higher lows on each side.

    Returns (resistance_levels, support_levels).
    """
    n = len(highs)
    if n < 2 * window + 1:
        return [], []

    resistance = []
    support = []

    for i in range(window, n - window):
        # Pivot high
        if all(highs[i] > highs[i - j] for j in range(1, window + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, window + 1)):
            resistance.append(float(highs[i]))

        # Pivot low
        if all(lows[i] < lows[i - j] for j in range(1, window + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, window + 1)):
            support.append(float(lows[i]))

    # Return only the 3 most recent levels
    return resistance[-3:], support[-3:]


def get_key_levels(df: pd.DataFrame) -> dict[str, list[float]]:
    """
    Compute support, resistance, and liquidity pool levels from OHLCV DataFrame.

    df must have columns: open, high, low, close, volume.
    """
    if df.empty or len(df) < 10:
        return {"support": [], "resistance": [], "liquidity_pools": []}

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values

    resistance, support = williams_fractal_pivots(highs, lows)

    # Liquidity pools: price levels with volume concentration (round numbers ± high volume)
    liquidity_pools = _find_liquidity_pools(closes, volumes)

    return {
        "support": sorted(support),
        "resistance": sorted(resistance, reverse=True),
        "liquidity_pools": liquidity_pools,
    }


def _find_liquidity_pools(prices: np.ndarray, volumes: np.ndarray, bins: int = 20) -> list[float]:
    """
    Identify price levels with high volume concentration using histogram approach.
    Returns top-3 liquidity pool price levels.
    """
    if len(prices) < 10:
        return []

    counts, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
    top_indices = np.argsort(counts)[-3:][::-1]
    pools = [
        round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 2)
        for i in top_indices
        if counts[i] > 0
    ]
    return sorted(pools)


def nearest_level(price: float, levels: list[float], direction: str = "below") -> float | None:
    """Find the nearest support (below) or resistance (above) level to current price."""
    if not levels:
        return None
    if direction == "below":
        candidates = [l for l in levels if l < price]
        return max(candidates) if candidates else None
    else:
        candidates = [l for l in levels if l > price]
        return min(candidates) if candidates else None
