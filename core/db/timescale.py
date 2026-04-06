from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import structlog

from core.db.postgres import get_pool

log = structlog.get_logger(__name__)


async def insert_ohlcv(
    asset: str,
    timeframe: str,
    bars: list[dict],
) -> None:
    """Bulk insert OHLCV bars into TimescaleDB hypertable."""
    pool = await get_pool()
    records = [
        (
            bar["time"],
            asset,
            timeframe,
            float(bar["open"]),
            float(bar["high"]),
            float(bar["low"]),
            float(bar["close"]),
            float(bar["volume"]),
            float(bar.get("vwap", 0)) or None,
            int(bar.get("trade_count", 0)) or None,
        )
        for bar in bars
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO ohlcv (time, asset, timeframe, open, high, low, close, volume, vwap, trade_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (time, asset, timeframe) DO NOTHING
            """,
            records,
        )


async def fetch_ohlcv(
    asset: str,
    timeframe: str = "1d",
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch OHLCV data for an asset from TimescaleDB."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT time, open, high, low, close, volume, vwap
            FROM ohlcv
            WHERE asset = $1 AND timeframe = $2
            ORDER BY time DESC
            LIMIT $3
            """,
            asset, timeframe, limit,
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "vwap"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").set_index("time")
    return df.astype(float)


async def get_latest_price(asset: str) -> float | None:
    """Get most recent close price from TimescaleDB."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT close FROM ohlcv WHERE asset = $1 ORDER BY time DESC LIMIT 1",
            asset,
        )


async def insert_indicator_snapshot(asset: str, indicators: dict) -> None:
    pool = await get_pool()
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO indicator_snapshots
            (time, asset, rsi14, vwap, atr14, obi, bb_upper, bb_middle, bb_lower, trend, volatility_regime)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (time, asset) DO UPDATE SET
                rsi14=EXCLUDED.rsi14, vwap=EXCLUDED.vwap, atr14=EXCLUDED.atr14,
                obi=EXCLUDED.obi, trend=EXCLUDED.trend, volatility_regime=EXCLUDED.volatility_regime
            """,
            now, asset,
            indicators.get("rsi14"), indicators.get("vwap"), indicators.get("atr14"),
            indicators.get("obi"), indicators.get("bb_upper"), indicators.get("bb_middle"),
            indicators.get("bb_lower"), indicators.get("trend"), indicators.get("volatility_regime"),
        )
