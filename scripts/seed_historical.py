"""
Seed TimescaleDB with 2 years of historical OHLCV data via yfinance.
Run: uv run python scripts/seed_historical.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf
import structlog

from core.config import settings
from core.db.postgres import close_pool, get_pool
from core.db.timescale import insert_ohlcv
from core.logging import configure_logging

log = structlog.get_logger(__name__)

TIMEFRAMES = {
    "1d": ("2y", "1d"),
    "1h": ("1y", "1h"),
    "5m": ("60d", "5m"),
}


async def seed_asset(asset: str) -> None:
    ticker_sym = asset.replace("/", "-")   # BTC/USD → BTC-USD for yfinance

    for timeframe, (period, interval) in TIMEFRAMES.items():
        try:
            log.info("fetching", asset=asset, timeframe=timeframe, period=period)
            df = yf.download(ticker_sym, period=period, interval=interval, auto_adjust=True, progress=False)

            if df.empty:
                log.warning("no_data", asset=asset, timeframe=timeframe)
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Rename Datetime/Date column to time
            time_col = next((c for c in df.columns if c in ("datetime", "date")), None)
            if time_col:
                df = df.rename(columns={time_col: "time"})

            # Ensure timezone aware
            if df["time"].dt.tz is None:
                df["time"] = df["time"].dt.tz_localize("UTC")

            bars = df[["time", "open", "high", "low", "close", "volume"]].dropna().to_dict("records")
            await insert_ohlcv(asset, timeframe, bars)
            log.info("seeded", asset=asset, timeframe=timeframe, rows=len(bars))

        except Exception as e:
            log.error("seed_failed", asset=asset, timeframe=timeframe, error=str(e))


async def main() -> None:
    configure_logging()
    log.info("seed_starting", assets=settings.all_assets)

    await get_pool()

    tasks = [seed_asset(asset) for asset in settings.all_assets]
    await asyncio.gather(*tasks)

    await close_pool()
    log.info("seed_complete")


if __name__ == "__main__":
    asyncio.run(main())
