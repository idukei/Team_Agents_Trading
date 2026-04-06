from __future__ import annotations

import time

import structlog

from agents.market_data.indicators import (
    classify_trend,
    classify_volatility_regime,
    compute_atr_from_prices,
    compute_bollinger_bands,
    compute_obi,
    compute_rsi,
    compute_vwap,
)
from agents.market_data.levels import get_key_levels
from agents.market_data.stream import buffer_registry
from core.db.timescale import fetch_ohlcv, insert_indicator_snapshot
from core.models.events import EventTrigger
from core.models.signals import MarketContext
from graph.state import TradingState

log = structlog.get_logger(__name__)


async def compute_market_context(asset: str, trigger: EventTrigger | None = None) -> MarketContext | None:
    """
    Core computation: builds MarketContext for an asset using:
    1. Real-time buffer (from Alpaca WebSocket)
    2. Historical OHLCV from TimescaleDB (for key levels)
    """
    buf = buffer_registry.get(asset)
    if buf is None or len(buf) < 20:
        log.warning("market_data_insufficient_buffer", asset=asset, size=len(buf) if buf else 0)
        return None

    prices = buf.prices
    volumes = buf.volumes
    bids = buf.bids
    asks = buf.asks
    latest = buf.latest

    if latest is None:
        return None

    # Indicators from real-time buffer
    rsi = compute_rsi(prices, period=14)
    vwap = compute_vwap(prices, volumes)
    atr = compute_atr_from_prices(prices, period=14)
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(prices)
    obi = compute_obi(bids, asks)
    trend = classify_trend(prices)
    vol_regime = classify_volatility_regime(atr, latest.price)
    vol_ratio = buf.volume_ma(5) / buf.volume_ma(20) if buf.volume_ma(20) > 0 else 1.0

    # Key levels from historical OHLCV (TimescaleDB)
    key_levels: dict = {"support": [], "resistance": [], "liquidity_pools": []}
    try:
        df = await fetch_ohlcv(asset, timeframe="1d", limit=100)
        if not df.empty:
            key_levels = get_key_levels(df)
    except Exception as e:
        log.warning("market_data_levels_error", asset=asset, error=str(e))

    ctx = MarketContext(
        asset=asset,
        price=latest.price,
        spread_bps=latest.spread_bps,
        obi=max(-1.0, min(1.0, obi)),
        atr14=atr,
        rsi14=rsi,
        vwap=vwap,
        trend=trend,
        volatility_regime=vol_regime,
        key_levels=key_levels,
        volume_spike_ratio=round(vol_ratio, 3),
    )

    # Async persist to TimescaleDB (fire and forget)
    try:
        await insert_indicator_snapshot(asset, {
            "rsi14": rsi, "vwap": vwap, "atr14": atr,
            "obi": obi, "bb_upper": bb_upper, "bb_middle": bb_middle,
            "bb_lower": bb_lower, "trend": trend, "volatility_regime": vol_regime,
        })
    except Exception:
        pass

    return ctx


async def market_data_node(state: TradingState) -> dict:
    """LangGraph node: computes MarketContext for the primary asset in event_trigger."""
    t0 = time.monotonic()
    trigger = state.get("event_trigger")
    if trigger is None:
        return {"market_context": None}

    # Determine which asset to analyze
    assets = trigger.affected_assets_guess
    if not assets:
        log.warning("market_data_no_assets", event_id=trigger.event_id)
        return {"market_context": None}

    primary_asset = assets[0]

    ctx = await compute_market_context(primary_asset, trigger)
    latency_ms = int((time.monotonic() - t0) * 1000)

    log.info(
        "market_data_computed",
        asset=primary_asset,
        price=ctx.price if ctx else None,
        rsi=ctx.rsi14 if ctx else None,
        trend=ctx.trend if ctx else None,
        latency_ms=latency_ms,
    )

    # Update session metadata latency
    metadata = state.get("session_metadata")
    if metadata:
        metadata.record_agent_latency("market_data", latency_ms)

    return {"market_context": ctx}
