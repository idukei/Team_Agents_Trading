from __future__ import annotations

from datetime import datetime, timezone

import structlog

from core.config import settings
from core.exceptions import RegimeFilterError
from core.models.portfolio import PortfolioState
from core.models.signals import MarketContext
from core.models.trade import Direction

log = structlog.get_logger(__name__)


def check_all_filters(
    asset: str,
    direction: Direction,
    market_ctx: MarketContext | None,
    portfolio: PortfolioState,
    pending_calendar_events_within_s: int = 0,
) -> None:
    """
    Run all hard regime filters. Raises RegimeFilterError on any failure.
    These are non-negotiable — no LLM can override them.
    """
    # Filter 1: Circuit breaker active
    if portfolio.circuit_breaker_active:
        raise RegimeFilterError(f"Circuit breaker active: {portfolio.circuit_breaker_reason}")

    # Filter 2: Daily loss limit reached
    if portfolio.daily_pnl_pct <= -settings.risk_circuit_breaker_pct:
        raise RegimeFilterError(
            f"Daily loss limit reached: {portfolio.daily_pnl_pct:.2%} <= -{settings.risk_circuit_breaker_pct:.2%}"
        )

    # Filter 3: Max open trades reached
    if len(portfolio.open_positions) >= settings.risk_max_open_trades:
        raise RegimeFilterError(
            f"Max open trades ({settings.risk_max_open_trades}) already active"
        )

    # Filter 4: Opposite direction trade already open for same asset
    for pos in portfolio.open_positions:
        if pos.asset == asset and pos.direction != direction:
            raise RegimeFilterError(
                f"Opposite direction position already open for {asset}: {pos.direction}"
            )

    # Filter 5: Spread too wide (if market context available)
    if market_ctx and market_ctx.spread_bps > 3 * 5.0:   # 3x normal ~5bps
        raise RegimeFilterError(
            f"Spread too wide: {market_ctx.spread_bps:.1f}bps (normal ~5bps)"
        )

    # Filter 6: High-impact economic event imminent
    if pending_calendar_events_within_s > 0 and pending_calendar_events_within_s <= settings.risk_pre_event_window_s:
        raise RegimeFilterError(
            f"High-impact event in {pending_calendar_events_within_s}s — waiting"
        )

    log.debug("regime_filters_passed", asset=asset, direction=direction)
