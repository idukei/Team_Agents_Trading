from __future__ import annotations

import structlog

from core.config import settings
from core.models.portfolio import PortfolioState
from core.models.signals import MarketContext

log = structlog.get_logger(__name__)


def check_systemic_layer(
    portfolio: PortfolioState,
    market_ctx: MarketContext | None,
    pending_event_s: int = 0,
) -> tuple[bool, str | None, float]:
    """
    Layer 3: Systemic / market regime checks.

    Returns (passed, reason, size_multiplier).
    A multiplier < 1.0 means reduce size but don't reject.
    """
    multiplier = 1.0

    # Check 1: Daily loss circuit breaker
    if portfolio.daily_pnl_pct <= -settings.risk_circuit_breaker_pct:
        return False, f"Circuit breaker: daily loss {portfolio.daily_pnl_pct:.2%}", 0.0

    # Check 2: Flash crash detection — extreme volatility
    if market_ctx and market_ctx.volatility_regime == "EXTREME":
        return False, "Flash crash / extreme volatility detected ��� no new entries", 0.0

    # Check 3: Pre-event window protection (±3 min of high-impact events)
    if 0 < pending_event_s <= settings.risk_pre_event_window_s:
        multiplier = 0.4    # reduce size to 40% near events
        log.info("systemic_pre_event_size_reduction", event_in_s=pending_event_s, multiplier=multiplier)

    # Check 4: High spread environment
    if market_ctx and market_ctx.spread_bps > 10:
        multiplier = min(multiplier, 0.6)
        log.info("systemic_high_spread_reduction", spread_bps=market_ctx.spread_bps)

    return True, None, multiplier
