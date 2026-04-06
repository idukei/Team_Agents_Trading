from __future__ import annotations

import structlog

from core.config import settings
from core.models.portfolio import PortfolioState
from core.models.trade import TradeProposal

log = structlog.get_logger(__name__)


def check_trade_layer(
    proposal: TradeProposal,
    portfolio: PortfolioState,
) -> tuple[bool, str | None, float]:
    """
    Layer 1: Per-trade risk checks.

    Returns (passed, rejection_reason, approved_size_usd).
    """
    capital = portfolio.capital_usd
    max_risk_usd = capital * settings.risk_max_trade_pct

    # Check 1: Risk amount
    risk_usd = proposal.risk_usd
    if risk_usd > max_risk_usd:
        # Try to reduce size to fit within risk limit
        reduction_factor = max_risk_usd / risk_usd
        adjusted_size = proposal.position_size_usd * reduction_factor
        if adjusted_size < 100:     # minimum trade size $100
            return False, f"Risk ${risk_usd:.2f} exceeds max ${max_risk_usd:.2f}, size too small after adjustment", 0.0
        log.info("trade_layer_size_adjusted", original=proposal.position_size_usd, adjusted=adjusted_size)
        approved_size = adjusted_size
    else:
        approved_size = proposal.position_size_usd

    # Check 2: R/R ratio
    if proposal.primary_rr < settings.risk_min_rr:
        return False, f"R/R ratio {proposal.primary_rr:.2f} below minimum {settings.risk_min_rr}", 0.0

    # Check 3: Slippage estimate (spread × size should not exceed 15% of TP)
    tp_gain = abs(proposal.take_profit[0] - proposal.entry_price) * (approved_size / proposal.entry_price)
    if tp_gain > 0:
        # Rough slippage estimate (assume 2bps round trip)
        slippage_est = approved_size * 0.0002
        slippage_pct_of_tp = slippage_est / tp_gain
        if slippage_pct_of_tp > settings.risk_slippage_max_pct:
            return False, f"Estimated slippage {slippage_pct_of_tp:.1%} exceeds {settings.risk_slippage_max_pct:.1%} of TP", 0.0

    return True, None, round(approved_size, 2)
