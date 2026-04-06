from __future__ import annotations

import numpy as np
import structlog

from core.config import settings
from core.models.portfolio import PortfolioState
from core.models.trade import TradeProposal

log = structlog.get_logger(__name__)


def check_portfolio_layer(
    proposal: TradeProposal,
    portfolio: PortfolioState,
    approved_size_usd: float,
) -> tuple[bool, str | None, float, dict]:
    """
    Layer 2: Portfolio-level risk checks.

    Returns (passed, rejection_reason, final_size_usd, risk_metrics).
    """
    capital = portfolio.capital_usd

    # Check 1: Total portfolio risk exposure
    current_risk = portfolio.total_risk_usd
    new_risk = approved_size_usd * settings.risk_max_trade_pct
    total_risk_pct = (current_risk + new_risk) / capital

    if total_risk_pct > 0.05:   # max 5% total capital at risk
        return False, f"Total portfolio risk would be {total_risk_pct:.1%} > 5%", 0.0, {}

    # Check 2: Correlation check (simplified — check if same sector/asset)
    correlation = _estimate_correlation(proposal.asset, portfolio)

    if correlation > settings.risk_max_portfolio_correlation:
        # Reduce size by 50% for highly correlated trades
        adjusted_size = approved_size_usd * 0.5
        log.info(
            "portfolio_layer_correlation_reduction",
            correlation=correlation,
            original=approved_size_usd,
            adjusted=adjusted_size,
        )
        approved_size_usd = adjusted_size

    # Check 3: Parametric VaR (95%, 1-day) — simplified calculation
    var_95 = _compute_var_95(approved_size_usd, proposal.asset)

    metrics = {
        "var_95": var_95,
        "portfolio_correlation": correlation,
        "total_risk_pct": total_risk_pct,
    }

    return True, None, round(approved_size_usd, 2), metrics


def _estimate_correlation(new_asset: str, portfolio: PortfolioState) -> float:
    """
    Estimate correlation between new trade and existing portfolio.
    Simplified: uses asset class matching as proxy.
    """
    if not portfolio.open_positions:
        return 0.0

    # Asset class buckets
    tech_assets = {"NVDA", "AAPL", "QQQ", "SMH", "AMD", "MSFT"}
    index_assets = {"SPY", "QQQ", "IWM", "DIA"}
    crypto_assets = {"BTC/USD", "ETH/USD"}
    bond_assets = {"TLT", "IEF", "SHY"}

    def get_class(asset: str) -> str:
        if asset in tech_assets:
            return "tech"
        if asset in index_assets:
            return "index"
        if asset in crypto_assets:
            return "crypto"
        if asset in bond_assets:
            return "bond"
        return "other"

    new_class = get_class(new_asset)
    existing_classes = [get_class(p.asset) for p in portfolio.open_positions]

    if not existing_classes:
        return 0.0

    same_class_count = sum(1 for c in existing_classes if c == new_class)
    correlation = same_class_count / len(existing_classes)
    return round(correlation, 3)


def _compute_var_95(size_usd: float, asset: str, daily_vol: float = 0.02) -> float:
    """
    Simplified parametric VaR (95%, 1-day).
    daily_vol = assumed 2% daily volatility (conservative default).
    """
    # VaR = size × daily_vol × 1.645 (z-score for 95%)
    return round(size_usd * daily_vol * 1.645, 2)
