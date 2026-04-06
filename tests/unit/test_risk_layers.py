import pytest
from agents.risk.portfolio_layer import check_portfolio_layer
from agents.risk.systemic_layer import check_systemic_layer
from agents.risk.trade_layer import check_trade_layer
from core.models.portfolio import PortfolioState, Position
from core.models.signals import MarketContext
from core.models.trade import Direction, OrderType, StrategyType, TradeProposal


def make_proposal(entry=100.0, sl=97.0, tp1=106.0, tp2=112.0, size=500.0):
    return TradeProposal(
        asset="NVDA",
        direction=Direction.LONG,
        entry_type=OrderType.LIMIT,
        entry_price=entry,
        position_size_usd=size,
        stop_loss=sl,
        take_profit=[tp1, tp2],
        max_hold_seconds=600,
        cts_score=0.75,
        strategy_type=StrategyType.POLITICAL_SCALP,
    )


def make_portfolio(capital=10000.0, daily_pnl_pct=0.0):
    return PortfolioState(capital_usd=capital, daily_pnl_pct=daily_pnl_pct)


# ── Layer 1 tests ─────────────────────────────────────────────────────────────

def test_layer1_passes_valid_trade():
    proposal = make_proposal()
    portfolio = make_portfolio()
    passed, reason, size = check_trade_layer(proposal, portfolio)
    assert passed, reason
    assert size > 0


def test_layer1_rejects_bad_rr():
    # TP very close to entry → bad R/R
    proposal = make_proposal(entry=100.0, sl=97.0, tp1=101.0, tp2=102.0)
    portfolio = make_portfolio()
    passed, reason, _ = check_trade_layer(proposal, portfolio)
    assert not passed
    assert "R/R" in (reason or "")


def test_layer1_reduces_oversized_trade():
    # Trade risk > 1% capital → should be reduced
    proposal = make_proposal(entry=100.0, sl=95.0, tp1=110.0, tp2=120.0, size=5000.0)
    portfolio = make_portfolio(capital=10000.0)
    passed, _, approved_size = check_trade_layer(proposal, portfolio)
    assert passed
    assert approved_size < 5000.0


# ── Layer 3 tests ─────────────────────────────────────────────────────────────

def test_layer3_passes_normal():
    portfolio = make_portfolio()
    passed, reason, mult = check_systemic_layer(portfolio, None)
    assert passed
    assert mult == 1.0


def test_layer3_blocks_circuit_breaker():
    portfolio = make_portfolio(daily_pnl_pct=-0.03)   # -3% > -2.5% threshold
    passed, reason, _ = check_systemic_layer(portfolio, None)
    assert not passed


def test_layer3_reduces_size_pre_event():
    portfolio = make_portfolio()
    passed, _, mult = check_systemic_layer(portfolio, None, pending_event_s=120)
    assert passed
    assert mult < 1.0   # reduced near event
