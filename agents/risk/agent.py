from __future__ import annotations

import time

import structlog

from agents.risk.portfolio_layer import check_portfolio_layer
from agents.risk.systemic_layer import check_systemic_layer
from agents.risk.trade_layer import check_trade_layer
from core.models.trade import RiskDecision, RiskDecisionStatus
from graph.state import TradingState

log = structlog.get_logger(__name__)


async def risk_node(state: TradingState) -> dict:
    """
    LangGraph node: 3-layer risk validation.
    All deterministic, < 5ms.

    Returns updated state with risk_decision.
    """
    t0 = time.monotonic()

    proposal = state.get("trade_proposal")
    if proposal is None:
        return {"risk_decision": None}

    portfolio = state.get("portfolio_state")
    market_ctx = state.get("market_context")
    metadata = state.get("session_metadata")

    layer_results: dict[str, bool] = {}
    adjustments: dict = {}

    # ── Layer 1: Per-trade ────────────────────────────────────────────────────
    l1_pass, l1_reason, approved_size = check_trade_layer(proposal, portfolio)
    layer_results["layer1"] = l1_pass

    if not l1_pass:
        decision = RiskDecision(
            trade_id=proposal.trade_id,
            status=RiskDecisionStatus.REJECTED,
            approved_size_usd=0.0,
            rejection_reason=f"Layer1: {l1_reason}",
            layer_results=layer_results,
        )
        log.info("risk_rejected_layer1", reason=l1_reason)
        return {"risk_decision": decision}

    # ── Layer 2: Portfolio ────────────────────────────────────────────────────
    l2_pass, l2_reason, approved_size, risk_metrics = check_portfolio_layer(
        proposal, portfolio, approved_size
    )
    layer_results["layer2"] = l2_pass

    if not l2_pass:
        decision = RiskDecision(
            trade_id=proposal.trade_id,
            status=RiskDecisionStatus.REJECTED,
            approved_size_usd=0.0,
            rejection_reason=f"Layer2: {l2_reason}",
            layer_results=layer_results,
        )
        log.info("risk_rejected_layer2", reason=l2_reason)
        return {"risk_decision": decision}

    # ── Layer 3: Systemic ─────────────────────────────────────────────────────
    l3_pass, l3_reason, size_multiplier = check_systemic_layer(portfolio, market_ctx)
    layer_results["layer3"] = l3_pass

    if not l3_pass:
        decision = RiskDecision(
            trade_id=proposal.trade_id,
            status=RiskDecisionStatus.REJECTED,
            approved_size_usd=0.0,
            rejection_reason=f"Layer3: {l3_reason}",
            layer_results=layer_results,
        )
        log.info("risk_rejected_layer3", reason=l3_reason)
        return {"risk_decision": decision}

    # Apply systemic size multiplier
    final_size = round(approved_size * size_multiplier, 2)
    if size_multiplier < 1.0:
        adjustments["systemic_size_multiplier"] = size_multiplier

    status = (
        RiskDecisionStatus.APPROVED_REDUCED
        if final_size < proposal.position_size_usd * 0.95
        else RiskDecisionStatus.APPROVED
    )

    decision = RiskDecision(
        trade_id=proposal.trade_id,
        status=status,
        approved_size_usd=final_size,
        adjustments=adjustments,
        var_95=risk_metrics.get("var_95"),
        portfolio_correlation=risk_metrics.get("portfolio_correlation"),
        layer_results=layer_results,
    )

    latency_ms = int((time.monotonic() - t0) * 1000)
    if metadata:
        metadata.record_agent_latency("risk", latency_ms)

    log.info(
        "risk_approved",
        trade_id=proposal.trade_id,
        status=status,
        original_size=proposal.position_size_usd,
        approved_size=final_size,
        latency_ms=latency_ms,
    )

    return {"risk_decision": decision}
