from __future__ import annotations

import asyncio
import time

import structlog

from agents.execution.order_manager import submit_order
from agents.execution.position_manager import manage_position
from agents.market_data.stream import buffer_registry
from core.models.trade import RiskDecisionStatus
from graph.state import TradingState

log = structlog.get_logger(__name__)


async def _get_price(asset: str) -> float | None:
    buf = buffer_registry.get(asset)
    return buf.latest_price if buf else None


async def execution_node(state: TradingState) -> dict:
    """
    LangGraph node: executes approved trade proposal via Alpaca.

    This node runs AFTER the human-in-the-loop interrupt point.
    If human rejected (graph resumed with awaiting_human_approval=False + risk_decision=REJECTED),
    this node is skipped by the conditional edge.
    """
    t0 = time.monotonic()

    proposal = state.get("trade_proposal")
    risk_decision = state.get("risk_decision")
    metadata = state.get("session_metadata")
    portfolio = state.get("portfolio_state")

    if proposal is None or risk_decision is None:
        return {"execution_result": None}

    if risk_decision.status == RiskDecisionStatus.REJECTED:
        log.info("execution_skipped_risk_rejected", trade_id=proposal.trade_id)
        return {"execution_result": None}

    approved_size = risk_decision.approved_size_usd
    log.info(
        "execution_starting",
        trade_id=proposal.trade_id,
        asset=proposal.asset,
        direction=proposal.direction,
        size_usd=approved_size,
    )

    # Submit the initial order
    result = await submit_order(proposal, approved_size)
    if result is None:
        log.error("execution_order_failed", trade_id=proposal.trade_id)
        return {
            "execution_result": None,
            "error_log": [f"execution_failed: order not filled for {proposal.trade_id}"],
        }

    log.info(
        "order_filled",
        trade_id=proposal.trade_id,
        filled_price=result.filled_price,
        slippage_bps=result.slippage_bps,
    )

    # Manage position lifecycle asynchronously
    # (TP1/TP2/SL monitoring runs in background — doesn't block graph)
    asyncio.create_task(
        _manage_and_update(proposal, result, portfolio),
        name=f"position_mgr_{proposal.trade_id}",
    )

    latency_ms = int((time.monotonic() - t0) * 1000)
    if metadata:
        metadata.record_agent_latency("execution", latency_ms)

    return {"execution_result": result}


async def _manage_and_update(proposal, initial_result, portfolio) -> None:
    """
    Background task: monitors position until exit, then updates portfolio.
    This runs outside the LangGraph state machine (post-execution).
    """
    try:
        final_result = await manage_position(proposal, initial_result, _get_price)

        # Update portfolio P&L
        if final_result.pnl_usd is not None:
            portfolio.daily_pnl_usd += final_result.pnl_usd
            portfolio.daily_pnl_pct = portfolio.daily_pnl_usd / portfolio.capital_usd
            portfolio.trades_today += 1
            if final_result.pnl_usd > 0:
                portfolio.winning_trades_today += 1

            # Remove position from open list
            portfolio.open_positions = [
                p for p in portfolio.open_positions
                if p.trade_id != proposal.trade_id
            ]

        log.info(
            "position_closed",
            trade_id=proposal.trade_id,
            pnl_usd=final_result.pnl_usd,
            exit_reason="TP" if final_result.tp2_hit else "SL" if final_result.sl_hit else "TIMEOUT",
        )

    except Exception as e:
        log.error("position_manager_error", trade_id=proposal.trade_id, error=str(e))
