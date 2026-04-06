from __future__ import annotations

import time

import structlog

from agents.strategy.cts_scorer import compute_cts, cts_to_size_multiplier
from agents.strategy.llm_optimizer import optimize_trade_params, validate_and_build_proposal
from agents.strategy.regime_filters import check_all_filters
from core.config import settings
from core.exceptions import RegimeFilterError
from core.models.trade import Direction
from graph.state import TradingState

log = structlog.get_logger(__name__)


def _determine_direction(sentiment, anomaly, market_ctx) -> Direction | None:
    """
    Determine trade direction from convergent signals.
    Sentiment signal takes priority; anomaly and market confirm.
    """
    if sentiment and not sentiment.conflicted:
        if sentiment.direction == Direction.LONG:
            return Direction.LONG
        elif sentiment.direction == Direction.SHORT:
            return Direction.SHORT

    if anomaly:
        exp = anomaly.expected_move.get("direction", "UNKNOWN")
        if exp == "UP":
            return Direction.LONG
        elif exp == "DOWN":
            return Direction.SHORT

    return None


async def strategy_node(state: TradingState) -> dict:
    """
    LangGraph node: produces TradeProposal from convergent signals.

    Steps:
    1. Regime filters (hard rules, instant)
    2. Determine direction
    3. CTS scoring
    4. If CTS >= threshold → LLM parameter optimization
    5. Build and return TradeProposal
    """
    t0 = time.monotonic()

    trigger = state.get("event_trigger")
    sentiment = state.get("sentiment_signal")
    market_ctx = state.get("market_context")
    anomaly = state.get("anomaly_alert")
    portfolio = state.get("portfolio_state")
    metadata = state.get("session_metadata")

    # Need at least one signal to trade
    if not sentiment and not anomaly:
        log.info("strategy_no_signals")
        return {"trade_proposal": None}

    # Determine target asset
    asset = None
    if sentiment:
        asset = sentiment.primary_asset
    elif market_ctx:
        asset = market_ctx.asset
    elif trigger and trigger.affected_assets_guess:
        asset = trigger.affected_assets_guess[0]

    if not asset:
        return {"trade_proposal": None}

    # Determine direction
    direction = _determine_direction(sentiment, anomaly, market_ctx)
    if direction is None or direction == Direction.NEUTRAL:
        log.info("strategy_neutral_direction", asset=asset)
        return {"trade_proposal": None}

    # Step 1: Regime filters
    try:
        check_all_filters(asset, direction, market_ctx, portfolio)
    except RegimeFilterError as e:
        log.info("strategy_regime_filter_blocked", reason=str(e))
        return {"trade_proposal": None, "error_log": [f"regime_filter: {e}"]}

    # Step 2: CTS scoring
    cts_score, cts_components = compute_cts(sentiment, anomaly, market_ctx, direction)
    size_multiplier = cts_to_size_multiplier(cts_score)

    log.info(
        "strategy_cts_computed",
        asset=asset,
        direction=direction,
        cts=cts_score,
        size_mult=size_multiplier,
        components=cts_components,
    )

    if size_multiplier == 0.0:
        log.info("strategy_cts_too_low", cts=cts_score)
        return {"trade_proposal": None}

    # Step 3: LLM parameter optimization
    available_capital = portfolio.capital_usd * size_multiplier
    try:
        params = await optimize_trade_params(
            asset=asset,
            direction=direction,
            cts_score=cts_score,
            sentiment=sentiment,
            market_ctx=market_ctx,
            anomaly=anomaly,
            available_capital_usd=available_capital,
        )
    except Exception as e:
        log.error("strategy_llm_optimizer_failed", error=str(e))
        return {"trade_proposal": None, "error_log": [f"strategy_llm_error: {str(e)[:100]}"]}

    # Step 4: Validate and build proposal
    try:
        proposal = validate_and_build_proposal(
            trade_id_hint=trigger.event_id if trigger else "manual",
            asset=asset,
            direction=direction,
            cts_score=cts_score,
            params=params,
            available_capital_usd=available_capital,
            event_id=trigger.event_id if trigger else None,
        )
    except (ValueError, Exception) as e:
        log.error("strategy_proposal_validation_failed", error=str(e))
        return {"trade_proposal": None, "error_log": [f"strategy_validation: {str(e)[:100]}"]}

    latency_ms = int((time.monotonic() - t0) * 1000)
    if metadata:
        metadata.record_agent_latency("strategy", latency_ms)

    log.info(
        "strategy_proposal_created",
        asset=proposal.asset,
        direction=proposal.direction,
        entry=proposal.entry_price,
        sl=proposal.stop_loss,
        tp=proposal.take_profit,
        cts=proposal.cts_score,
        size_usd=proposal.position_size_usd,
        latency_ms=latency_ms,
    )

    return {"trade_proposal": proposal}
