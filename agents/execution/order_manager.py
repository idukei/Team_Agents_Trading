from __future__ import annotations

import asyncio
import time

import structlog

from agents.execution.alpaca_client import (
    cancel_order,
    get_order_status,
    place_limit_order,
    place_market_order,
)
from core.models.trade import Direction, ExecutionResult, OrderType, TradeProposal

log = structlog.get_logger(__name__)

TIME_TO_FILL_S = 30     # cancel limit order if not filled in 30s


async def submit_order(proposal: TradeProposal, approved_size_usd: float) -> ExecutionResult | None:
    """
    Submit order to Alpaca. Handle LIMIT with time-to-fill logic.
    Returns ExecutionResult on fill, None on failure.
    """
    side = "buy" if proposal.direction == Direction.LONG else "sell"
    symbol = proposal.asset.replace("/", "")     # BTC/USD → BTCUSD for Alpaca

    # Calculate qty from size
    qty = round(approved_size_usd / proposal.entry_price, 6)
    if qty < 0.0001:
        log.warning("order_qty_too_small", qty=qty, size=approved_size_usd)
        return None

    order_info: dict | None = None

    if proposal.entry_type == OrderType.LIMIT:
        order_info = await place_limit_order(symbol, side, qty, proposal.entry_price)
        if not order_info:
            return None

        # Wait for fill with timeout
        order_info = await _wait_for_fill(order_info["order_id"], timeout_s=TIME_TO_FILL_S)

        if not order_info or order_info.get("filled_qty", 0) == 0:
            # Convert to MARKET order if urgency is high
            log.info("limit_not_filled_converting_to_market", trade_id=proposal.trade_id)
            await cancel_order(order_info["order_id"] if order_info else "")
            order_info = await place_market_order(symbol, side, qty)

    else:
        # MARKET order (immediate)
        order_info = await place_market_order(symbol, side, qty)

    if not order_info:
        return None

    # Wait for market order fill
    if order_info.get("status") != "filled":
        order_info = await _wait_for_fill(order_info["order_id"], timeout_s=10)

    if not order_info or float(order_info.get("filled_qty", 0)) == 0:
        log.error("order_not_filled", trade_id=proposal.trade_id)
        return None

    filled_price = float(order_info.get("filled_avg_price") or proposal.entry_price)
    filled_qty = float(order_info.get("filled_qty", qty))
    slippage_bps = ((filled_price - proposal.entry_price) / proposal.entry_price) * 10_000

    return ExecutionResult(
        trade_id=proposal.trade_id,
        order_id=order_info["order_id"],
        filled_price=filled_price,
        filled_qty=filled_qty,
        slippage_bps=round(slippage_bps, 2),
    )


async def _wait_for_fill(order_id: str, timeout_s: int = 30) -> dict | None:
    """Poll order status until filled or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = await get_order_status(order_id)
        if status and status.get("status") == "filled":
            return status
        await asyncio.sleep(1)
    return await get_order_status(order_id)
