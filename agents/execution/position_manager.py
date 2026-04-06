from __future__ import annotations

import asyncio
import time

import structlog

from agents.execution.alpaca_client import place_market_order
from core.models.portfolio import Position
from core.models.trade import Direction, ExecutionResult, TradeProposal

log = structlog.get_logger(__name__)


async def manage_position(
    proposal: TradeProposal,
    result: ExecutionResult,
    get_current_price_fn,
) -> ExecutionResult:
    """
    Monitor open position and manage exits:
    - TP1: close 50%, move SL to breakeven
    - TP2: close remaining 100%
    - SL: close full position
    - max_hold_seconds: timeout close

    get_current_price_fn: async callable(asset) -> float | None
    """
    symbol = proposal.asset.replace("/", "")
    deadline = time.monotonic() + proposal.max_hold_seconds
    tp1_hit = False
    total_qty = result.filled_qty
    remaining_qty = total_qty
    entry_price = result.filled_price
    sl = proposal.stop_loss
    tp_levels = proposal.take_profit

    log.info(
        "position_monitoring_started",
        asset=proposal.asset,
        direction=proposal.direction,
        entry=entry_price,
        sl=sl,
        tps=tp_levels,
        max_hold_s=proposal.max_hold_seconds,
    )

    while time.monotonic() < deadline and remaining_qty > 0:
        await asyncio.sleep(1)

        current_price = await get_current_price_fn(proposal.asset)
        if current_price is None:
            continue

        if proposal.direction == Direction.LONG:
            # SL hit
            if current_price <= sl:
                await _close_position(symbol, "sell", remaining_qty, "SL")
                result = result.model_copy(update={
                    "sl_hit": True,
                    "exit_price": current_price,
                    "exit_timestamp_ms": int(time.time() * 1000),
                    "pnl_usd": _calc_pnl(entry_price, current_price, total_qty, Direction.LONG, tp1_hit),
                })
                return result

            # TP1
            if not tp1_hit and len(tp_levels) >= 1 and current_price >= tp_levels[0]:
                half_qty = remaining_qty / 2
                await _close_position(symbol, "sell", half_qty, "TP1")
                remaining_qty -= half_qty
                sl = entry_price   # move SL to breakeven
                tp1_hit = True
                result = result.model_copy(update={"tp1_hit": True})
                log.info("tp1_hit_sl_to_breakeven", asset=proposal.asset, sl=sl)

            # TP2
            if tp1_hit and len(tp_levels) >= 2 and current_price >= tp_levels[1]:
                await _close_position(symbol, "sell", remaining_qty, "TP2")
                result = result.model_copy(update={
                    "tp2_hit": True,
                    "exit_price": current_price,
                    "exit_timestamp_ms": int(time.time() * 1000),
                    "pnl_usd": _calc_pnl(entry_price, current_price, total_qty, Direction.LONG, tp1_hit),
                })
                return result

        else:   # SHORT
            if current_price >= sl:
                await _close_position(symbol, "buy", remaining_qty, "SL")
                result = result.model_copy(update={
                    "sl_hit": True,
                    "exit_price": current_price,
                    "exit_timestamp_ms": int(time.time() * 1000),
                    "pnl_usd": _calc_pnl(entry_price, current_price, total_qty, Direction.SHORT, tp1_hit),
                })
                return result

            if not tp1_hit and len(tp_levels) >= 1 and current_price <= tp_levels[0]:
                half_qty = remaining_qty / 2
                await _close_position(symbol, "buy", half_qty, "TP1")
                remaining_qty -= half_qty
                sl = entry_price
                tp1_hit = True
                result = result.model_copy(update={"tp1_hit": True})

            if tp1_hit and len(tp_levels) >= 2 and current_price <= tp_levels[1]:
                await _close_position(symbol, "buy", remaining_qty, "TP2")
                result = result.model_copy(update={
                    "tp2_hit": True,
                    "exit_price": current_price,
                    "exit_timestamp_ms": int(time.time() * 1000),
                    "pnl_usd": _calc_pnl(entry_price, current_price, total_qty, Direction.SHORT, tp1_hit),
                })
                return result

    # Timeout exit
    if remaining_qty > 0:
        current_price = await get_current_price_fn(proposal.asset) or entry_price
        close_side = "sell" if proposal.direction == Direction.LONG else "buy"
        await _close_position(symbol, close_side, remaining_qty, "TIMEOUT")
        result = result.model_copy(update={
            "timeout_exit": True,
            "exit_price": current_price,
            "exit_timestamp_ms": int(time.time() * 1000),
            "pnl_usd": _calc_pnl(entry_price, current_price, total_qty, proposal.direction, tp1_hit),
        })

    return result


async def _close_position(symbol: str, side: str, qty: float, reason: str) -> None:
    log.info("closing_position", symbol=symbol, side=side, qty=qty, reason=reason)
    try:
        await place_market_order(symbol, side, qty)
    except Exception as e:
        log.error("close_position_failed", symbol=symbol, error=str(e))


def _calc_pnl(
    entry: float,
    exit_price: float,
    qty: float,
    direction: Direction,
    tp1_was_hit: bool,
) -> float:
    effective_qty = qty * 0.5 if tp1_was_hit else qty
    if direction == Direction.LONG:
        return round((exit_price - entry) * effective_qty, 4)
    return round((entry - exit_price) * effective_qty, 4)
