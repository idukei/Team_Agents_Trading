from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog

from agents.sentiment.rag import store_precedent
from core.db.postgres import get_pool
from core.models.portfolio import PortfolioState

log = structlog.get_logger(__name__)


async def run_daily_analysis(portfolio: PortfolioState) -> None:
    """
    Runs at 22:00 UTC. Analyzes today's trades, updates RAG precedents,
    and logs performance metrics.
    """
    log.info("post_trade_analysis_starting", trades_today=portfolio.trades_today)
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Get today's trades from DB
        today = datetime.now(timezone.utc).date()
        rows = await conn.fetch(
            """
            SELECT tp.trade_id, tp.asset, tp.direction, tp.reasoning_summary,
                   er.pnl_usd, er.filled_price, er.exit_price,
                   et.raw_content as event_text
            FROM execution_results er
            JOIN trade_proposals tp ON er.trade_id = tp.trade_id
            LEFT JOIN event_triggers et ON tp.event_id = et.event_id
            WHERE er.created_at::date = $1 AND er.pnl_usd IS NOT NULL
            """,
            today,
        )

        # Record daily performance
        await conn.execute(
            """
            INSERT INTO daily_performance
            (date, starting_capital, ending_capital, pnl_usd, pnl_pct,
             total_trades, winning_trades, losing_trades)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (date) DO UPDATE SET
                ending_capital=EXCLUDED.ending_capital,
                pnl_usd=EXCLUDED.pnl_usd, pnl_pct=EXCLUDED.pnl_pct,
                total_trades=EXCLUDED.total_trades,
                winning_trades=EXCLUDED.winning_trades,
                losing_trades=EXCLUDED.losing_trades
            """,
            today,
            portfolio.capital_usd - portfolio.daily_pnl_usd,
            portfolio.capital_usd,
            portfolio.daily_pnl_usd,
            portfolio.daily_pnl_pct,
            portfolio.trades_today,
            portfolio.winning_trades_today,
            portfolio.trades_today - portfolio.winning_trades_today,
        )

    # Store successful trades as new RAG precedents
    for row in rows:
        if row["event_text"] and row["pnl_usd"] is not None:
            pnl = float(row["pnl_usd"])
            entry = float(row["filled_price"]) if row["filled_price"] else 1.0
            return_pct = (pnl / (entry * 100)) * 100   # rough % return
            try:
                await store_precedent(
                    event_id=f"trade_{row['trade_id']}",
                    event_text=str(row["event_text"])[:500],
                    asset=row["asset"],
                    direction=row["direction"],
                    return_15min=return_pct,
                    return_60min=return_pct,
                    outcome_summary=f"Trade closed with P&L ${pnl:.2f} ({return_pct:.2f}%)",
                )
            except Exception as e:
                log.warning("rag_precedent_store_failed", trade_id=row["trade_id"], error=str(e))

    log.info(
        "post_trade_analysis_complete",
        trades=portfolio.trades_today,
        pnl_usd=portfolio.daily_pnl_usd,
        pnl_pct=portfolio.daily_pnl_pct,
    )


async def schedule_daily_analysis(portfolio: PortfolioState) -> None:
    """Wait until 22:00 UTC, then run analysis."""
    while True:
        now = datetime.now(timezone.utc)
        target = now.replace(hour=22, minute=0, second=0, microsecond=0)
        if now >= target:
            from datetime import timedelta
            target += timedelta(days=1)
        wait_s = (target - now).total_seconds()
        log.info("post_trade_analysis_scheduled", in_hours=round(wait_s / 3600, 1))
        await asyncio.sleep(wait_s)
        await run_daily_analysis(portfolio)
