"""
TeamTrade — Sistema Multi-Agente IA para Trading Event-Driven
Entrypoint principal.

Arranca:
1. Conexiones a bases de datos
2. Dashboard web (FastAPI + WebSocket) en http://localhost:8000
3. Telegram Bot
4. Monitor Agent (4 sub-monitores paralelos)
5. LangGraph pipeline consumer (lee Redis Stream → activa graph)
6. Post-trade analyst scheduler
7. Portfolio heartbeat → dashboard WebSocket

Uso:
    uv run python main.py                  # sistema completo
    uv run python main.py --monitor-only   # solo monitoreo + Telegram (Fase 1)
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys

import structlog

from core.config import settings
from core.db.postgres import close_pool, get_pool
from core.db.qdrant import close_client as close_qdrant, ensure_collections, ping_qdrant
from core.db.redis import close_client as close_redis, consume_events, ping_redis
from core.logging import configure_logging
from core.models.events import EventTrigger
from core.models.portfolio import PortfolioState, SessionMetadata

log = structlog.get_logger(__name__)


async def startup_checks() -> None:
    """Verify all infrastructure is reachable."""
    log.info("startup_checks_starting")

    pg_ok = False
    try:
        await get_pool()
        pg_ok = True
        log.info("postgres_ok")
    except Exception as e:
        log.error("postgres_failed", error=str(e))

    redis_ok = await ping_redis()
    log.info("redis_ok" if redis_ok else "redis_failed")

    qdrant_ok = await ping_qdrant()
    if qdrant_ok:
        await ensure_collections()
        log.info("qdrant_ok")
    else:
        log.warning("qdrant_not_available_rag_disabled")

    if not pg_ok or not redis_ok:
        log.error("critical_services_unavailable")
        sys.exit(1)


async def run_graph_consumer(graph, portfolio: PortfolioState) -> None:
    """
    Consume EventTrigger events from Redis Stream and run LangGraph pipeline.
    Each event gets its own thread_id for independent checkpoint history.
    """
    from graph.checkpointer import make_thread_id
    from graph.state import initial_state

    log.info("graph_consumer_started", stream=settings.redis_stream_events)

    async for entry_id, fields in consume_events(settings.redis_stream_events, last_id="$"):
        try:
            trigger = EventTrigger.from_redis_dict(fields)
            log.info("graph_consumer_received", eps=trigger.eps_score, source=trigger.event_source)

            # Create fresh session metadata for this run
            from datetime import datetime, timezone
            metadata = SessionMetadata(
                started_at_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
                pipeline_mode="FULL" if trigger.eps_score >= settings.eps_full_pipeline else "LIGHT",
            )

            state = initial_state(portfolio, metadata)
            state["event_trigger"] = trigger

            thread_id = make_thread_id(trigger.event_id)
            config = {"configurable": {"thread_id": thread_id}}

            # Run the graph (auto-mode: no interrupt wait)
            async for chunk in graph.astream(state, config=config):
                log.debug("graph_chunk", node=list(chunk.keys()))

        except Exception as e:
            log.error("graph_consumer_error", error=str(e), entry_id=entry_id)


async def main(monitor_only: bool = False) -> None:
    configure_logging(settings.log_level, settings.log_format)
    log.info("teamtrade_starting", mode="monitor_only" if monitor_only else "full")

    await startup_checks()

    # ── Setup Telegram ─────────────────────────────────────────────────────────
    from agents.notification.telegram_bot import build_application, set_portfolio_ref
    telegram_app, approval_handler = build_application()

    # ── Portfolio state (shared across all agents) ─────────────────────────────
    try:
        account = await __import__(
            "agents.execution.alpaca_client",
            fromlist=["get_account"]
        ).get_account()
        capital = account["equity"]
    except Exception:
        capital = settings.capital_usd
        log.warning("alpaca_account_fetch_failed_using_config", capital=capital)

    portfolio = PortfolioState(capital_usd=capital)
    set_portfolio_ref(portfolio)

    # ── Setup agent instances ──────────────────────────────────────────────────
    from agents.notification.agent import NotificationAgent
    from agents.supervisor.agent import SupervisorAgent
    notification_agent = NotificationAgent(telegram_app, approval_handler)
    supervisor_agent = SupervisorAgent()

    from graph.nodes import set_notification_agent, set_supervisor_agent
    set_notification_agent(notification_agent)
    set_supervisor_agent(supervisor_agent)

    # ── Dashboard web server ───────────────────────────────────────────────────
    import uvicorn
    from api.server import app as dashboard_app

    dashboard_config = uvicorn.Config(
        dashboard_app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="warning",
        access_log=False,
    )
    dashboard_server = uvicorn.Server(dashboard_config)

    # ── Portfolio heartbeat → dashboard ────────────────────────────────────────
    async def portfolio_heartbeat(portfolio: PortfolioState) -> None:
        from api.state_broadcaster import broadcaster
        while True:
            try:
                await broadcaster.broadcast_portfolio(portfolio)
            except Exception:
                pass
            await asyncio.sleep(2)

    # ── Monitor Agent ──────────────────────────────────────────────────────────
    from agents.monitor.agent import MonitorAgent
    monitor = MonitorAgent()

    tasks = [
        asyncio.create_task(dashboard_server.serve(), name="Dashboard"),
        asyncio.create_task(portfolio_heartbeat(portfolio), name="PortfolioHeartbeat"),
        asyncio.create_task(monitor.start(), name="MonitorAgent"),
        asyncio.create_task(telegram_app.initialize()),
        asyncio.create_task(telegram_app.start()),
        asyncio.create_task(telegram_app.updater.start_polling()),
    ]

    if not monitor_only:
        # ── Build LangGraph ────────────────────────────────────────────────────
        from graph.builder import build_trading_graph
        graph = await build_trading_graph()
        log.info("langgraph_ready")

        # ── Post-trade analyst ─────────────────────────────────────────────────
        from agents.supervisor.post_trade_analyst import schedule_daily_analysis
        tasks.append(asyncio.create_task(run_graph_consumer(graph, portfolio), name="GraphConsumer"))
        tasks.append(asyncio.create_task(schedule_daily_analysis(portfolio), name="PostTradeAnalyst"))

    # Startup notification
    await notification_agent.send(
        f"*TeamTrade iniciado*\n"
        f"Modo: `{'MONITOR' if monitor_only else 'FULL'}`\n"
        f"Capital: `${portfolio.capital_usd:,.2f}`\n"
        f"Activos monitoreados: `{len(settings.all_assets)}`\n"
        f"Dashboard: `http://localhost:{settings.dashboard_port}`"
    )

    log.info("teamtrade_all_systems_running", dashboard=f"http://{settings.dashboard_host}:{settings.dashboard_port}")

    # ── Graceful shutdown ──────────────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _on_signal():
        log.info("shutdown_signal_received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    await stop_event.wait()

    log.info("teamtrade_shutting_down")
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    await telegram_app.stop()
    await telegram_app.shutdown()
    await close_redis()
    await close_qdrant()
    await close_pool()

    log.info("teamtrade_shutdown_complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor-only", action="store_true", help="Only run monitor + Telegram (no trading)")
    args = parser.parse_args()
    asyncio.run(main(monitor_only=args.monitor_only))
