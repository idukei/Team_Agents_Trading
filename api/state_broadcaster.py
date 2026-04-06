"""
StateBroadcaster — WebSocket manager for real-time dashboard updates.

Singleton that maintains connected WebSocket clients and broadcasts
TradingState snapshots whenever a LangGraph node completes.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import WebSocket

log = structlog.get_logger(__name__)


class StateBroadcaster:
    """Push state updates to all connected dashboard WebSocket clients."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._latest_state: dict[str, Any] = {
            "type": "init",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        log.debug("ws_client_connected", total=len(self._clients))
        # Send current state snapshot to new client immediately
        try:
            await ws.send_json(self._latest_state)
        except Exception:
            pass

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        log.debug("ws_client_disconnected", total=len(self._clients))

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send JSON payload to all connected clients. Dead clients are removed."""
        self._latest_state = data
        if not self._clients:
            return

        dead: set[WebSocket] = set()
        async with self._lock:
            clients = set(self._clients)

        for ws in clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)

        if dead:
            async with self._lock:
                self._clients -= dead
            log.debug("ws_clients_removed", count=len(dead))

    async def broadcast_portfolio(self, portfolio_state: Any) -> None:
        """Broadcast a portfolio-only update (called by periodic heartbeat)."""
        from core.models.portfolio import PortfolioState
        if not isinstance(portfolio_state, PortfolioState):
            return

        positions = [
            {
                "trade_id": str(p.trade_id),
                "asset": p.asset,
                "direction": p.direction.value,
                "entry_price": p.entry_price,
                "current_size_usd": p.current_size_usd,
                "unrealized_pnl": p.unrealized_pnl,
                "tp1_hit": p.tp1_hit,
            }
            for p in portfolio_state.open_positions
        ]

        await self.broadcast({
            "type": "portfolio_heartbeat",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "capital_usd": portfolio_state.capital_usd,
            "daily_pnl_usd": portfolio_state.daily_pnl_usd,
            "daily_pnl_pct": portfolio_state.daily_pnl_pct,
            "total_exposure_pct": portfolio_state.total_exposure_pct,
            "open_positions": positions,
            "circuit_breaker_active": portfolio_state.circuit_breaker_active,
            "circuit_breaker_reason": portfolio_state.circuit_breaker_reason,
            "trades_today": portfolio_state.trades_today,
            "win_rate_today": portfolio_state.win_rate_today,
        })

    def serialize_state(self, state: dict[str, Any], node_name: str) -> dict[str, Any]:
        """Convert a TradingState dict to a JSON-serializable dashboard payload."""
        payload: dict[str, Any] = {
            "type": "pipeline_update",
            "node": node_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Event trigger
        trigger = state.get("event_trigger")
        if trigger:
            payload["event"] = {
                "event_id": str(trigger.event_id),
                "source": trigger.event_source.value,
                "eps_score": trigger.eps_score,
                "urgency": trigger.urgency.value,
                "raw_content": trigger.raw_content[:200],
                "affected_assets": trigger.affected_assets_guess,
                "age_seconds": trigger.age_seconds,
            }

        # Sentiment signal
        sig = state.get("sentiment_signal")
        if sig:
            payload["sentiment"] = {
                "direction": sig.direction.value,
                "magnitude": sig.magnitude,
                "confidence": sig.confidence,
                "time_horizon": sig.time_horizon.value,
                "reasoning": sig.llm_reasoning[:300] if sig.llm_reasoning else "",
                "conflicted": sig.conflicted,
            }

        # Market context
        ctx = state.get("market_context")
        if ctx:
            payload["market"] = {
                "asset": ctx.asset,
                "price": ctx.price,
                "rsi14": ctx.rsi14,
                "atr14": ctx.atr14,
                "trend": ctx.trend.value if ctx.trend else None,
                "volatility_regime": ctx.volatility_regime.value if ctx.volatility_regime else None,
                "obi": ctx.obi,
                "spread_bps": ctx.spread_bps,
                "volume_spike_ratio": ctx.volume_spike_ratio,
            }

        # Anomaly alert
        alert = state.get("anomaly_alert")
        if alert:
            payload["anomaly"] = {
                "type": alert.anomaly_type.value,
                "severity": alert.severity,
                "confidence": alert.ml_confidence,
                "classification": alert.llm_classification,
                "models_agreed": alert.models_agreed,
                "z_score": alert.z_score,
            }

        # Trade proposal
        proposal = state.get("trade_proposal")
        if proposal:
            payload["proposal"] = {
                "trade_id": str(proposal.trade_id),
                "asset": proposal.asset,
                "direction": proposal.direction.value,
                "entry_price": proposal.entry_price,
                "position_size_usd": proposal.position_size_usd,
                "stop_loss": proposal.stop_loss,
                "take_profit": proposal.take_profit,
                "cts_score": proposal.cts_score,
                "strategy": proposal.strategy_type.value,
                "risk_usd": proposal.risk_usd,
                "primary_rr": proposal.primary_rr,
            }

        # Risk decision
        risk = state.get("risk_decision")
        if risk:
            payload["risk"] = {
                "status": risk.status.value,
                "approved_size_usd": risk.approved_size_usd,
                "rejection_reason": risk.rejection_reason,
                "var_95": risk.var_95,
                "layer_results": risk.layer_results,
            }

        # Execution result
        result = state.get("execution_result")
        if result:
            payload["execution"] = {
                "trade_id": str(result.trade_id),
                "order_id": result.order_id,
                "filled_price": result.filled_price,
                "filled_qty": result.filled_qty,
                "slippage_bps": result.slippage_bps,
                "pnl_usd": result.pnl_usd,
                "pnl_pct": result.pnl_pct,
                "is_closed": result.is_closed,
            }

        # Portfolio
        portfolio = state.get("portfolio_state")
        if portfolio:
            payload["portfolio"] = {
                "capital_usd": portfolio.capital_usd,
                "daily_pnl_usd": portfolio.daily_pnl_usd,
                "daily_pnl_pct": portfolio.daily_pnl_pct,
                "open_positions_count": len(portfolio.open_positions),
                "circuit_breaker_active": portfolio.circuit_breaker_active,
                "trades_today": portfolio.trades_today,
                "win_rate_today": portfolio.win_rate_today,
            }

        # Session metadata
        meta = state.get("session_metadata")
        if meta:
            payload["session"] = {
                "session_id": str(meta.session_id),
                "pipeline_mode": meta.pipeline_mode,
                "fireworks_cost_usd": meta.fireworks_cost_usd,
                "agent_latencies_ms": meta.agent_latencies_ms,
                "errors": state.get("error_log", []),
                "degraded_agents": meta.degraded_agents,
            }

        payload["awaiting_human_approval"] = state.get("awaiting_human_approval", False)
        return payload


# Global singleton
broadcaster = StateBroadcaster()
