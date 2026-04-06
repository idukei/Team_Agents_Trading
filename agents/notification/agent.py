from __future__ import annotations

import structlog
from telegram.ext import Application

from agents.notification.approval_handler import ApprovalHandler, ApprovalResult
from agents.notification.message_templates import (
    fmt_anomaly_alert,
    fmt_circuit_breaker,
    fmt_economic_event,
    fmt_leader_alert,
    fmt_risk_rejected,
    fmt_system_status,
    fmt_trade_closed,
    fmt_trade_executed,
    fmt_trade_proposal,
)
from core.config import settings
from core.models.events import EventSource, EventTrigger
from core.models.signals import AnomalyAlert
from core.models.trade import ExecutionResult, RiskDecision, TradeProposal
from graph.state import TradingState

log = structlog.get_logger(__name__)

POLITICAL_SOURCES = {
    EventSource.TRUMP_TRUTH_SOCIAL,
    EventSource.TRUMP_X,
    EventSource.FED_POWELL,
    EventSource.FED_ANNOUNCEMENT,
    EventSource.ECB_LAGARDE,
    EventSource.ECB_ANNOUNCEMENT,
    EventSource.FOMC,
    EventSource.TREASURY,
    EventSource.OPEC,
}


class NotificationAgent:
    """
    Sends structured Telegram notifications at each pipeline stage.
    Also handles human-in-the-loop approval for trade proposals.
    """

    def __init__(self, app: Application, approval_handler: ApprovalHandler) -> None:
        self._app = app
        self._approval = approval_handler

    async def send(self, text: str) -> None:
        try:
            await self._app.bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=text,
                parse_mode="Markdown",
            )
        except Exception as e:
            log.warning("telegram_send_failed", error=str(e))

    async def notify_event_detected(self, trigger: EventTrigger) -> None:
        if trigger.event_source in POLITICAL_SOURCES:
            text = fmt_leader_alert(trigger)
        elif trigger.event_source in {
            EventSource.CPI, EventSource.NFP, EventSource.PCE,
            EventSource.GDP, EventSource.FOMC, EventSource.PPI,
        }:
            text = fmt_economic_event(trigger)
        else:
            return   # Don't notify for every price/news event to avoid noise
        await self.send(text)

    async def notify_anomaly(self, alert: AnomalyAlert) -> None:
        if alert.severity in ("HIGH", "CRITICAL"):
            await self.send(fmt_anomaly_alert(alert))

    async def notify_trade_proposal(self, proposal: TradeProposal) -> None:
        await self.send(fmt_trade_proposal(proposal))

    async def request_trade_approval(self, proposal: TradeProposal) -> ApprovalResult:
        """Send trade proposal with approve/reject buttons. Return decision."""
        text = fmt_trade_proposal(proposal)
        return await self._approval.request_approval(
            trade_id=proposal.trade_id,
            message_text=text,
        )

    async def notify_trade_executed(self, proposal: TradeProposal, result: ExecutionResult) -> None:
        await self.send(fmt_trade_executed(proposal, result))

    async def notify_trade_closed(self, proposal: TradeProposal, result: ExecutionResult) -> None:
        await self.send(fmt_trade_closed(proposal, result))

    async def notify_risk_rejected(self, proposal: TradeProposal, decision: RiskDecision) -> None:
        await self.send(fmt_risk_rejected(proposal, decision))

    async def notify_circuit_breaker(self, reason: str, daily_pnl_pct: float) -> None:
        await self.send(fmt_circuit_breaker(reason, daily_pnl_pct))


# ── LangGraph node function ───────────────────────────────────────────────────

async def notification_node(state: TradingState) -> dict:
    """
    LangGraph node: sends Telegram notification based on current state.
    Called twice in the pipeline:
      1. After risk_node (propose trade or notify rejection)
      2. After execution_node (confirm fill or close)
    """
    # This thin wrapper is called from graph/nodes.py which injects the agent instance
    # The actual notification logic uses NotificationAgent methods
    return {}
