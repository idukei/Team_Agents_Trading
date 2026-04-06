from __future__ import annotations

from langgraph.types import Send

from core.config import settings
from core.models.trade import RiskDecisionStatus
from graph.state import TradingState


def route_after_supervisor(state: TradingState) -> list[Send] | str:
    """
    Scatter pattern: fan-out to parallel analysis agents based on EPS score.

    Full pipeline (EPS >= 75):  sentiment + market_data + anomaly in parallel
    Light pipeline (EPS 50-75): market_data only
    Skip (EPS < 50):            end
    """
    trigger = state.get("event_trigger")
    if trigger is None:
        return "end_node"

    eps = trigger.eps_score

    if eps >= settings.eps_full_pipeline:
        return [
            Send("sentiment", state),
            Send("market_data", state),
            Send("anomaly", state),
        ]
    elif eps >= settings.eps_light_pipeline:
        return [
            Send("market_data", state),
            Send("anomaly", state),
        ]
    else:
        return "end_node"


def route_after_risk(state: TradingState) -> str:
    """Route to notification (approve/reject) or end based on risk decision."""
    risk_decision = state.get("risk_decision")
    if risk_decision is None:
        return "end_node"

    if risk_decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.APPROVED_REDUCED):
        return "notify_proposal"
    else:
        return "notify_result"


def route_after_notification(state: TradingState) -> str:
    """
    After notifying proposal:
    - If awaiting human approval → interrupt (graph pauses here)
    - Otherwise → proceed to execution

    The graph is compiled with interrupt_before=["execution"],
    so this edge just routes to execution; LangGraph handles the interrupt.
    """
    if state.get("awaiting_human_approval", False):
        return "__end__"     # will be interrupted by LangGraph before execution
    return "execution"
