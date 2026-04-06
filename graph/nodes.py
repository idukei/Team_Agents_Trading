from __future__ import annotations

"""
Thin wrapper node functions that call each agent.
The LangGraph StateGraph calls these functions — they must be module-level callables.

Agent instances (notification, etc.) are injected via module-level singletons
set at startup in main.py.
"""

from agents.anomaly.agent import anomaly_node as _anomaly_node
from agents.execution.agent import execution_node as _execution_node
from agents.market_data.agent import market_data_node as _market_data_node
from agents.risk.agent import risk_node as _risk_node
from agents.sentiment.agent import sentiment_node as _sentiment_node
from agents.strategy.agent import strategy_node as _strategy_node
from graph.state import TradingState

# Agent instances — set by main.py before graph compilation
_notification_agent = None
_supervisor_agent = None


async def _broadcast(state: TradingState, node_name: str) -> None:
    """Push state snapshot to all connected dashboard WebSocket clients."""
    try:
        from api.state_broadcaster import broadcaster
        payload = broadcaster.serialize_state(dict(state), node_name)
        await broadcaster.broadcast(payload)
    except Exception:
        pass  # dashboard broadcast must never block or crash the trading pipeline


def set_notification_agent(agent) -> None:
    global _notification_agent
    _notification_agent = agent


def set_supervisor_agent(agent) -> None:
    global _supervisor_agent
    _supervisor_agent = agent


async def supervisor_node_fn(state: TradingState) -> dict:
    """Entry node: validates state, applies conflict resolution, sets pipeline mode."""
    await _broadcast(state, "supervisor")
    if _supervisor_agent:
        result = await _supervisor_agent.process(state)
        merged = {**dict(state), **result}
        await _broadcast(merged, "supervisor_done")
        return result
    return {}


async def sentiment_node(state: TradingState) -> dict:
    await _broadcast(state, "sentiment")
    result = await _sentiment_node(state)
    await _broadcast({**dict(state), **result}, "sentiment_done")
    return result


async def market_data_node(state: TradingState) -> dict:
    await _broadcast(state, "market_data")
    result = await _market_data_node(state)
    await _broadcast({**dict(state), **result}, "market_data_done")
    return result


async def anomaly_node(state: TradingState) -> dict:
    await _broadcast(state, "anomaly")
    result = await _anomaly_node(state)
    await _broadcast({**dict(state), **result}, "anomaly_done")
    return result


async def strategy_node(state: TradingState) -> dict:
    await _broadcast(state, "strategy")
    result = await _strategy_node(state)
    await _broadcast({**dict(state), **result}, "strategy_done")
    return result


async def risk_node(state: TradingState) -> dict:
    await _broadcast(state, "risk")
    result = await _risk_node(state)
    await _broadcast({**dict(state), **result}, "risk_done")
    return result


async def execution_node(state: TradingState) -> dict:
    await _broadcast(state, "execution")
    result = await _execution_node(state)
    await _broadcast({**dict(state), **result}, "execution_done")
    return result


async def notify_proposal_node_fn(state: TradingState) -> dict:
    """Send trade proposal to Telegram + optionally wait for approval."""
    if _notification_agent is None:
        return {}

    proposal = state.get("trade_proposal")
    risk_decision = state.get("risk_decision")

    if proposal is None:
        return {}

    from core.config import settings
    from agents.notification.approval_handler import ApprovalResult

    # Always notify
    await _notification_agent.notify_trade_proposal(proposal)

    # Human-in-the-loop: wait for approval if configured
    # Default: automatic (no waiting) for paper trading
    result = ApprovalResult.APPROVED   # automatic mode
    # Uncomment for manual approval mode:
    # result = await _notification_agent.request_trade_approval(proposal)

    if result == ApprovalResult.REJECTED:
        from core.models.trade import RiskDecision, RiskDecisionStatus
        return {
            "risk_decision": RiskDecision(
                trade_id=proposal.trade_id,
                status=RiskDecisionStatus.REJECTED,
                rejection_reason="Human rejected via Telegram",
            ),
            "awaiting_human_approval": False,
        }

    return {"awaiting_human_approval": False}


async def notify_result_node_fn(state: TradingState) -> dict:
    """Send final trade result (execution or rejection) to Telegram."""
    if _notification_agent is None:
        return {}

    proposal = state.get("trade_proposal")
    risk_decision = state.get("risk_decision")
    execution_result = state.get("execution_result")

    if proposal is None:
        return {}

    if execution_result:
        await _notification_agent.notify_trade_executed(proposal, execution_result)
    elif risk_decision and risk_decision.status == "REJECTED":
        await _notification_agent.notify_risk_rejected(proposal, risk_decision)

    return {}


async def end_node_fn(state: TradingState) -> dict:
    """No-op end node."""
    return {}


# Export all node functions for graph builder
__all__ = [
    "supervisor_node_fn",
    "sentiment_node",
    "market_data_node",
    "anomaly_node",
    "strategy_node",
    "risk_node",
    "execution_node",
    "notify_proposal_node_fn",
    "notify_result_node_fn",
    "end_node_fn",
    "set_notification_agent",
    "set_supervisor_agent",
]
