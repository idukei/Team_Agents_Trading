from __future__ import annotations

from typing import Annotated

from typing_extensions import TypedDict

from core.models.events import EventTrigger
from core.models.portfolio import PortfolioState, SessionMetadata
from core.models.signals import AnomalyAlert, MarketContext, SentimentSignal
from core.models.trade import ExecutionResult, RiskDecision, TradeProposal


def _append_errors(existing: list[str], new: list[str]) -> list[str]:
    """Reducer: accumulate errors without overwriting previous entries."""
    return existing + new


class TradingState(TypedDict):
    """
    Global state flowing through the LangGraph StateGraph.

    Design decisions:
    - error_log uses Annotated reducer so parallel nodes can append without race conditions.
    - All other fields are simple last-write-wins (only one node writes each).
    - awaiting_human_approval is used by the interrupt mechanism.
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    event_trigger: EventTrigger | None

    # ── Analysis outputs (populated in parallel) ──────────────────────────────
    sentiment_signal: SentimentSignal | None
    market_context: MarketContext | None
    anomaly_alert: AnomalyAlert | None

    # ── Decision pipeline ─────────────────────────────────────────────────────
    trade_proposal: TradeProposal | None
    risk_decision: RiskDecision | None

    # ── Execution ─────────────────────────────────────────────────────────────
    execution_result: ExecutionResult | None

    # ── Persistent state ──────────────────────────────────────────────────────
    portfolio_state: PortfolioState
    session_metadata: SessionMetadata

    # ── Control flags ─────────────────────────────────────────────────────────
    awaiting_human_approval: bool

    # ── Accumulator (reducer prevents overwrite) ──────────────────────────────
    error_log: Annotated[list[str], _append_errors]


def initial_state(
    portfolio_state: PortfolioState,
    session_metadata: SessionMetadata,
) -> TradingState:
    """Return a clean initial state for a new pipeline run."""
    return TradingState(
        event_trigger=None,
        sentiment_signal=None,
        market_context=None,
        anomaly_alert=None,
        trade_proposal=None,
        risk_decision=None,
        execution_result=None,
        portfolio_state=portfolio_state,
        session_metadata=session_metadata,
        awaiting_human_approval=False,
        error_log=[],
    )
