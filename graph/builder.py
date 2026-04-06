from __future__ import annotations

import structlog
from langgraph.graph import END, START, StateGraph

from graph.checkpointer import get_checkpointer
from graph.edges import route_after_notification, route_after_risk, route_after_supervisor
from graph.nodes import (
    anomaly_node,
    end_node_fn,
    execution_node,
    market_data_node,
    notify_proposal_node_fn,
    notify_result_node_fn,
    risk_node,
    sentiment_node,
    strategy_node,
    supervisor_node_fn,
)
from graph.state import TradingState

log = structlog.get_logger(__name__)


async def build_trading_graph():
    """
    Build and compile the TeamTrade LangGraph StateGraph.

    Graph topology:
    START → supervisor → [scatter] → sentiment (parallel)
                                   → market_data (parallel)
                                   → anomaly (parallel)
           ↓ [converge at strategy]
           strategy → risk → notify_proposal →[interrupt]→ execution → notify_result → END
                          ↘ (rejected)         notify_result → END
    """
    builder = StateGraph(TradingState)

    # ── Add nodes ──────────────────────────────────────────────────────────────
    builder.add_node("supervisor", supervisor_node_fn)
    builder.add_node("sentiment", sentiment_node)
    builder.add_node("market_data", market_data_node)
    builder.add_node("anomaly", anomaly_node)
    builder.add_node("strategy", strategy_node)
    builder.add_node("risk", risk_node)
    builder.add_node("notify_proposal", notify_proposal_node_fn)
    builder.add_node("execution", execution_node)
    builder.add_node("notify_result", notify_result_node_fn)
    builder.add_node("end_node", end_node_fn)

    # ── Entry point ────────────────────────────────────────────────────────────
    builder.add_edge(START, "supervisor")

    # ── Supervisor → scatter via Send API ─────────────────────────────────────
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "sentiment": "sentiment",
            "market_data": "market_data",
            "anomaly": "anomaly",
            "end_node": "end_node",
        },
    )

    # ── Analysis agents converge at strategy ──────────────────────────────────
    builder.add_edge("sentiment", "strategy")
    builder.add_edge("market_data", "strategy")
    builder.add_edge("anomaly", "strategy")

    # ── Decision pipeline ─────────────────────────────────────────────────────
    builder.add_edge("strategy", "risk")

    builder.add_conditional_edges(
        "risk",
        route_after_risk,
        {
            "notify_proposal": "notify_proposal",
            "notify_result": "notify_result",
            "end_node": "end_node",
        },
    )

    # ── Notification → execution (with interrupt point) ───────────────────────
    builder.add_conditional_edges(
        "notify_proposal",
        route_after_notification,
        {
            "execution": "execution",
            "__end__": END,
        },
    )

    # ── Post-execution flow ───────────────────────────────────────────────────
    builder.add_edge("execution", "notify_result")
    builder.add_edge("notify_result", END)
    builder.add_edge("end_node", END)

    # ── Compile with PostgresSaver checkpointer ───────────────────────────────
    checkpointer = await get_checkpointer()

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["execution"],     # pause before executing — human-in-the-loop
    )

    log.info("trading_graph_compiled")
    return graph
