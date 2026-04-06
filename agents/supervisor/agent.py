from __future__ import annotations

import time

import structlog

from agents.supervisor.budget_tracker import budget_tracker
from agents.supervisor.health_monitor import HealthMonitor
from core.models.trade import Direction
from graph.state import TradingState

log = structlog.get_logger(__name__)


class SupervisorAgent:
    """
    Supervisor Agent: entry node in the LangGraph graph.

    Responsibilities:
    1. Validate incoming EventTrigger
    2. Conflict resolution (sentiment vs anomaly direction disagreement)
    3. Health check and pipeline mode determination
    4. Budget guard
    """

    def __init__(self) -> None:
        self._health = HealthMonitor()

    async def process(self, state: TradingState) -> dict:
        t0 = time.monotonic()
        trigger = state.get("event_trigger")
        metadata = state.get("session_metadata")
        portfolio = state.get("portfolio_state")

        if trigger is None:
            return {}

        # Budget check
        if budget_tracker.is_budget_exceeded:
            log.warning("supervisor_budget_exceeded_degraded_mode")
            if metadata:
                metadata.pipeline_mode = "TECHNICAL_ONLY"
                metadata.degraded_agents.append("fireworks_budget")

        # Conflict resolution between sentiment and anomaly (post-analysis)
        sentiment = state.get("sentiment_signal")
        anomaly = state.get("anomaly_alert")

        if sentiment and anomaly:
            sent_dir = sentiment.direction
            anomaly_dir_raw = anomaly.expected_move.get("direction", "UNKNOWN")
            anomaly_dir = Direction.LONG if anomaly_dir_raw == "UP" else Direction.SHORT if anomaly_dir_raw == "DOWN" else None

            if anomaly_dir and sent_dir in (Direction.LONG, Direction.SHORT) and sent_dir != anomaly_dir:
                # Conflict: apply tiebreaker based on confidence
                log.info(
                    "supervisor_signal_conflict",
                    sentiment=sent_dir,
                    anomaly=anomaly_dir,
                    sent_conf=sentiment.confidence,
                    anomaly_conf=anomaly.ml_confidence,
                )
                if anomaly.ml_confidence > sentiment.confidence + 0.15:
                    # Anomaly confidence much higher → mark sentiment conflicted
                    from core.models.signals import SentimentSignal
                    new_sentiment = sentiment.model_copy(update={"conflicted": True})
                    return {"sentiment_signal": new_sentiment}
                else:
                    # Sentiment wins (default) → reduce anomaly weight
                    log.info("supervisor_sentiment_wins_conflict")

        latency_ms = int((time.monotonic() - t0) * 1000)
        if metadata:
            metadata.record_agent_latency("supervisor", latency_ms)

        return {}
