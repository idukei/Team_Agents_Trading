from __future__ import annotations

import time
from datetime import datetime, timezone

import structlog

from agents.sentiment.coherence import validate_coherence
from agents.sentiment.llm_analyzer import analyze_sentiment, parse_sentiment_signal
from agents.sentiment.rag import format_precedents_for_prompt, query_precedents
from core.config import settings
from core.exceptions import FireworksBudgetExceededError
from core.models.signals import MarketContext, SentimentSignal
from graph.state import TradingState

log = structlog.get_logger(__name__)


def _build_market_summary(ctx: MarketContext | None) -> str:
    if ctx is None:
        return "Datos de mercado no disponibles."
    return (
        f"Activo: {ctx.asset} @ ${ctx.price:.2f} | RSI: {ctx.rsi14:.1f} | "
        f"Tendencia: {ctx.trend} | Volatilidad: {ctx.volatility_regime} | "
        f"OBI: {ctx.obi:+.3f} | VWAP: ${ctx.vwap:.2f} | "
        f"Volumen ratio: {ctx.volume_spike_ratio:.1f}x"
    )


async def sentiment_node(state: TradingState) -> dict:
    """
    LangGraph node: analyses event sentiment using RAG + Fireworks.ai LLM.

    Pipeline (all < 500ms combined):
    1. RAG query → Qdrant for similar precedents (< 100ms)
    2. LLM analysis → Fireworks.ai (< 400ms)
    3. Coherence check → deterministic (< 5ms)
    """
    t0 = time.monotonic()
    trigger = state.get("event_trigger")
    if trigger is None:
        return {"sentiment_signal": None}

    pipeline_mode = state.get("session_metadata", {})

    # Phase 1: RAG query
    precedents = await query_precedents(trigger.raw_content, limit=5)
    precedents_text = format_precedents_for_prompt(precedents)

    # Phase 2: LLM analysis
    market_ctx = state.get("market_context")
    market_summary = _build_market_summary(market_ctx)
    timestamp_str = datetime.fromtimestamp(
        trigger.timestamp_ms / 1000, tz=timezone.utc
    ).isoformat()

    signal: SentimentSignal | None = None

    try:
        data, total_tokens, cost = await analyze_sentiment(
            content=trigger.raw_content,
            source=trigger.event_source,
            timestamp=timestamp_str,
            precedents_text=precedents_text,
            market_summary=market_summary,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        signal = parse_sentiment_signal(data, data.get("model_used", settings.model_llm_large), latency_ms)

        # Update session budget tracking
        metadata = state.get("session_metadata")
        if metadata:
            metadata.record_llm_usage(total_tokens, cost)
            metadata.record_agent_latency("sentiment", latency_ms)

    except FireworksBudgetExceededError:
        log.warning("sentiment_budget_exceeded_skipping")
        # Graceful degradation: pipeline continues without sentiment signal
        metadata = state.get("session_metadata")
        if metadata:
            metadata.degraded_agents.append("sentiment")
        return {"sentiment_signal": None}

    except Exception as e:
        log.error("sentiment_llm_error", error=str(e))
        metadata = state.get("session_metadata")
        if metadata:
            metadata.errors.append(f"sentiment: {str(e)[:100]}")
        return {"sentiment_signal": None, "error_log": [f"sentiment_error: {str(e)[:100]}"]}

    # Phase 3: Coherence validation
    signal = validate_coherence(signal)

    log.info(
        "sentiment_analyzed",
        asset=signal.primary_asset,
        direction=signal.direction,
        magnitude=signal.magnitude,
        confidence=signal.confidence,
        conflicted=signal.conflicted,
        latency_ms=signal.latency_ms,
    )

    return {"sentiment_signal": signal}
