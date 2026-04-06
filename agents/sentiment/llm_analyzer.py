from __future__ import annotations

import json
import time
from typing import Any

import structlog
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import settings
from core.exceptions import FireworksBudgetExceededError, FireworksError
from core.models.signals import SentimentSignal
from core.models.trade import Direction, TimeHorizon

log = structlog.get_logger(__name__)

# Fireworks.ai cost per 1M tokens (approximate)
MODEL_COST_PER_1M: dict[str, float] = {
    settings.model_llm_large: 0.90,
    settings.model_llm_fast: 0.20,
    settings.model_llm_cheap: 0.10,
    settings.model_llm_cot: 3.00,
}

_client: AsyncOpenAI | None = None
_daily_cost_usd: float = 0.0


def get_llm_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=settings.fireworks_base_url,
            api_key=settings.fireworks_api_key.get_secret_value(),
        )
    return _client


def _estimate_cost(model: str, total_tokens: int) -> float:
    cost_per_1m = MODEL_COST_PER_1M.get(model, 0.90)
    return (total_tokens / 1_000_000) * cost_per_1m


SENTIMENT_SYSTEM_PROMPT = """Eres un analista cuantitativo de trading de alto nivel.
Recibes una declaración/noticia y debes determinar su impacto en el mercado.
Responde ÚNICAMENTE en JSON con el esquema exacto indicado. Sin texto adicional.
Sé preciso y conciso. Basa tu análisis en el contexto histórico proporcionado."""

SENTIMENT_USER_TEMPLATE = """CONTEXTO HISTÓRICO (precedentes similares):
{precedents}

CONDICIONES ACTUALES DEL MERCADO:
{market_summary}

DECLARACIÓN/NOTICIA A ANALIZAR:
"{content}"
Fuente: {source} | Timestamp: {timestamp}

Responde con este JSON exacto:
{{
  "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
  "primary_asset": "ticker del activo más afectado",
  "secondary_assets": ["lista", "de", "tickers"],
  "magnitude": 0.0-1.0,
  "confidence": 0.0-1.0,
  "time_horizon": "IMMEDIATE" | "5MIN" | "30MIN" | "1H",
  "expected_volatility_spike": true | false,
  "precedent_summary": "resumen del precedente más relevante o null",
  "precedent_outcome_pct": número o null,
  "llm_reasoning": "razonamiento en máximo 2 oraciones"
}}"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def analyze_sentiment(
    content: str,
    source: str,
    timestamp: str,
    precedents_text: str,
    market_summary: str,
    model: str | None = None,
) -> tuple[dict[str, Any], int, float]:
    """
    Call Fireworks.ai to analyze sentiment.
    Returns (parsed_response_dict, total_tokens, cost_usd).
    """
    global _daily_cost_usd

    if _daily_cost_usd >= settings.fireworks_budget_daily_usd:
        raise FireworksBudgetExceededError(
            f"Daily budget ${settings.fireworks_budget_daily_usd} exceeded"
        )

    # Choose model based on content complexity
    selected_model = model or (
        settings.model_llm_large if len(content) > 200 else settings.model_llm_cheap
    )

    messages = [
        {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
        {"role": "user", "content": SENTIMENT_USER_TEMPLATE.format(
            precedents=precedents_text,
            market_summary=market_summary,
            content=content[:1000],
            source=source,
            timestamp=timestamp,
        )},
    ]

    client = get_llm_client()
    t0 = time.monotonic()
    response = await client.chat.completions.create(
        model=selected_model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=512,
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    total_tokens = response.usage.total_tokens if response.usage else 0
    cost = _estimate_cost(selected_model, total_tokens)
    _daily_cost_usd += cost

    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise FireworksError(f"LLM returned invalid JSON: {raw[:200]}")

    log.info(
        "llm_sentiment_call",
        model=selected_model,
        tokens=total_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
    )
    return parsed, total_tokens, cost


def parse_sentiment_signal(
    data: dict[str, Any],
    model_used: str,
    latency_ms: int,
) -> SentimentSignal:
    """Parse raw LLM JSON dict into validated SentimentSignal."""
    direction_map = {"BULLISH": Direction.LONG, "BEARISH": Direction.SHORT, "NEUTRAL": Direction.NEUTRAL}
    horizon_map = {
        "IMMEDIATE": TimeHorizon.IMMEDIATE,
        "5MIN": TimeHorizon.FIVE_MIN,
        "30MIN": TimeHorizon.THIRTY_MIN,
        "1H": TimeHorizon.ONE_HOUR,
    }

    direction = direction_map.get(data.get("direction", "NEUTRAL"), Direction.NEUTRAL)
    horizon = horizon_map.get(data.get("time_horizon", "IMMEDIATE"), TimeHorizon.IMMEDIATE)

    return SentimentSignal(
        direction=direction,
        primary_asset=str(data.get("primary_asset", "SPY")).upper(),
        secondary_assets=[str(a).upper() for a in data.get("secondary_assets", [])],
        magnitude=float(min(1.0, max(0.0, data.get("magnitude", 0.5)))),
        confidence=float(min(1.0, max(0.0, data.get("confidence", 0.5)))),
        time_horizon=horizon,
        expected_volatility_spike=bool(data.get("expected_volatility_spike", False)),
        precedent_summary=data.get("precedent_summary"),
        precedent_outcome_pct=data.get("precedent_outcome_pct"),
        llm_reasoning=str(data.get("llm_reasoning", ""))[:500],
        fireworks_model_used=model_used,
        latency_ms=latency_ms,
    )
