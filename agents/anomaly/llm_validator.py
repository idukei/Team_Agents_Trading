from __future__ import annotations

import json
import time

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.sentiment.llm_analyzer import get_llm_client
from core.config import settings
from core.models.signals import AnomalyClassification

log = structlog.get_logger(__name__)

VALIDATION_PROMPT = """Eres un analista de mercado experto en detección de anomalías.
Recibes datos de una anomalía estadística detectada y debes clasificarla.
Responde ÚNICAMENTE con JSON exacto."""

VALIDATION_USER = """ANOMALÍA DETECTADA:
- Activo: {asset}
- Tipo: {anomaly_type}
- Z-score: {z_score}
- Ratio de volumen: {volume_ratio}
- OBI (imbalance de libro): {obi}
- Evento reciente: {recent_event}

Clasifica como uno de:
INSTITUTIONAL_ACCUMULATION, INSTITUTIONAL_DISTRIBUTION, NEWS_OVERREACTION,
BULL_TRAP, BEAR_TRAP, DATA_GLITCH, GENUINE_BREAKOUT, UNKNOWN

Responde: {{"classification": "...", "reasoning": "max 1 oración"}}"""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
async def validate_anomaly_llm(
    asset: str,
    anomaly_type: str,
    z_score: float,
    volume_ratio: float,
    obi: float,
    recent_event: str = "No hay evento reciente conocido",
) -> tuple[AnomalyClassification, str]:
    """
    Uses fast cheap model (llama-8b) to classify detected anomaly.
    Returns (classification, reasoning).
    """
    t0 = time.monotonic()
    client = get_llm_client()

    response = await client.chat.completions.create(
        model=settings.model_llm_fast,         # fast + cheap
        messages=[
            {"role": "system", "content": VALIDATION_PROMPT},
            {"role": "user", "content": VALIDATION_USER.format(
                asset=asset,
                anomaly_type=anomaly_type,
                z_score=f"{z_score:.2f}",
                volume_ratio=f"{volume_ratio:.2f}",
                obi=f"{obi:.3f}",
                recent_event=recent_event[:200],
            )},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=100,
    )

    latency_ms = int((time.monotonic() - t0) * 1000)
    raw = response.choices[0].message.content or "{}"

    try:
        data = json.loads(raw)
        raw_cls = data.get("classification", "UNKNOWN").upper()
        classification = AnomalyClassification(raw_cls) if raw_cls in AnomalyClassification._value2member_map_ else AnomalyClassification.UNKNOWN
        reasoning = str(data.get("reasoning", ""))[:200]
    except (json.JSONDecodeError, ValueError):
        classification = AnomalyClassification.UNKNOWN
        reasoning = "Parse error"

    log.info(
        "anomaly_llm_validated",
        asset=asset,
        classification=classification,
        latency_ms=latency_ms,
    )
    return classification, reasoning
