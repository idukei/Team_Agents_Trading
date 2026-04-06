from __future__ import annotations

import json
import time
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.sentiment.llm_analyzer import get_llm_client
from core.config import settings
from core.models.signals import AnomalyAlert, MarketContext, SentimentSignal
from core.models.trade import Direction, OrderType, StrategyType, TradeProposal

log = structlog.get_logger(__name__)

OPTIMIZER_SYSTEM = """Eres un trader cuantitativo experto. Recibes señales de múltiples agentes
y debes determinar los parámetros óptimos para el trade.
Responde ÚNICAMENTE con JSON exacto. Sé conservador con el riesgo."""

OPTIMIZER_USER = """SEÑAL DE SENTIMIENTO:
{sentiment_summary}

CONTEXTO TÉCNICO:
{market_summary}

SEÑAL DE ANOMALÍA:
{anomaly_summary}

ACTIVO: {asset}
DIRECCIÓN: {direction}
CTS SCORE: {cts}
CAPITAL DISPONIBLE PARA TRADE: ${capital_usd:.0f}

Reglas obligatorias:
- SL basado en ATR({atr:.2f}): SL = {sl_formula}
- R/R mínimo 1.5:1
- Máximo riesgo: 1% del capital = ${max_risk:.0f}
- TP1 cierra 50% de posición, TP2 cierra resto

Responde con JSON:
{{
  "entry_type": "LIMIT" | "MARKET",
  "entry_price": número,
  "stop_loss": número,
  "take_profit_1": número,
  "take_profit_2": número,
  "max_hold_seconds": 60-1200,
  "exit_triggers": ["lista", "de", "condiciones"],
  "strategy_type": "POLITICAL_SCALP" | "ECONOMIC_EVENT_SCALP" | "ANOMALY_BREAKOUT" | "MEAN_REVERSION",
  "reasoning": "máximo 2 oraciones"
}}"""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
async def optimize_trade_params(
    asset: str,
    direction: Direction,
    cts_score: float,
    sentiment: SentimentSignal | None,
    market_ctx: MarketContext | None,
    anomaly: AnomalyAlert | None,
    available_capital_usd: float,
) -> dict[str, Any]:
    """Call Fireworks.ai with chain-of-thought to determine optimal trade parameters."""
    t0 = time.monotonic()

    price = market_ctx.price if market_ctx else 0.0
    atr = market_ctx.atr14 if market_ctx else price * 0.01
    max_risk = available_capital_usd * settings.risk_max_trade_pct

    # ATR-based SL formula
    atr_mult = 1.5
    if direction == Direction.LONG:
        sl_formula = f"entry_price - {atr_mult} × ATR = approx ${price - atr * atr_mult:.2f}"
    else:
        sl_formula = f"entry_price + {atr_mult} × ATR = approx ${price + atr * atr_mult:.2f}"

    sentiment_summary = (
        f"Dirección: {sentiment.direction} | Magnitud: {sentiment.magnitude:.2f} | "
        f"Confianza: {sentiment.confidence:.2f} | Horizonte: {sentiment.time_horizon}\n"
        f"Razonamiento: {sentiment.llm_reasoning[:200]}"
        if sentiment else "No disponible"
    )

    market_summary = (
        f"Precio: ${price:.2f} | RSI: {market_ctx.rsi14:.1f} | ATR: {atr:.4f} | "
        f"VWAP: ${market_ctx.vwap:.2f} | OBI: {market_ctx.obi:+.3f} | "
        f"Tendencia: {market_ctx.trend} | Niveles soporte: {market_ctx.key_levels.get('support', [])} | "
        f"Niveles resistencia: {market_ctx.key_levels.get('resistance', [])}"
        if market_ctx else "No disponible"
    )

    anomaly_summary = (
        f"Tipo: {anomaly.anomaly_type} | Clasificación: {anomaly.llm_classification} | "
        f"Confianza: {anomaly.ml_confidence:.2f}"
        if anomaly else "No hay anomalía detectada"
    )

    client = get_llm_client()
    response = await client.chat.completions.create(
        model=settings.model_llm_large,
        messages=[
            {"role": "system", "content": OPTIMIZER_SYSTEM},
            {"role": "user", "content": OPTIMIZER_USER.format(
                sentiment_summary=sentiment_summary,
                market_summary=market_summary,
                anomaly_summary=anomaly_summary,
                asset=asset,
                direction=direction,
                cts=cts_score,
                capital_usd=available_capital_usd,
                atr=atr,
                sl_formula=sl_formula,
                max_risk=max_risk,
            )},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=400,
    )

    latency_ms = int((time.monotonic() - t0) * 1000)
    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    log.info("trade_params_optimized", asset=asset, direction=direction, latency_ms=latency_ms)
    return data


def validate_and_build_proposal(
    trade_id_hint: str,
    asset: str,
    direction: Direction,
    cts_score: float,
    params: dict[str, Any],
    available_capital_usd: float,
    event_id: str | None = None,
) -> TradeProposal:
    """
    Deterministic validator: ensures LLM params meet hard constraints.
    Adjusts if needed, raises ValueError if impossible.
    """
    entry_price = float(params.get("entry_price", 0))
    stop_loss = float(params.get("stop_loss", 0))
    tp1 = float(params.get("take_profit_1", 0))
    tp2 = float(params.get("take_profit_2", 0))
    max_hold = int(params.get("max_hold_seconds", 600))

    if entry_price <= 0:
        raise ValueError("Invalid entry price from LLM")

    # Calculate position size based on max 1% risk rule
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit <= 0:
        raise ValueError("Stop loss equals entry price")

    max_risk_usd = available_capital_usd * settings.risk_max_trade_pct
    max_units = max_risk_usd / risk_per_unit
    position_size_usd = min(max_units * entry_price, available_capital_usd * 0.20)

    # Validate R/R ratio (auto-adjust TP if needed)
    reward = abs(tp1 - entry_price)
    risk = risk_per_unit
    actual_rr = reward / risk if risk > 0 else 0.0

    if actual_rr < settings.risk_min_rr:
        # Auto-adjust TP1 to achieve minimum R/R
        if direction == Direction.LONG:
            tp1 = entry_price + risk * settings.risk_min_rr
            tp2 = entry_price + risk * settings.risk_min_rr * 2
        else:
            tp1 = entry_price - risk * settings.risk_min_rr
            tp2 = entry_price - risk * settings.risk_min_rr * 2
        log.info("tp_adjusted_for_min_rr", new_tp1=tp1, new_tp2=tp2)

    entry_type_str = str(params.get("entry_type", "LIMIT")).upper()
    entry_type = OrderType.LIMIT if entry_type_str == "LIMIT" else OrderType.MARKET

    strategy_str = str(params.get("strategy_type", "POLITICAL_SCALP")).upper()
    strategy = StrategyType(strategy_str) if strategy_str in StrategyType._value2member_map_ else StrategyType.POLITICAL_SCALP

    return TradeProposal(
        event_id=event_id,
        asset=asset,
        direction=direction,
        entry_type=entry_type,
        entry_price=round(entry_price, 4),
        position_size_usd=round(position_size_usd, 2),
        stop_loss=round(stop_loss, 4),
        take_profit=[round(tp1, 4), round(tp2, 4)],
        max_hold_seconds=min(max(60, max_hold), 1800),
        exit_triggers=params.get("exit_triggers", []),
        cts_score=cts_score,
        strategy_type=strategy,
        reasoning_summary=str(params.get("reasoning", ""))[:400],
    )
