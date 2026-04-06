from __future__ import annotations

from core.models.events import EventTrigger
from core.models.signals import AnomalyAlert, SentimentSignal
from core.models.trade import ExecutionResult, RiskDecision, TradeProposal


def fmt_leader_alert(trigger: EventTrigger) -> str:
    assets = ", ".join(trigger.affected_assets_guess[:4]) or "múltiples activos"
    return (
        f"*ALERTA CRITICA — {trigger.event_source}*\n"
        f"\n"
        f"*EPS Score:* `{trigger.eps_score:.0f}/100` | *Urgencia:* `{trigger.urgency}`\n"
        f"*Activos afectados:* `{assets}`\n"
        f"\n"
        f"*Declaración:*\n"
        f"_{trigger.raw_content[:400]}_\n"
        f"\n"
        f"Pipeline activado. Analizando..."
    )


def fmt_economic_event(trigger: EventTrigger) -> str:
    parts = [
        f"*EVENTO ECONÓMICO — {trigger.event_name or trigger.event_source}*",
        "",
    ]
    if trigger.actual is not None:
        parts.append(f"*Actual:* `{trigger.actual}`")
    if trigger.consensus is not None:
        parts.append(f"*Consenso:* `{trigger.consensus}`")
    if trigger.surprise is not None:
        direction = "POSITIVO" if trigger.surprise > 0 else "NEGATIVO"
        parts.append(f"*Sorpresa:* `{trigger.surprise:+.3f}` ({direction})")
    parts += ["", f"*EPS:* `{trigger.eps_score:.0f}` | Analizando impacto..."]
    return "\n".join(parts)


def fmt_trade_proposal(proposal: TradeProposal) -> str:
    direction_emoji = "" if proposal.direction.value == "LONG" else ""
    tp_str = " | ".join(f"`${tp:.2f}`" for tp in proposal.take_profit)
    return (
        f"{direction_emoji} *TRADE PROPUESTO — {proposal.asset} {proposal.direction}*\n"
        f"\n"
        f"*Entrada:* `{proposal.entry_type} @ ${proposal.entry_price:.2f}`\n"
        f"*Tamaño:* `${proposal.position_size_usd:.0f}`\n"
        f"*Stop Loss:* `${proposal.stop_loss:.2f}`\n"
        f"*Take Profits:* {tp_str}\n"
        f"*Max tiempo:* `{proposal.max_hold_seconds}s`\n"
        f"\n"
        f"*CTS Score:* `{proposal.cts_score:.2f}` | *Estrategia:* `{proposal.strategy_type}`\n"
        f"\n"
        f"*Razonamiento:*\n"
        f"_{proposal.reasoning_summary[:300]}_"
    )


def fmt_trade_executed(proposal: TradeProposal, result: ExecutionResult) -> str:
    direction_emoji = "" if proposal.direction.value == "LONG" else ""
    slippage = f"+{result.slippage_bps:.1f}" if result.slippage_bps >= 0 else f"{result.slippage_bps:.1f}"
    return (
        f"*ORDEN EJECUTADA — {proposal.asset} {proposal.direction}*\n"
        f"\n"
        f"*Precio fill:* `${result.filled_price:.2f}` (slippage: `{slippage}bps`)\n"
        f"*Cantidad:* `{result.filled_qty}`\n"
        f"*Order ID:* `{result.order_id}`\n"
        f"\n"
        f"_Gestionando posición automáticamente..._"
    )


def fmt_trade_closed(proposal: TradeProposal, result: ExecutionResult) -> str:
    pnl = result.pnl_usd or 0.0
    pnl_pct = result.pnl_pct or 0.0
    emoji = "" if pnl >= 0 else ""

    exit_reason = "TP2" if result.tp2_hit else "TP1" if result.tp1_hit else "SL" if result.sl_hit else "TIMEOUT" if result.timeout_exit else "SEÑAL CONTRARIA"
    duration = f"{result.duration_seconds:.0f}s" if result.duration_seconds else "?"

    return (
        f"{emoji} *TRADE CERRADO — {proposal.asset}*\n"
        f"\n"
        f"*P&L:* `{pnl:+.2f} USD` (`{pnl_pct:+.2f}%`)\n"
        f"*Motivo salida:* `{exit_reason}`\n"
        f"*Duración:* `{duration}`\n"
        f"*CTS fue:* `{proposal.cts_score:.2f}`"
    )


def fmt_risk_rejected(proposal: TradeProposal, decision: RiskDecision) -> str:
    return (
        f"*TRADE RECHAZADO — {proposal.asset}*\n"
        f"\n"
        f"*Motivo:* `{decision.rejection_reason}`\n"
        f"*CTS Score:* `{proposal.cts_score:.2f}`"
    )


def fmt_anomaly_alert(alert: AnomalyAlert) -> str:
    return (
        f"*ANOMALIA DETECTADA — {alert.asset}*\n"
        f"\n"
        f"*Tipo:* `{alert.anomaly_type}`\n"
        f"*Severidad:* `{alert.severity}` | *Confianza ML:* `{alert.ml_confidence:.0%}`\n"
        f"*Clasificación LLM:* `{alert.llm_classification}`\n"
        f"*Modelos en acuerdo:* `{alert.models_agreed}/3`\n"
        f"*Riesgo falso positivo:* `{alert.false_positive_risk}`"
    )


def fmt_circuit_breaker(reason: str, daily_pnl_pct: float) -> str:
    return (
        f"*CIRCUIT BREAKER ACTIVADO*\n"
        f"\n"
        f"*Motivo:* `{reason}`\n"
        f"*P&L Diario:* `{daily_pnl_pct:.2%}`\n"
        f"\n"
        f"Sistema en modo CLOSE-ONLY hasta próxima sesión.\n"
        f"Usa /resume para reactivar manualmente."
    )


def fmt_system_status(portfolio, session=None) -> str:
    positions_str = ""
    for pos in portfolio.open_positions:
        positions_str += f"\n  • `{pos.asset}` {pos.direction} `${pos.current_size_usd:.0f}` P&L: `{pos.unrealized_pnl:+.2f}`"
    if not positions_str:
        positions_str = "\n  _Sin posiciones abiertas_"

    cb_str = " CIRCUIT BREAKER ACTIVO" if portfolio.circuit_breaker_active else ""
    return (
        f"*ESTADO DEL SISTEMA*{cb_str}\n"
        f"\n"
        f"*Capital:* `${portfolio.capital_usd:,.2f}`\n"
        f"*P&L Hoy:* `{portfolio.daily_pnl_usd:+.2f} USD` (`{portfolio.daily_pnl_pct:+.2%}`)\n"
        f"*Trades hoy:* `{portfolio.trades_today}` (Win: `{portfolio.winning_trades_today}`)\n"
        f"\n"
        f"*Posiciones abiertas ({len(portfolio.open_positions)}):{positions_str}"
    )
