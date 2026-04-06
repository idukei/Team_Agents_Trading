"""
ChatAgent — Permanent dashboard chat using OpenRouter (qwen/qwen3.6-plus:free).

Completely independent from the Fireworks.ai trading agents.
Provides streaming responses with awareness of the current system state.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import structlog
from openai import AsyncOpenAI

from core.config import settings

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """\
Eres el asistente inteligente de TeamTrade, un sistema multi-agente de trading event-driven.
Respondes siempre en español de forma concisa y útil.

Tu rol:
- Explicar el estado actual del sistema de trading
- Responder preguntas sobre los agentes, señales, posiciones y rendimiento
- Ayudar a interpretar eventos de mercado, anomalías y propuestas de trade
- Dar contexto sobre las decisiones tomadas por los agentes

Sistema TeamTrade (resumen técnico):
- 9 agentes: Monitor, Supervisor, Sentiment, MarketData, Anomaly, Strategy, Risk, Execution, Notification
- Orquestado con LangGraph StateGraph con ejecución paralela via Send API
- LLMs para trading: Fireworks.ai (llama-70b, llama-8b, deepseek-r1)
- Detección de anomalías: ensemble 3 modelos (IsolationForest + Z-score adaptativo + LSTM Autoencoder)
- Human-in-the-loop: interrupt_before=["execution"] + aprobación por Telegram
- Event Priority Score (EPS): (source×0.35)+(keywords×0.30)+(novelty×0.20)+(vix×0.15)×100
- Composite Trade Score (CTS): señales ponderadas de sentiment, anomalía, mercado, precedentes, OBI
- Risk: 3 capas — trade (R/R min 1.5:1), portfolio (VaR, correlación), sistémica (circuit breaker)

Estado actual del sistema:
{context}

Si no tienes datos sobre algo, dilo claramente. No inventes información de mercado o posiciones.
"""


class ChatAgent:
    """
    Streaming chat agent backed by OpenRouter qwen/qwen3.6-plus:free.

    The model is free-tier on OpenRouter and does not consume Fireworks.ai budget.
    Each call builds a fresh client (connection pooling handled by httpx internally).
    """

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = "no-key"
            if settings.openrouter_api_key:
                api_key = settings.openrouter_api_key.get_secret_value()

            self._client = AsyncOpenAI(
                base_url=settings.openrouter_base_url,
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": f"http://localhost:{settings.dashboard_port}",
                    "X-Title": "TeamTrade Dashboard",
                },
            )
        return self._client

    def _build_context(self, system_state: dict[str, Any] | None) -> str:
        """Serialize current system state into a human-readable context string."""
        if not system_state:
            return "Sin datos de estado disponibles (sistema iniciando)."

        lines: list[str] = []

        # Portfolio
        portfolio = system_state.get("portfolio")
        if portfolio:
            lines.append(
                f"Capital: ${portfolio.get('capital_usd', 0):,.2f} | "
                f"P&L hoy: ${portfolio.get('daily_pnl_usd', 0):+.2f} "
                f"({portfolio.get('daily_pnl_pct', 0):+.2%}) | "
                f"Posiciones abiertas: {portfolio.get('open_positions_count', 0)} | "
                f"Trades hoy: {portfolio.get('trades_today', 0)} | "
                f"Win rate: {portfolio.get('win_rate_today', 0):.0%} | "
                f"Circuit breaker: {'ACTIVO ⚠️' if portfolio.get('circuit_breaker_active') else 'inactivo'}"
            )

        # Last event
        event = system_state.get("event")
        if event:
            lines.append(
                f"Último evento: [{event.get('source', '?')}] EPS={event.get('eps_score', 0):.0f} "
                f"urgencia={event.get('urgency', '?')} — \"{event.get('raw_content', '')[:150]}\""
            )

        # Current proposal
        proposal = system_state.get("proposal")
        if proposal:
            lines.append(
                f"Propuesta activa: {proposal.get('direction', '?')} {proposal.get('asset', '?')} "
                f"@ ${proposal.get('entry_price', 0):.2f} | "
                f"Tamaño: ${proposal.get('position_size_usd', 0):.0f} | "
                f"CTS: {proposal.get('cts_score', 0):.2f} | "
                f"SL: ${proposal.get('stop_loss', 0):.2f} | "
                f"TP: {proposal.get('take_profit', [])}"
            )

        # Risk decision
        risk = system_state.get("risk")
        if risk:
            lines.append(
                f"Decisión de riesgo: {risk.get('status', '?')} | "
                f"Tamaño aprobado: ${risk.get('approved_size_usd', 0):.0f} | "
                f"VaR 95%: {risk.get('var_95', 0):.2%}"
            )

        # Session info
        session = system_state.get("session")
        if session:
            lines.append(
                f"Sesión: modo={session.get('pipeline_mode', '?')} | "
                f"Coste LLM: ${session.get('fireworks_cost_usd', 0):.4f} | "
                f"Agentes degradados: {session.get('degraded_agents', []) or 'ninguno'}"
            )

        # Anomaly
        anomaly = system_state.get("anomaly")
        if anomaly:
            lines.append(
                f"Anomalía detectada: {anomaly.get('type', '?')} | "
                f"Severidad: {anomaly.get('severity', '?')} | "
                f"Confianza ML: {anomaly.get('confidence', 0):.0%} | "
                f"Modelos de acuerdo: {anomaly.get('models_agreed', 0)}/3"
            )

        return "\n".join(lines) if lines else "Sistema activo, sin pipeline en curso."

    async def stream(
        self,
        messages: list[dict[str, str]],
        system_state: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Yield streaming text chunks from OpenRouter.

        messages: list of {"role": "user"|"assistant", "content": "..."}
        system_state: current dashboard state (from StateBroadcaster._latest_state)
        """
        if not settings.openrouter_api_key:
            yield "⚠️ OPENROUTER_API_KEY no configurada. Añade la key en el archivo .env para activar el chat."
            return

        client = self._get_client()
        context = self._build_context(system_state)
        system_msg = SYSTEM_PROMPT.format(context=context)

        full_messages = [
            {"role": "system", "content": system_msg},
            *messages,
        ]

        try:
            async with client.beta.chat.completions.stream(
                model=settings.model_chat,
                messages=full_messages,
                temperature=0.7,
                max_tokens=1024,
            ) as stream:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

        except Exception as exc:
            log.warning("chat_stream_error", error=str(exc), model=settings.model_chat)
            yield f"\n\n⚠️ Error al conectar con OpenRouter: {exc}"

    async def complete(
        self,
        messages: list[dict[str, str]],
        system_state: dict[str, Any] | None = None,
    ) -> str:
        """Non-streaming version — collects full response."""
        chunks: list[str] = []
        async for chunk in self.stream(messages, system_state):
            chunks.append(chunk)
        return "".join(chunks)


# Global singleton
chat_agent = ChatAgent()
