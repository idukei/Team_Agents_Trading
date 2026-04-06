from __future__ import annotations

import asyncio

import structlog
from telegram import Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from agents.notification.approval_handler import ApprovalHandler
from core.config import settings

log = structlog.get_logger(__name__)

# Portfolio reference injected at startup for /status and /portfolio commands
_portfolio_ref = None


def set_portfolio_ref(portfolio) -> None:
    global _portfolio_ref
    _portfolio_ref = portfolio


async def _cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    from agents.notification.message_templates import fmt_system_status
    if _portfolio_ref is None:
        await update.message.reply_text("Sistema iniciando...")
        return
    text = fmt_system_status(_portfolio_ref)
    await update.message.reply_text(text, parse_mode="Markdown")


async def _cmd_portfolio(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _cmd_status(update, ctx)


async def _cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    # Signal pause via shared state — set circuit_breaker_active temporarily
    if _portfolio_ref:
        _portfolio_ref.circuit_breaker_active = True
        _portfolio_ref.circuit_breaker_reason = "Manual pause via Telegram"
    await update.message.reply_text("Sistema pausado. Usa /resume para reactivar.", parse_mode="Markdown")
    log.info("system_paused_via_telegram")


async def _cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if _portfolio_ref:
        _portfolio_ref.circuit_breaker_active = False
        _portfolio_ref.circuit_breaker_reason = None
    await update.message.reply_text("Sistema reanudado.", parse_mode="Markdown")
    log.info("system_resumed_via_telegram")


async def _cmd_close_all(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*KILL SWITCH activado* — Cerrando todas las posiciones...",
        parse_mode="Markdown",
    )
    log.warning("kill_switch_activated_via_telegram")
    # Actual close logic is handled by ExecutionAgent listening to this flag
    if _portfolio_ref:
        _portfolio_ref.circuit_breaker_active = True
        _portfolio_ref.circuit_breaker_reason = "KILL_SWITCH"


async def _cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "*TeamTrade — Comandos disponibles*\n\n"
        "/status — Estado del sistema y posiciones abiertas\n"
        "/portfolio — Resumen de portfolio y P\\&L del día\n"
        "/pause — Pausar sistema \\(no nuevas entradas\\)\n"
        "/resume — Reanudar operación normal\n"
        "/close\\_all — Cerrar TODAS las posiciones \\(emergencia\\)\n"
        "/help — Mostrar esta ayuda"
    )
    await update.message.reply_text(text, parse_mode="MarkdownV2")


async def _handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if ":" not in data:
        return

    action, trade_id = data.split(":", 1)
    approved = action == "approve"

    handler: ApprovalHandler | None = ctx.bot_data.get("approval_handler")
    if handler:
        handler.handle_callback(trade_id, approved)

    label = "APROBADO" if approved else "CANCELADO"
    await query.edit_message_text(
        f"{query.message.text}\n\n*Decisión: {label}*",
        parse_mode="Markdown",
    )


def build_application() -> tuple[Application, ApprovalHandler]:
    app = (
        Application.builder()
        .token(settings.telegram_bot_token.get_secret_value())
        .build()
    )

    approval_handler = ApprovalHandler(app)
    app.bot_data["approval_handler"] = approval_handler

    app.add_handler(CommandHandler("status", _cmd_status))
    app.add_handler(CommandHandler("portfolio", _cmd_portfolio))
    app.add_handler(CommandHandler("pause", _cmd_pause))
    app.add_handler(CommandHandler("resume", _cmd_resume))
    app.add_handler(CommandHandler("close_all", _cmd_close_all))
    app.add_handler(CommandHandler("help", _cmd_help))
    app.add_handler(CallbackQueryHandler(_handle_callback))

    return app, approval_handler
