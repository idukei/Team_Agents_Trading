from __future__ import annotations

import asyncio
from enum import StrEnum

import structlog
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.ext import Application

from core.config import settings

log = structlog.get_logger(__name__)


class ApprovalResult(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"


class ApprovalHandler:
    """
    Manages human-in-the-loop approval flow via Telegram inline buttons.

    Usage:
        result = await handler.request_approval(trade_id, message_text)
        # returns APPROVED / REJECTED / TIMEOUT
    """

    def __init__(self, app: Application) -> None:
        self._app = app
        self._pending: dict[str, asyncio.Event] = {}
        self._decisions: dict[str, ApprovalResult] = {}

    async def request_approval(
        self,
        trade_id: str,
        message_text: str,
        timeout_s: int | None = None,
    ) -> ApprovalResult:
        timeout = timeout_s or settings.telegram_approval_timeout_s

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("APROBAR", callback_data=f"approve:{trade_id}"),
                InlineKeyboardButton("CANCELAR", callback_data=f"reject:{trade_id}"),
            ]
        ])

        await self._app.bot.send_message(
            chat_id=settings.telegram_chat_id,
            text=message_text + f"\n\n_Respondiendo automáticamente en {timeout}s..._",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )

        event = asyncio.Event()
        self._pending[trade_id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            result = self._decisions.pop(trade_id, ApprovalResult.TIMEOUT)
        except asyncio.TimeoutError:
            result = ApprovalResult.TIMEOUT
        finally:
            self._pending.pop(trade_id, None)

        log.info("approval_result", trade_id=trade_id, result=result)
        return result

    def handle_callback(self, trade_id: str, approved: bool) -> None:
        """Called by Telegram callback query handler when user taps button."""
        self._decisions[trade_id] = ApprovalResult.APPROVED if approved else ApprovalResult.REJECTED
        event = self._pending.get(trade_id)
        if event:
            event.set()
