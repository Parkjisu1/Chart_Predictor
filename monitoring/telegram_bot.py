"""Telegram bot for trade notifications and monitoring."""

from __future__ import annotations

import asyncio
from typing import Any

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)


class TelegramNotifier:
    """Send trade alerts and status updates via Telegram."""

    def __init__(self):
        settings = get_settings()
        self.token = settings.telegram.bot_token
        self.chat_id = settings.telegram.chat_id
        self.enabled = bool(self.token and self.chat_id)
        self._bot = None

    async def _get_bot(self):
        if self._bot is None and self.enabled:
            try:
                from telegram import Bot
                self._bot = Bot(token=self.token)
            except ImportError:
                logger.warning("telegram_not_installed")
                self.enabled = False
        return self._bot

    async def send_message(self, text: str) -> bool:
        """Send a text message to the configured chat."""
        if not self.enabled:
            return False

        try:
            bot = await self._get_bot()
            if bot:
                await bot.send_message(chat_id=self.chat_id, text=text,
                                       parse_mode="HTML")
                return True
        except Exception as e:
            logger.error("telegram_send_failed", error=str(e))
        return False

    async def notify_trade_opened(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        leverage: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        """Send trade open notification."""
        msg = (
            f"<b>{'🟢' if side == 'long' else '🔴'} NEW TRADE</b>\n"
            f"Symbol: {symbol}\n"
            f"Side: {side.upper()}\n"
            f"Price: ${price:,.2f}\n"
            f"Qty: {quantity:.6f}\n"
            f"Leverage: {leverage}x\n"
        )
        if stop_loss:
            msg += f"SL: ${stop_loss:,.2f}\n"
        if take_profit:
            msg += f"TP: ${take_profit:,.2f}\n"

        await self.send_message(msg)

    async def notify_trade_closed(
        self,
        symbol: str,
        side: str,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ) -> None:
        """Send trade close notification."""
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"<b>{emoji} TRADE CLOSED</b>\n"
            f"Symbol: {symbol} ({side.upper()})\n"
            f"PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%)\n"
            f"Reason: {reason}\n"
        )
        await self.send_message(msg)

    async def notify_kill_switch(self, reason: str) -> None:
        """Send kill switch alert."""
        msg = f"<b>🚨 KILL SWITCH TRIGGERED</b>\n\nReason: {reason}\n\nAll trading halted."
        await self.send_message(msg)

    async def send_daily_report(self, report: str) -> None:
        """Send daily performance report."""
        msg = f"<b>📊 Daily Report</b>\n<pre>{report}</pre>"
        await self.send_message(msg)

    def send_sync(self, text: str) -> bool:
        """Synchronous send wrapper."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.send_message(text))
                return True
            else:
                return loop.run_until_complete(self.send_message(text))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(self.send_message(text))
            loop.close()
            return result
