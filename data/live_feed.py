"""Bybit WebSocket live data feed."""

from __future__ import annotations

import asyncio
import json
from typing import Callable

import websockets

from config.logging_config import get_logger

logger = get_logger(__name__)

BYBIT_WS_PUBLIC = "wss://stream.bybit.com/v5/public/linear"
BYBIT_WS_TESTNET = "wss://stream-testnet.bybit.com/v5/public/linear"


class LiveFeed:
    """WebSocket connection to Bybit for real-time data."""

    def __init__(self, testnet: bool = True):
        self.url = BYBIT_WS_TESTNET if testnet else BYBIT_WS_PUBLIC
        self.callbacks: dict[str, list[Callable]] = {}
        self._running = False
        self._ws = None

    def on_kline(self, callback: Callable) -> None:
        """Register callback for kline/candle updates."""
        self.callbacks.setdefault("kline", []).append(callback)

    def on_ticker(self, callback: Callable) -> None:
        """Register callback for ticker updates."""
        self.callbacks.setdefault("ticker", []).append(callback)

    def on_trade(self, callback: Callable) -> None:
        """Register callback for trade updates."""
        self.callbacks.setdefault("trade", []).append(callback)

    async def start(self, symbols: list[str], interval: str = "15") -> None:
        """Start WebSocket connection and subscribe to channels."""
        self._running = True

        while self._running:
            try:
                async with websockets.connect(self.url, ping_interval=20) as ws:
                    self._ws = ws
                    logger.info("websocket_connected", url=self.url)

                    # Subscribe to channels
                    subs = []
                    for symbol in symbols:
                        clean = symbol.replace("/", "").replace(":USDT", "")
                        subs.extend([
                            f"kline.{interval}.{clean}",
                            f"tickers.{clean}",
                        ])

                    subscribe_msg = {
                        "op": "subscribe",
                        "args": subs,
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info("subscribed", channels=subs)

                    # Process messages
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            await self._dispatch(data)
                        except json.JSONDecodeError:
                            continue

            except websockets.exceptions.ConnectionClosed:
                logger.warning("websocket_disconnected, reconnecting...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error("websocket_error", error=str(e))
                await asyncio.sleep(10)

    async def _dispatch(self, data: dict) -> None:
        """Dispatch message to registered callbacks."""
        topic = data.get("topic", "")

        if topic.startswith("kline"):
            for cb in self.callbacks.get("kline", []):
                cb(data.get("data", []))

        elif topic.startswith("tickers"):
            for cb in self.callbacks.get("ticker", []):
                cb(data.get("data", {}))

    async def stop(self) -> None:
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            logger.info("websocket_stopped")
