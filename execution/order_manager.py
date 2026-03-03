"""Order management using ccxt for Bybit futures."""

from __future__ import annotations

import json
import time

import ccxt

from config.logging_config import get_logger
from config.settings import get_settings
from config.constants import Side, OrderType
from execution.rate_limiter import RateLimiter
from data.database import Database
from data.models import Trade

logger = get_logger(__name__)


class OrderManager:
    """Manage order placement, modification, and cancellation."""

    def __init__(self, db: Database | None = None):
        settings = get_settings()
        self.exchange = ccxt.bybit({
            "apiKey": settings.bybit.api_key,
            "secret": settings.bybit.api_secret,
            "options": {"defaultType": "linear"},
            "enableRateLimit": True,
        })
        if settings.bybit.testnet:
            self.exchange.set_sandbox_mode(True)

        self.db = db or Database(settings.db_path)
        self.rate_limiter = RateLimiter()

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        leverage: int = 1,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict | None:
        """Place a market order with optional SL/TP."""
        if not self.rate_limiter.acquire():
            logger.warning("rate_limit_exceeded")
            return None

        try:
            # Set leverage
            self.exchange.set_leverage(leverage, symbol)

            params = {}
            if stop_loss:
                params["stopLoss"] = {"triggerPrice": stop_loss}
            if take_profit:
                params["takeProfit"] = {"triggerPrice": take_profit}

            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side="buy" if side == Side.LONG.value else "sell",
                amount=quantity,
                params=params,
            )

            logger.info(
                "order_placed",
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_id=order.get("id"),
            )

            # Record trade
            trade = Trade(
                symbol=symbol,
                side=side,
                entry_price=float(order.get("average", order.get("price", 0))),
                quantity=quantity,
                leverage=leverage,
                entry_time=order.get("datetime", ""),
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            self.db.insert_trade(trade)

            return order

        except ccxt.InsufficientFunds as e:
            logger.error("insufficient_funds", error=str(e))
        except ccxt.InvalidOrder as e:
            logger.error("invalid_order", error=str(e))
        except Exception as e:
            logger.error("order_failed", error=str(e))

        return None

    def close_position(self, symbol: str, side: str, quantity: float) -> dict | None:
        """Close an existing position."""
        if not self.rate_limiter.acquire():
            return None

        try:
            close_side = "sell" if side == Side.LONG.value else "buy"
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=close_side,
                amount=quantity,
                params={"reduceOnly": True},
            )

            logger.info("position_closed", symbol=symbol, order_id=order.get("id"))
            return order

        except Exception as e:
            logger.error("close_failed", error=str(e))
            return None

    def get_open_positions(self) -> list[dict]:
        """Fetch current open positions from exchange."""
        if not self.rate_limiter.acquire():
            return []

        try:
            positions = self.exchange.fetch_positions()
            open_positions = [
                {
                    "symbol": p["symbol"],
                    "side": p["side"],
                    "size": float(p.get("contracts", 0)),
                    "entry_price": float(p.get("entryPrice", 0)),
                    "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
                    "leverage": int(p.get("leverage", 1)),
                    "liquidation_price": float(p.get("liquidationPrice", 0)),
                }
                for p in positions
                if float(p.get("contracts", 0)) > 0
            ]
            return open_positions

        except Exception as e:
            logger.error("fetch_positions_failed", error=str(e))
            return []

    def get_balance(self) -> dict:
        """Get account balance."""
        if not self.rate_limiter.acquire():
            return {}

        try:
            balance = self.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            return {
                "total": float(usdt.get("total", 0)),
                "free": float(usdt.get("free", 0)),
                "used": float(usdt.get("used", 0)),
            }
        except Exception as e:
            logger.error("balance_fetch_failed", error=str(e))
            return {}
