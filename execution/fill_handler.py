"""Handle order fills and update positions."""

from __future__ import annotations

from config.logging_config import get_logger
from execution.position_tracker import PositionTracker
from data.database import Database
from data.models import Trade

logger = get_logger(__name__)


class FillHandler:
    """Process order fills and update system state."""

    def __init__(self, tracker: PositionTracker, db: Database):
        self.tracker = tracker
        self.db = db

    def handle_fill(self, fill: dict) -> None:
        """Process an order fill event."""
        symbol = fill.get("symbol", "")
        side = fill.get("side", "")
        price = float(fill.get("price", 0))
        quantity = float(fill.get("amount", 0))
        order_type = fill.get("type", "market")
        reduce_only = fill.get("reduceOnly", False)

        logger.info("fill_received", symbol=symbol, side=side,
                     price=price, quantity=quantity)

        if reduce_only:
            # This is a position close
            pos = self.tracker.remove_position(symbol)
            if pos:
                if pos.side == "long":
                    pnl = (price - pos.entry_price) * pos.quantity
                else:
                    pnl = (pos.entry_price - price) * pos.quantity

                logger.info("position_closed_fill", symbol=symbol,
                           pnl=round(pnl, 2))
        else:
            # New position
            self.tracker.add_position(
                symbol=symbol,
                side=side,
                entry_price=price,
                quantity=quantity,
            )
