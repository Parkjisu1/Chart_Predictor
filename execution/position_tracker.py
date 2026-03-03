"""Track open positions and their P&L."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from config.logging_config import get_logger
from config.constants import Side

logger = get_logger(__name__)


@dataclass
class TrackedPosition:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    leverage: int
    entry_time: str
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop: float | None = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    peak_pnl_pct: float = 0.0


class PositionTracker:
    """Track and manage open positions."""

    def __init__(self):
        self.positions: dict[str, TrackedPosition] = {}

    def add_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        leverage: int = 1,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> TrackedPosition:
        """Add a new tracked position."""
        pos = TrackedPosition(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            leverage=leverage,
            entry_time=datetime.now().isoformat(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=entry_price,
        )
        self.positions[symbol] = pos
        logger.info("position_tracked", symbol=symbol, side=side,
                     price=entry_price)
        return pos

    def update_price(self, symbol: str, current_price: float) -> TrackedPosition | None:
        """Update position with current market price."""
        pos = self.positions.get(symbol)
        if not pos:
            return None

        pos.current_price = current_price

        # Calculate unrealized P&L
        if pos.side == Side.LONG.value:
            pnl = (current_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - current_price) * pos.quantity

        notional = pos.entry_price * pos.quantity
        pos.unrealized_pnl = pnl
        pos.unrealized_pnl_pct = pnl / notional if notional > 0 else 0

        # Track peak P&L for trailing stop
        if pos.unrealized_pnl_pct > pos.peak_pnl_pct:
            pos.peak_pnl_pct = pos.unrealized_pnl_pct

        return pos

    def check_exits(self, symbol: str) -> str | None:
        """Check if position should be exited. Returns reason or None."""
        pos = self.positions.get(symbol)
        if not pos:
            return None

        price = pos.current_price

        # Stop loss
        if pos.stop_loss:
            if pos.side == Side.LONG.value and price <= pos.stop_loss:
                return "stop_loss"
            if pos.side == Side.SHORT.value and price >= pos.stop_loss:
                return "stop_loss"

        # Take profit
        if pos.take_profit:
            if pos.side == Side.LONG.value and price >= pos.take_profit:
                return "take_profit"
            if pos.side == Side.SHORT.value and price <= pos.take_profit:
                return "take_profit"

        # Trailing stop
        if pos.trailing_stop and pos.peak_pnl_pct > 0.02:
            drawback = pos.peak_pnl_pct - pos.unrealized_pnl_pct
            if drawback > pos.trailing_stop:
                return "trailing_stop"

        return None

    def remove_position(self, symbol: str) -> TrackedPosition | None:
        return self.positions.pop(symbol, None)

    def get_total_exposure(self, capital: float) -> float:
        """Get total exposure as percentage of capital."""
        total = sum(
            p.entry_price * p.quantity / capital
            for p in self.positions.values()
        )
        return total

    def get_all_positions(self) -> list[dict]:
        return [
            {
                "symbol": p.symbol,
                "side": p.side,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "pnl": round(p.unrealized_pnl, 2),
                "pnl_pct": round(p.unrealized_pnl_pct * 100, 2),
            }
            for p in self.positions.values()
        ]
