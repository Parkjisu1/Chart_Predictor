"""Position sizing with risk-based constraints."""

from __future__ import annotations

from dataclasses import dataclass

from config.constants import MAX_POSITION_PCT, MAX_TOTAL_EXPOSURE_PCT


@dataclass
class PositionSize:
    size_pct: float       # % of capital
    size_value: float     # $ value
    quantity: float       # Coin quantity
    leverage: int
    margin_required: float


class PositionSizer:
    """Calculate position sizes respecting risk limits."""

    def __init__(
        self,
        max_position_pct: float = MAX_POSITION_PCT,
        max_total_exposure_pct: float = MAX_TOTAL_EXPOSURE_PCT,
    ):
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure_pct

    def calculate(
        self,
        capital: float,
        price: float,
        signal_confidence: float,
        kelly_pct: float,
        leverage: int = 1,
        current_exposure_pct: float = 0.0,
    ) -> PositionSize:
        """Calculate position size."""
        # Cap position by constraints
        available_exposure = max(0, self.max_total_exposure - current_exposure_pct)
        position_pct = min(kelly_pct, self.max_position_pct, available_exposure)

        if position_pct <= 0 or price <= 0:
            return PositionSize(0, 0, 0, 1, 0)

        size_value = capital * position_pct
        margin_required = size_value  # Margin = position / leverage, but we size by capital %
        notional = size_value * leverage
        quantity = notional / price

        return PositionSize(
            size_pct=round(position_pct, 4),
            size_value=round(size_value, 2),
            quantity=round(quantity, 8),
            leverage=leverage,
            margin_required=round(margin_required, 2),
        )

    def adjust_for_volatility(
        self,
        base_size: PositionSize,
        current_vol: float,
        avg_vol: float,
    ) -> PositionSize:
        """Reduce position size in high-volatility regimes."""
        if avg_vol <= 0:
            return base_size

        vol_ratio = current_vol / avg_vol
        if vol_ratio > 1.5:
            adjustment = 1.5 / vol_ratio
            return PositionSize(
                size_pct=round(base_size.size_pct * adjustment, 4),
                size_value=round(base_size.size_value * adjustment, 2),
                quantity=round(base_size.quantity * adjustment, 8),
                leverage=base_size.leverage,
                margin_required=round(base_size.margin_required * adjustment, 2),
            )
        return base_size
