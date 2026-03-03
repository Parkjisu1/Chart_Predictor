"""Slippage estimation model."""

from __future__ import annotations

import numpy as np


class SlippageModel:
    """Estimate expected slippage based on order size and market conditions."""

    def __init__(
        self,
        base_slippage_bps: float = 1.0,  # 0.01% base
        volume_impact_factor: float = 0.5,
    ):
        self.base_slippage_bps = base_slippage_bps
        self.volume_impact_factor = volume_impact_factor

    def estimate(
        self,
        order_value: float,
        avg_volume_24h: float,
        current_spread_bps: float = 1.0,
        is_market_order: bool = True,
    ) -> float:
        """Estimate slippage in basis points.

        Args:
            order_value: Order size in USD
            avg_volume_24h: Average 24h volume in USD
            current_spread_bps: Current bid-ask spread in bps
            is_market_order: True for market orders

        Returns:
            Estimated slippage in basis points
        """
        if avg_volume_24h <= 0:
            return self.base_slippage_bps * 5  # Assume high slippage

        # Volume impact: larger orders relative to volume = more slippage
        volume_ratio = order_value / avg_volume_24h
        volume_impact = self.volume_impact_factor * np.sqrt(volume_ratio) * 10000

        # Spread impact for market orders
        spread_impact = current_spread_bps / 2 if is_market_order else 0

        total = self.base_slippage_bps + volume_impact + spread_impact
        return round(total, 2)

    def estimate_cost(
        self,
        order_value: float,
        avg_volume_24h: float,
        current_spread_bps: float = 1.0,
    ) -> float:
        """Estimate slippage cost in USD."""
        slippage_bps = self.estimate(order_value, avg_volume_24h, current_spread_bps)
        return round(order_value * slippage_bps / 10000, 4)
