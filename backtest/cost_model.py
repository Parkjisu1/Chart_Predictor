"""Realistic cost model: fees + slippage + funding."""

from __future__ import annotations

from config.constants import MAKER_FEE, TAKER_FEE, FUNDING_INTERVAL_HOURS
from risk.slippage import SlippageModel


class CostModel:
    """Calculate all trading costs for backtesting accuracy."""

    def __init__(
        self,
        maker_fee: float = MAKER_FEE,
        taker_fee: float = TAKER_FEE,
        use_market_orders: bool = True,
        slippage_model: SlippageModel | None = None,
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.use_market_orders = use_market_orders
        self.slippage = slippage_model or SlippageModel()

    def calculate_entry_cost(
        self,
        notional_value: float,
        avg_volume_24h: float = 1e9,
    ) -> dict:
        """Calculate costs for opening a position."""
        fee_rate = self.taker_fee if self.use_market_orders else self.maker_fee
        fee = notional_value * fee_rate

        slippage_bps = self.slippage.estimate(
            notional_value, avg_volume_24h, is_market_order=self.use_market_orders
        )
        slippage_cost = notional_value * slippage_bps / 10000

        return {
            "fee": round(fee, 4),
            "slippage": round(slippage_cost, 4),
            "total": round(fee + slippage_cost, 4),
            "fee_rate": fee_rate,
            "slippage_bps": slippage_bps,
        }

    def calculate_exit_cost(
        self,
        notional_value: float,
        avg_volume_24h: float = 1e9,
    ) -> dict:
        """Calculate costs for closing a position."""
        return self.calculate_entry_cost(notional_value, avg_volume_24h)

    def calculate_funding_cost(
        self,
        notional_value: float,
        funding_rate: float,
        hours_held: float,
    ) -> float:
        """Calculate cumulative funding costs."""
        funding_periods = hours_held / FUNDING_INTERVAL_HOURS
        total_funding = notional_value * funding_rate * funding_periods
        return round(total_funding, 4)

    def calculate_total_round_trip(
        self,
        entry_notional: float,
        exit_notional: float,
        funding_rate: float = 0.0001,
        hours_held: float = 8,
        avg_volume_24h: float = 1e9,
    ) -> dict:
        """Calculate total round-trip cost."""
        entry_costs = self.calculate_entry_cost(entry_notional, avg_volume_24h)
        exit_costs = self.calculate_exit_cost(exit_notional, avg_volume_24h)
        funding = self.calculate_funding_cost(entry_notional, funding_rate, hours_held)

        total = entry_costs["total"] + exit_costs["total"] + abs(funding)
        return {
            "entry_costs": entry_costs,
            "exit_costs": exit_costs,
            "funding_cost": funding,
            "total_cost": round(total, 4),
            "cost_pct": round(total / entry_notional * 100, 4) if entry_notional > 0 else 0,
        }
