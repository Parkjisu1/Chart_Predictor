"""Trading limits and constraint checks."""

from __future__ import annotations

from dataclasses import dataclass

from config.constants import (
    MAX_POSITION_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MAX_DAILY_LOSS_PCT,
    MAX_DRAWDOWN_PCT,
)


@dataclass
class LimitCheck:
    passed: bool
    violations: list[str]
    details: dict


class TradingLimits:
    """Enforce all trading limits and constraints."""

    def __init__(
        self,
        max_position_pct: float = MAX_POSITION_PCT,
        max_exposure_pct: float = MAX_TOTAL_EXPOSURE_PCT,
        max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT,
        max_drawdown_pct: float = MAX_DRAWDOWN_PCT,
        max_open_positions: int = 4,
        max_leverage: int = 5,
    ):
        self.max_position_pct = max_position_pct
        self.max_exposure_pct = max_exposure_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_open_positions = max_open_positions
        self.max_leverage = max_leverage

    def check_new_trade(
        self,
        position_pct: float,
        current_exposure_pct: float,
        daily_pnl_pct: float,
        drawdown_pct: float,
        open_positions: int,
        leverage: int,
    ) -> LimitCheck:
        """Check all limits before opening a new trade."""
        violations = []
        details = {}

        if position_pct > self.max_position_pct:
            violations.append(
                f"Position {position_pct:.1%} exceeds max {self.max_position_pct:.1%}"
            )

        new_exposure = current_exposure_pct + position_pct
        if new_exposure > self.max_exposure_pct:
            violations.append(
                f"Total exposure {new_exposure:.1%} exceeds max {self.max_exposure_pct:.1%}"
            )

        if daily_pnl_pct < -self.max_daily_loss_pct:
            violations.append(
                f"Daily loss {daily_pnl_pct:.1%} exceeds limit"
            )

        if drawdown_pct > self.max_drawdown_pct:
            violations.append(
                f"Drawdown {drawdown_pct:.1%} exceeds limit"
            )

        if open_positions >= self.max_open_positions:
            violations.append(
                f"Max open positions reached ({open_positions})"
            )

        if leverage > self.max_leverage:
            violations.append(
                f"Leverage {leverage}x exceeds max {self.max_leverage}x"
            )

        details = {
            "position_pct": position_pct,
            "new_exposure": new_exposure,
            "daily_pnl_pct": daily_pnl_pct,
            "drawdown_pct": drawdown_pct,
            "open_positions": open_positions,
            "leverage": leverage,
        }

        return LimitCheck(
            passed=len(violations) == 0,
            violations=violations,
            details=details,
        )
