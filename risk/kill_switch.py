"""Kill switch for emergency position closure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from config.constants import MAX_DAILY_LOSS_PCT, MAX_DRAWDOWN_PCT
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class KillSwitchStatus:
    triggered: bool
    reason: str
    details: dict


class KillSwitch:
    """Emergency kill switch that halts all trading."""

    def __init__(
        self,
        max_daily_loss: float = MAX_DAILY_LOSS_PCT,
        max_drawdown: float = MAX_DRAWDOWN_PCT,
    ):
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.is_active = False
        self.triggered_at: str | None = None
        self.trigger_reason: str | None = None

    def check(
        self,
        daily_pnl_pct: float,
        total_drawdown_pct: float,
        equity: float,
        initial_capital: float,
    ) -> KillSwitchStatus:
        """Check if kill switch should be triggered."""
        if self.is_active:
            return KillSwitchStatus(
                triggered=True,
                reason=self.trigger_reason or "Previously triggered",
                details={"triggered_at": self.triggered_at},
            )

        # Check daily loss
        if daily_pnl_pct < -self.max_daily_loss:
            return self._trigger(
                f"Daily loss {daily_pnl_pct:.2%} exceeds limit {-self.max_daily_loss:.2%}",
                {"daily_pnl": daily_pnl_pct, "limit": self.max_daily_loss},
            )

        # Check drawdown
        if total_drawdown_pct > self.max_drawdown:
            return self._trigger(
                f"Drawdown {total_drawdown_pct:.2%} exceeds limit {self.max_drawdown:.2%}",
                {"drawdown": total_drawdown_pct, "limit": self.max_drawdown},
            )

        # Check absolute ruin (>50% loss)
        if equity < initial_capital * 0.5:
            return self._trigger(
                f"Equity {equity:.2f} below 50% of initial capital",
                {"equity": equity, "initial": initial_capital},
            )

        return KillSwitchStatus(triggered=False, reason="", details={})

    def _trigger(self, reason: str, details: dict) -> KillSwitchStatus:
        self.is_active = True
        self.triggered_at = datetime.now().isoformat()
        self.trigger_reason = reason
        logger.critical("kill_switch_triggered", reason=reason, **details)
        return KillSwitchStatus(triggered=True, reason=reason, details=details)

    def reset(self) -> None:
        """Manually reset kill switch (requires human intervention)."""
        self.is_active = False
        self.triggered_at = None
        self.trigger_reason = None
        logger.warning("kill_switch_reset")
