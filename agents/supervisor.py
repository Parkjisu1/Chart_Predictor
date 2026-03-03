"""Supervisor Agent - Daily review and kill switch authority."""

from __future__ import annotations

import json
import time

from agents.base import AgentBase, AgentResult, ClaudeCLIRunner
from config.logging_config import get_logger
from risk.kill_switch import KillSwitch
from strategy.sentiment.macro_prompts import build_daily_review_prompt

logger = get_logger(__name__)


class SupervisorAgent(AgentBase):
    """Agent 5: Supervisor with kill switch authority.

    Performs daily reviews, monitors risk levels, and can halt trading.
    Uses Claude CLI for daily analysis.
    """

    def __init__(self, kill_switch: KillSwitch | None = None, use_claude: bool = True):
        super().__init__("supervisor")
        self.kill_switch = kill_switch or KillSwitch()
        self.claude = ClaudeCLIRunner() if use_claude else None

    def analyze(
        self,
        symbol: str = "BTC/USDT",
        daily_pnl: float = 0.0,
        win_rate: float = 0.5,
        total_trades: int = 0,
        max_drawdown: float = 0.0,
        open_positions: list[dict] | None = None,
        equity: float = 0.0,
        initial_capital: float = 100_000,
        **kwargs,
    ) -> AgentResult:
        """Run daily supervisor review."""
        start = time.time()

        # 1. Check kill switch conditions
        drawdown_pct = max_drawdown
        daily_pnl_pct = daily_pnl / initial_capital if initial_capital > 0 else 0

        ks_status = self.kill_switch.check(
            daily_pnl_pct=daily_pnl_pct,
            total_drawdown_pct=drawdown_pct,
            equity=equity,
            initial_capital=initial_capital,
        )

        if ks_status.triggered:
            return self._make_result(
                signal_value=0.0,
                confidence=1.0,
                reasoning=f"KILL SWITCH: {ks_status.reason}",
                details={
                    "kill_switch": True,
                    "reason": ks_status.reason,
                    "should_continue": False,
                },
                execution_time=time.time() - start,
            )

        # 2. Claude CLI daily review
        claude_review = None
        if self.claude:
            claude_review = self._run_claude_review(
                symbol, daily_pnl_pct, win_rate, total_trades,
                max_drawdown, open_positions or [],
            )

        # 3. Compile assessment
        should_continue = True
        risk_level = "low"

        if claude_review:
            should_continue = claude_review.get("should_continue", True)
            risk_level = claude_review.get("risk_level", "low")
            if claude_review.get("kill_switch", False):
                self.kill_switch._trigger(
                    claude_review.get("kill_reason", "Claude supervisor recommendation"),
                    {"claude_review": claude_review},
                )
                should_continue = False

        elapsed = time.time() - start
        return self._make_result(
            signal_value=0.0,
            confidence=0.8,
            reasoning=f"Daily review: risk={risk_level}, continue={should_continue}",
            details={
                "kill_switch": not should_continue,
                "should_continue": should_continue,
                "risk_level": risk_level,
                "claude_review": claude_review,
                "daily_pnl_pct": round(daily_pnl_pct, 4),
                "max_drawdown": round(max_drawdown, 4),
            },
            execution_time=elapsed,
        )

    def _run_claude_review(
        self,
        symbol: str,
        daily_pnl: float,
        win_rate: float,
        total_trades: int,
        max_drawdown: float,
        open_positions: list[dict],
    ) -> dict | None:
        """Get Claude's daily review."""
        try:
            prompt = build_daily_review_prompt(
                symbol=symbol,
                daily_pnl=daily_pnl * 100,
                win_rate=win_rate * 100,
                total_trades=total_trades,
                max_drawdown=max_drawdown * 100,
                open_positions=open_positions,
            )
            return self.claude.run(prompt)
        except Exception as e:
            self.logger.warning("claude_review_failed", error=str(e))
            return None

    def force_kill(self, reason: str) -> None:
        """Manually trigger kill switch."""
        self.kill_switch._trigger(reason, {"source": "manual"})
        logger.critical("manual_kill_switch", reason=reason)
