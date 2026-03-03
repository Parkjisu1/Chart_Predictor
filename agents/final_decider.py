"""Final Decider Agent - Modified Half-Kelly position sizing."""

from __future__ import annotations

import time
from dataclasses import dataclass

from agents.base import AgentBase, AgentResult
from config.constants import Signal, Side


@dataclass
class TradeDecision:
    execute: bool
    side: str | None
    position_size_pct: float
    leverage: int
    stop_loss_pct: float
    take_profit_pct: float
    confidence: float
    reasoning: str


class FinalDeciderAgent(AgentBase):
    """Agent 4: Makes final trade decision with position sizing.

    Uses Modified Half-Kelly criterion for position sizing.
    """

    def __init__(self, max_leverage: int = 5):
        super().__init__("final_decider")
        self.max_leverage = max_leverage

    def analyze(
        self,
        risk_result: AgentResult | None = None,
        current_capital: float = 100_000,
        current_price: float = 0,
        historical_win_rate: float = 0.5,
        avg_win_loss_ratio: float = 1.5,
        **kwargs,
    ) -> AgentResult:
        """Make final trade decision."""
        start = time.time()

        if risk_result is None or not risk_result.details.get("approved", False):
            return self._make_result(
                reasoning="Trade rejected by risk reviewer",
                details={"execute": False},
                execution_time=time.time() - start,
            )

        signal_value = risk_result.signal_value
        confidence = risk_result.confidence
        risk_details = risk_result.details

        # Determine side
        side = Side.LONG if signal_value > 0 else Side.SHORT

        # Modified Half-Kelly position sizing
        kelly_pct = self._half_kelly(
            historical_win_rate, avg_win_loss_ratio, confidence
        )

        # Cap by risk reviewer's max position
        max_pos = risk_details.get("max_position_pct", 0.25)
        position_pct = min(kelly_pct, max_pos)

        # Leverage based on confidence
        leverage = self._determine_leverage(confidence, abs(signal_value))

        # Adjust stops from risk reviewer
        stop_loss_pct = risk_details.get("stop_loss_pct", 0.02)
        take_profit_pct = risk_details.get("take_profit_pct", 0.03)

        # Minimum position filter
        min_position_value = 10  # $10 minimum
        position_value = current_capital * position_pct * leverage
        execute = position_value >= min_position_value and position_pct > 0.01

        decision = TradeDecision(
            execute=execute,
            side=side.value,
            position_size_pct=round(position_pct, 4),
            leverage=leverage,
            stop_loss_pct=round(stop_loss_pct, 4),
            take_profit_pct=round(take_profit_pct, 4),
            confidence=round(confidence, 4),
            reasoning=f"{side.value.upper()} {position_pct:.1%} @ {leverage}x "
                      f"(Kelly={kelly_pct:.1%})",
        )

        elapsed = time.time() - start
        return self._make_result(
            signal_value=signal_value,
            confidence=confidence,
            reasoning=decision.reasoning,
            details={
                "execute": decision.execute,
                "side": decision.side,
                "position_size_pct": decision.position_size_pct,
                "leverage": decision.leverage,
                "stop_loss_pct": decision.stop_loss_pct,
                "take_profit_pct": decision.take_profit_pct,
                "kelly_raw": round(kelly_pct, 4),
                "position_value": round(position_value, 2),
            },
            execution_time=elapsed,
        )

    def _half_kelly(
        self,
        win_rate: float,
        win_loss_ratio: float,
        confidence: float,
    ) -> float:
        """Modified Half-Kelly criterion.

        Kelly% = (p * b - q) / b
        Half-Kelly = Kelly% / 2  (more conservative)
        Modified = Half-Kelly * confidence
        """
        p = win_rate
        q = 1 - p
        b = win_loss_ratio

        if b <= 0:
            return 0.0

        kelly = (p * b - q) / b
        if kelly <= 0:
            return 0.0

        half_kelly = kelly / 2
        modified = half_kelly * confidence
        return max(0.0, min(modified, 0.25))  # Hard cap at 25%

    def _determine_leverage(self, confidence: float, signal_strength: float) -> int:
        """Determine leverage based on confidence and signal strength."""
        combined = confidence * signal_strength
        if combined > 0.7:
            lev = min(self.max_leverage, 5)
        elif combined > 0.5:
            lev = min(self.max_leverage, 3)
        elif combined > 0.3:
            lev = min(self.max_leverage, 2)
        else:
            lev = 1
        return lev
