"""Risk Reviewer Agent - Decision matrix cross-validation."""

from __future__ import annotations

import time
from dataclasses import dataclass

from agents.base import AgentBase, AgentResult
from config.constants import Signal, MAX_POSITION_PCT, MAX_DRAWDOWN_PCT


@dataclass
class RiskAssessment:
    approved: bool
    risk_score: float  # 0 (safe) to 1 (dangerous)
    max_position_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    reasons: list[str]


class RiskReviewerAgent(AgentBase):
    """Agent 3: Cross-validates signals against risk constraints.

    Applies a decision matrix to ensure trades meet risk criteria
    before passing to the final decider.
    """

    def __init__(self):
        super().__init__("risk_reviewer")

    def analyze(
        self,
        technical_result: AgentResult | None = None,
        sentiment_result: AgentResult | None = None,
        current_drawdown: float = 0.0,
        open_positions: int = 0,
        daily_pnl: float = 0.0,
        correlation_risk: float = 0.0,
        **kwargs,
    ) -> AgentResult:
        """Review and validate trading signals against risk criteria."""
        start = time.time()

        if technical_result is None:
            return self._make_result(
                error="No technical analysis provided",
                confidence=0.0,
            )

        checks = []
        risk_score = 0.0

        # Check 1: Signal strength
        signal_strength = abs(technical_result.signal_value)
        if signal_strength < 0.1:
            checks.append("Signal too weak (< 0.1)")
            risk_score += 0.3

        # Check 2: Confidence threshold
        if technical_result.confidence < 0.3:
            checks.append(f"Low confidence ({technical_result.confidence:.2f})")
            risk_score += 0.2

        # Check 3: Drawdown check
        if current_drawdown > MAX_DRAWDOWN_PCT:
            checks.append(f"Drawdown limit exceeded ({current_drawdown:.1%})")
            risk_score += 0.5

        # Check 4: Daily loss check
        if daily_pnl < -0.03:
            checks.append(f"Daily loss high ({daily_pnl:.1%})")
            risk_score += 0.3

        # Check 5: Position concentration
        if open_positions >= 3:
            checks.append(f"Too many open positions ({open_positions})")
            risk_score += 0.2

        # Check 6: Correlation risk
        if correlation_risk > 0.7:
            checks.append(f"High correlation risk ({correlation_risk:.2f})")
            risk_score += 0.2

        # Check 7: Signal agreement (tech vs sentiment)
        if sentiment_result and sentiment_result.confidence > 0.3:
            if (technical_result.signal_value * sentiment_result.signal_value) < 0:
                checks.append("Tech/sentiment disagreement")
                risk_score += 0.15

        risk_score = min(risk_score, 1.0)
        approved = risk_score < 0.5 and signal_strength >= 0.1

        # Position sizing based on risk
        if approved:
            position_pct = MAX_POSITION_PCT * (1 - risk_score)
            stop_loss = 0.02 * (1 + risk_score)  # Tighter stop when risky
            take_profit = 0.03 * (1 + signal_strength)
        else:
            position_pct = 0.0
            stop_loss = 0.02
            take_profit = 0.03

        assessment = RiskAssessment(
            approved=approved,
            risk_score=risk_score,
            max_position_pct=position_pct,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            reasons=checks,
        )

        elapsed = time.time() - start
        return self._make_result(
            signal_value=technical_result.signal_value if approved else 0.0,
            confidence=technical_result.confidence * (1 - risk_score),
            reasoning=f"{'APPROVED' if approved else 'REJECTED'}: risk={risk_score:.2f}",
            details={
                "approved": approved,
                "risk_score": risk_score,
                "max_position_pct": position_pct,
                "stop_loss_pct": stop_loss,
                "take_profit_pct": take_profit,
                "checks": checks,
            },
            execution_time=elapsed,
        )
