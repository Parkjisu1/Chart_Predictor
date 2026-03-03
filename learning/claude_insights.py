"""Claude CLI integration for learning loop insights."""

from __future__ import annotations

import json

from agents.base import ClaudeCLIRunner
from config.logging_config import get_logger
from strategy.sentiment.macro_prompts import build_learning_insight_prompt

logger = get_logger(__name__)


class ClaudeInsightsEngine:
    """Use Claude CLI to analyze strategy performance and suggest adjustments."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.claude = ClaudeCLIRunner() if enabled else None

    def get_insights(
        self,
        iteration: int,
        win_rate: float,
        loss_breakdown: dict,
        current_params: dict,
        recent_trades_summary: str,
    ) -> dict:
        """Get Claude's analysis and parameter adjustment suggestions."""
        if not self.enabled or not self.claude:
            return self._fallback_insights(win_rate, loss_breakdown)

        prompt = build_learning_insight_prompt(
            iteration=iteration,
            win_rate=win_rate,
            loss_breakdown=loss_breakdown,
            current_params=current_params,
            recent_trades_summary=recent_trades_summary,
        )

        result = self.claude.run(prompt)

        if result:
            logger.info(
                "claude_insights_received",
                iteration=iteration,
                adjustments=list(result.get("parameter_adjustments", {}).keys()),
            )
            return result

        logger.warning("claude_insights_fallback", iteration=iteration)
        return self._fallback_insights(win_rate, loss_breakdown)

    def _fallback_insights(
        self, win_rate: float, loss_breakdown: dict
    ) -> dict:
        """Rule-based fallback when Claude CLI is unavailable."""
        adjustments = {}
        primary_issue = "unknown"

        # Find dominant failure mode
        modes = loss_breakdown.get("failure_modes", {})
        if modes:
            dominant = max(modes.items(), key=lambda x: x[1].get("count", 0))
            primary_issue = dominant[0]

            # Simple rule-based adjustments
            if primary_issue == "early_entry":
                adjustments["stop_loss_atr_multiplier"] = "increase_10pct"
                adjustments["signal_threshold"] = "increase_5pct"
            elif primary_issue == "wrong_direction":
                adjustments["strong_signal_threshold"] = "increase_5pct"
                adjustments["weight_momentum"] = "increase_5pct"
            elif primary_issue == "overleveraged":
                adjustments["max_position_pct"] = "decrease_10pct"
            elif primary_issue == "high_volatility":
                adjustments["weight_garch"] = "increase_10pct"
            elif primary_issue == "premature_exit":
                adjustments["take_profit_atr_multiplier"] = "increase_10pct"

        return {
            "analysis": f"Win rate {win_rate:.1%}. Dominant issue: {primary_issue}.",
            "primary_issue": primary_issue,
            "parameter_adjustments": adjustments,
            "confidence_in_adjustments": 0.5,
            "convergence_assessment": "improving" if win_rate > 0.6 else "stagnant",
        }
