"""Prompt templates for macro-level market analysis."""

from __future__ import annotations


def build_daily_review_prompt(
    symbol: str,
    daily_pnl: float,
    win_rate: float,
    total_trades: int,
    max_drawdown: float,
    open_positions: list[dict],
) -> str:
    """Build prompt for supervisor daily review."""
    positions_str = ""
    for p in open_positions:
        positions_str += (
            f"  - {p.get('symbol', '?')}: {p.get('side', '?')} "
            f"@ {p.get('entry_price', 0):.2f}, "
            f"PnL: {p.get('pnl_pct', 0):+.2f}%\n"
        )

    return f"""Daily trading review for {symbol} futures system.

Performance Today:
- Daily PnL: {daily_pnl:+.2f}%
- Win Rate: {win_rate:.1f}%
- Total Trades: {total_trades}
- Max Drawdown: {max_drawdown:.2f}%

Open Positions:
{positions_str if positions_str else "  None"}

Respond in exactly this JSON format (no other text):
{{
    "overall_assessment": "<good|acceptable|concerning|critical>",
    "should_continue": true,
    "risk_level": "<low|medium|high|critical>",
    "recommendations": ["<rec1>", "<rec2>"],
    "kill_switch": false,
    "kill_reason": ""
}}

Rules:
- Set kill_switch=true ONLY if max_drawdown > 15% or daily loss > 5%
- Be honest but not overly cautious
- Focus on risk management recommendations
"""


def build_learning_insight_prompt(
    iteration: int,
    win_rate: float,
    loss_breakdown: dict,
    current_params: dict,
    recent_trades_summary: str,
) -> str:
    """Build prompt for learning loop insights."""
    return f"""Analyze trading strategy performance after iteration {iteration}.

Current Win Rate: {win_rate:.1f}%
Target Win Rate: 90%

Loss Breakdown by Failure Mode:
{_format_dict(loss_breakdown)}

Current Key Parameters:
{_format_dict(current_params)}

Recent Trades Summary:
{recent_trades_summary}

Respond in exactly this JSON format (no other text):
{{
    "analysis": "<2-3 sentence analysis>",
    "primary_issue": "<main failure mode to address>",
    "parameter_adjustments": {{
        "<param_name>": <new_value>,
        "<param_name2>": <new_value2>
    }},
    "confidence_in_adjustments": <float 0.0 to 1.0>,
    "convergence_assessment": "<improving|stagnant|diverging>"
}}

Rules:
- Suggest at most 3 parameter changes per iteration
- Keep adjustments within 10% of current values
- Focus on the dominant failure mode
- If win rate is already above 85%, suggest very small adjustments
"""


def _format_dict(d: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in d.items())
