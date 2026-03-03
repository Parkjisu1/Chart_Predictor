"""Prompt templates for Claude CLI on-chain/market sentiment analysis."""

from __future__ import annotations


def build_market_sentiment_prompt(
    symbol: str,
    price: float,
    price_change_24h: float,
    volume_24h: float,
    funding_rate: float,
    rsi: float,
    additional_context: str = "",
) -> str:
    """Build prompt for Claude CLI to analyze market sentiment."""
    return f"""Analyze the current market sentiment for {symbol} futures trading.

Market Data:
- Current Price: ${price:,.2f}
- 24h Price Change: {price_change_24h:+.2f}%
- 24h Volume: ${volume_24h:,.0f}
- Current Funding Rate: {funding_rate:.6f}
- RSI(14): {rsi:.1f}
{f"- Additional: {additional_context}" if additional_context else ""}

Respond in exactly this JSON format (no other text):
{{
    "sentiment_score": <float from -1.0 to 1.0>,
    "confidence": <float from 0.0 to 1.0>,
    "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
    "risk_level": "<low|medium|high>",
    "recommendation": "<strong_long|long|neutral|short|strong_short>"
}}

Rules:
- Base your analysis on the quantitative data provided
- Consider funding rate extremes as contrarian indicators
- High RSI + high funding rate = increased short risk
- Low RSI + negative funding = potential long opportunity
- Be conservative; default to neutral when uncertain
"""


def build_macro_prompt(
    symbol: str,
    recent_events: str = "",
) -> str:
    """Build prompt for macro analysis via Claude CLI."""
    return f"""Provide a brief macro sentiment assessment for {symbol} crypto futures.

{f"Recent context: {recent_events}" if recent_events else ""}

Respond in exactly this JSON format (no other text):
{{
    "macro_sentiment": <float from -1.0 to 1.0>,
    "confidence": <float from 0.0 to 1.0>,
    "key_themes": ["<theme1>", "<theme2>"],
    "risk_events": ["<event1>"]
}}

Be concise and data-driven. Default to 0.0 sentiment and low confidence if uncertain.
"""
