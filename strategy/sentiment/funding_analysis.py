"""Funding rate sentiment analysis."""

from __future__ import annotations

import pandas as pd
import numpy as np

from strategy.signals import SignalOutput


def analyze_funding_sentiment(funding_df: pd.DataFrame) -> SignalOutput:
    """Analyze funding rate for market sentiment.

    Positive funding = longs pay shorts (bullish crowded trade)
    Negative funding = shorts pay longs (bearish crowded trade)
    Extreme funding -> contrarian signal
    """
    if funding_df.empty or "funding_rate" not in funding_df.columns:
        return SignalOutput(name="funding", value=0.0, confidence=0.0)

    recent_rates = funding_df["funding_rate"].tail(24)  # Last ~3 days
    current_rate = recent_rates.iloc[-1] if len(recent_rates) > 0 else 0
    avg_rate = recent_rates.mean()

    # Extreme funding -> contrarian signal
    if current_rate > 0.001:  # 0.1%+ = very bullish crowd -> short signal
        value = -min(current_rate / 0.003, 1.0) * 0.5
        confidence = 0.6
    elif current_rate < -0.001:  # Very bearish crowd -> long signal
        value = min(abs(current_rate) / 0.003, 1.0) * 0.5
        confidence = 0.6
    else:
        # Normal range
        value = -current_rate / 0.001 * 0.2
        confidence = 0.3

    return SignalOutput(
        name="funding",
        value=max(-1.0, min(1.0, value)),
        confidence=confidence,
        details={
            "current_rate": round(current_rate, 6),
            "avg_rate_3d": round(avg_rate, 6),
            "interpretation": (
                "extreme_bullish" if current_rate > 0.001
                else "extreme_bearish" if current_rate < -0.001
                else "neutral"
            ),
        },
    )
