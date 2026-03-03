"""RSI indicator with divergence detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals import StrategyParameters, SignalOutput


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_divergence(
    close: pd.Series, rsi: pd.Series, lookback: int = 20
) -> str:
    """Detect bullish/bearish divergence between price and RSI.

    Returns: 'bullish', 'bearish', or 'none'
    """
    if len(close) < lookback:
        return "none"

    recent_close = close.iloc[-lookback:]
    recent_rsi = rsi.iloc[-lookback:]

    # Find local lows/highs
    close_min_idx = recent_close.idxmin()
    close_max_idx = recent_close.idxmax()

    # Bullish divergence: price makes lower low, RSI makes higher low
    if (close.iloc[-1] < recent_close.loc[close_min_idx] * 1.001 and
            rsi.iloc[-1] > recent_rsi.min() + 2):
        return "bullish"

    # Bearish divergence: price makes higher high, RSI makes lower high
    if (close.iloc[-1] > recent_close.loc[close_max_idx] * 0.999 and
            rsi.iloc[-1] < recent_rsi.max() - 2):
        return "bearish"

    return "none"


def analyze_rsi(df: pd.DataFrame, params: StrategyParameters) -> SignalOutput:
    """Full RSI analysis including divergence."""
    close = df["close"]
    rsi = compute_rsi(close, params.rsi_period)

    if rsi.empty or rsi.isna().all():
        return SignalOutput(name="rsi", value=0.0, confidence=0.0)

    current_rsi = rsi.iloc[-1]
    divergence = detect_divergence(close, rsi, params.rsi_divergence_lookback)

    # Base signal from RSI level
    if current_rsi <= params.rsi_oversold:
        value = (params.rsi_oversold - current_rsi) / params.rsi_oversold  # 0 to 1
        confidence = 0.7
    elif current_rsi >= params.rsi_overbought:
        value = -(current_rsi - params.rsi_overbought) / (100 - params.rsi_overbought)
        confidence = 0.7
    else:
        # Neutral zone: slight signal based on distance from 50
        value = (50 - current_rsi) / 50 * 0.3
        confidence = 0.3

    # Boost on divergence
    if divergence == "bullish":
        value = min(value + 0.3, 1.0)
        confidence = min(confidence + 0.2, 1.0)
    elif divergence == "bearish":
        value = max(value - 0.3, -1.0)
        confidence = min(confidence + 0.2, 1.0)

    value = max(-1.0, min(1.0, value))

    return SignalOutput(
        name="rsi",
        value=value,
        confidence=confidence,
        details={
            "rsi": round(current_rsi, 2),
            "divergence": divergence,
        },
    )
