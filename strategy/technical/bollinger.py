"""Bollinger Bands with %B and squeeze detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals import StrategyParameters, SignalOutput


def compute_bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands. Returns (middle, upper, lower, %B)."""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    pct_b = (close - lower) / (upper - lower)
    return middle, upper, lower, pct_b


def detect_squeeze(
    close: pd.Series, period: int = 20, std_dev: float = 2.0,
    threshold: float = 0.05,
) -> bool:
    """Detect Bollinger Band squeeze (low volatility -> potential breakout)."""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bandwidth = (upper - lower) / middle
    if bandwidth.empty or bandwidth.isna().all():
        return False
    return bandwidth.iloc[-1] < threshold


def analyze_bollinger(df: pd.DataFrame, params: StrategyParameters) -> SignalOutput:
    """Full Bollinger Bands analysis."""
    close = df["close"]
    middle, upper, lower, pct_b = compute_bollinger_bands(
        close, params.bb_period, params.bb_std_dev
    )

    if pct_b.empty or pct_b.isna().all():
        return SignalOutput(name="bollinger", value=0.0, confidence=0.0)

    current_pct_b = pct_b.iloc[-1]
    is_squeeze = detect_squeeze(close, params.bb_period, params.bb_std_dev,
                                 params.bb_squeeze_threshold)

    # Signal based on %B
    if current_pct_b < 0:
        # Below lower band - oversold
        value = min(abs(current_pct_b), 1.0)
        confidence = 0.7
    elif current_pct_b > 1:
        # Above upper band - overbought
        value = -min(current_pct_b - 1, 1.0)
        confidence = 0.7
    elif current_pct_b < 0.2:
        # Near lower band
        value = (0.2 - current_pct_b) / 0.2 * 0.5
        confidence = 0.5
    elif current_pct_b > 0.8:
        # Near upper band
        value = -(current_pct_b - 0.8) / 0.2 * 0.5
        confidence = 0.5
    else:
        # Mid-band, neutral
        value = (0.5 - current_pct_b) * 0.2
        confidence = 0.2

    # Squeeze modifier
    if is_squeeze:
        confidence = min(confidence + 0.15, 1.0)

    value = max(-1.0, min(1.0, value))

    return SignalOutput(
        name="bollinger",
        value=value,
        confidence=confidence,
        details={
            "pct_b": round(current_pct_b, 4),
            "is_squeeze": is_squeeze,
            "upper": round(upper.iloc[-1], 2) if not upper.isna().all() else None,
            "lower": round(lower.iloc[-1], 2) if not lower.isna().all() else None,
        },
    )
