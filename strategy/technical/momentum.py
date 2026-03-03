"""Momentum indicators: MACD and ADX."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals import StrategyParameters, SignalOutput


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, min_periods=period).mean()

    return adx


def compute_di(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series]:
    """Compute +DI and -DI."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr
    return plus_di, minus_di


def analyze_momentum(df: pd.DataFrame, params: StrategyParameters) -> SignalOutput:
    """Full momentum analysis: MACD crossover + ADX trend strength."""
    close = df["close"]
    macd_line, signal_line, histogram = compute_macd(
        close, params.macd_fast, params.macd_slow, params.macd_signal
    )
    adx = compute_adx(df, params.adx_period)
    plus_di, minus_di = compute_di(df, params.adx_period)

    if histogram.isna().all() or adx.isna().all():
        return SignalOutput(name="momentum", value=0.0, confidence=0.0)

    current_hist = histogram.iloc[-1]
    prev_hist = histogram.iloc[-2] if len(histogram) > 1 else 0
    current_adx = adx.iloc[-1]
    current_plus_di = plus_di.iloc[-1]
    current_minus_di = minus_di.iloc[-1]

    # MACD signal
    macd_value = 0.0
    if current_hist > 0 and prev_hist <= 0:
        macd_value = 0.6  # Bullish crossover
    elif current_hist < 0 and prev_hist >= 0:
        macd_value = -0.6  # Bearish crossover
    elif current_hist > 0:
        macd_value = min(0.4, current_hist / (abs(close.iloc[-1]) * 0.001 + 1e-10))
    else:
        macd_value = max(-0.4, current_hist / (abs(close.iloc[-1]) * 0.001 + 1e-10))

    # ADX trend direction from DI
    di_signal = 0.0
    if current_adx > params.adx_strong_trend:
        di_diff = (current_plus_di - current_minus_di) / 100
        di_signal = max(-0.4, min(0.4, di_diff))

    value = macd_value * 0.6 + di_signal * 0.4

    # ADX confidence
    if current_adx > params.adx_strong_trend:
        confidence = 0.7
    elif current_adx > params.adx_weak_trend:
        confidence = 0.5
    else:
        confidence = 0.3
        value *= 0.5  # Weak trend -> dampened signal

    value = max(-1.0, min(1.0, value))

    return SignalOutput(
        name="momentum",
        value=value,
        confidence=confidence,
        details={
            "macd_histogram": round(current_hist, 4),
            "adx": round(current_adx, 2),
            "plus_di": round(current_plus_di, 2),
            "minus_di": round(current_minus_di, 2),
            "crossover": "bullish" if current_hist > 0 and prev_hist <= 0
                        else "bearish" if current_hist < 0 and prev_hist >= 0
                        else "none",
        },
    )
