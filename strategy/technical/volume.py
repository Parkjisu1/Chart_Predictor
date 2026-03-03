"""Volume indicators: OBV, VWAP, and volume spike detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals import StrategyParameters, SignalOutput


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (volume * direction).cumsum()
    return obv


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute Volume Weighted Average Price (intraday reset)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap


def detect_volume_spike(
    volume: pd.Series, period: int = 20, multiplier: float = 2.0
) -> bool:
    """Detect if current volume is a spike relative to moving average."""
    vol_ma = volume.rolling(window=period).mean()
    if vol_ma.empty or vol_ma.isna().all():
        return False
    return volume.iloc[-1] > vol_ma.iloc[-1] * multiplier


def analyze_volume(df: pd.DataFrame, params: StrategyParameters) -> SignalOutput:
    """Full volume analysis: OBV trend + VWAP deviation + spikes."""
    close = df["close"]
    volume = df["volume"]

    obv = compute_obv(close, volume)
    vwap = compute_vwap(df)
    has_spike = detect_volume_spike(volume, params.obv_ma_period,
                                     params.volume_spike_multiplier)

    if obv.empty or obv.isna().all():
        return SignalOutput(name="volume", value=0.0, confidence=0.0)

    # OBV trend: compare OBV MA slope
    obv_ma = obv.rolling(window=params.obv_ma_period).mean()
    if len(obv_ma.dropna()) < 2:
        return SignalOutput(name="volume", value=0.0, confidence=0.1)

    obv_slope = obv_ma.iloc[-1] - obv_ma.iloc[-5] if len(obv_ma.dropna()) >= 5 else 0
    obv_signal = np.sign(obv_slope) * min(abs(obv_slope) / (abs(obv_ma.iloc[-1]) + 1e-10), 1.0) * 0.4

    # VWAP deviation
    vwap_signal = 0.0
    if not vwap.isna().all() and vwap.iloc[-1] > 0:
        deviation = (close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
        # Price below VWAP = potential long, above = potential short
        vwap_signal = -deviation / (params.vwap_deviation_threshold / 100) * 0.3
        vwap_signal = max(-0.3, min(0.3, vwap_signal))

    value = obv_signal + vwap_signal
    confidence = 0.4
    if has_spike:
        confidence = 0.6

    value = max(-1.0, min(1.0, value))

    return SignalOutput(
        name="volume",
        value=value,
        confidence=confidence,
        details={
            "obv_slope": round(obv_slope, 2) if not np.isnan(obv_slope) else 0,
            "vwap": round(vwap.iloc[-1], 2) if not vwap.isna().all() else None,
            "volume_spike": has_spike,
        },
    )
