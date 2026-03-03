"""Alexander Elder 전략 - Triple Screen, Force Index, Elder Ray.

출처: "Trading for a Living" / "The New Trading for a Living" (Alexander Elder)
- Triple Screen: 3개 타임프레임 필터 (추세→파도→타이밍)
- Force Index: 가격변화 × 거래량 → 매수/매도 압력 측정
- Elder Ray: Bull Power(고가-EMA) / Bear Power(저가-EMA) → 매수/매도 힘
- Impulse System: EMA 기울기 + MACD-H 방향 → 매수/매도/중립 색상
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals import StrategyParameters, SignalOutput


def compute_force_index(
    close: pd.Series, volume: pd.Series, period: int = 13
) -> pd.Series:
    """Force Index = 가격변화 × 거래량 (EMA 평활화).
    양수: 매수 압력 우세, 음수: 매도 압력 우세
    """
    raw_force = close.diff() * volume
    force_index = raw_force.ewm(span=period, adjust=False).mean()
    return force_index


def compute_elder_ray(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13
) -> tuple[pd.Series, pd.Series]:
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA.
    - Bull Power > 0 & 상승 → 강세
    - Bear Power < 0 & 상승 → 약세 약화 (반등 기회)
    """
    ema = close.ewm(span=period, adjust=False).mean()
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power


def compute_impulse_system(
    close: pd.Series, period: int = 13,
    macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
) -> pd.Series:
    """Impulse System: EMA 기울기 + MACD 히스토그램 방향.
    +1: 녹색(매수 허용), -1: 적색(매도 허용), 0: 파란색(양쪽 허용)
    """
    ema = close.ewm(span=period, adjust=False).mean()
    ema_slope = ema.diff()

    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
    histogram = macd_line - signal_line
    hist_slope = histogram.diff()

    impulse = pd.Series(0, index=close.index)
    impulse[(ema_slope > 0) & (hist_slope > 0)] = 1   # 녹색: 매수만
    impulse[(ema_slope < 0) & (hist_slope < 0)] = -1  # 적색: 매도만
    return impulse


def analyze_elder(df: pd.DataFrame, params: StrategyParameters) -> SignalOutput:
    """Elder 종합 분석: Force Index + Elder Ray + Impulse System."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    force = compute_force_index(close, volume, 13)
    bull_power, bear_power = compute_elder_ray(high, low, close, 13)
    impulse = compute_impulse_system(close, 13)

    if force.isna().all() or bull_power.isna().all():
        return SignalOutput(name="elder", value=0.0, confidence=0.0)

    current_force = force.iloc[-1]
    current_bull = bull_power.iloc[-1]
    current_bear = bear_power.iloc[-1]
    current_impulse = impulse.iloc[-1]

    # Force Index 신호 (-0.4 ~ 0.4)
    force_ma = force.rolling(20).mean().iloc[-1]
    if force_ma != 0:
        force_signal = np.clip(current_force / (abs(force_ma) * 3 + 1e-10), -0.4, 0.4)
    else:
        force_signal = 0.0

    # Elder Ray 신호 (-0.3 ~ 0.3)
    elder_signal = 0.0
    if current_bull > 0 and current_bear > 0:
        elder_signal = 0.3  # 강세
    elif current_bull < 0 and current_bear < 0:
        elder_signal = -0.3  # 약세
    elif current_bear < 0 and bear_power.diff().iloc[-1] > 0:
        elder_signal = 0.2  # 약세 약화 → 반등
    elif current_bull > 0 and bull_power.diff().iloc[-1] < 0:
        elder_signal = -0.2  # 강세 약화 → 하락

    # Impulse System 필터
    impulse_modifier = 1.0
    if current_impulse == 1:
        # 녹색: 매수만 허용
        if force_signal < 0:
            force_signal *= 0.3  # 매도 신호 대폭 축소
        impulse_modifier = 1.2
    elif current_impulse == -1:
        # 적색: 매도만 허용
        if force_signal > 0:
            force_signal *= 0.3  # 매수 신호 대폭 축소
        impulse_modifier = 1.2

    value = (force_signal * 0.5 + elder_signal * 0.5) * impulse_modifier
    value = max(-1.0, min(1.0, value))

    confidence = 0.5
    if current_impulse != 0:
        confidence = 0.65  # Impulse 확인 시 신뢰도 상승
    if abs(current_force) > abs(force_ma) * 2:
        confidence = min(confidence + 0.15, 1.0)  # 강한 Force 시

    return SignalOutput(
        name="elder",
        value=value,
        confidence=confidence,
        details={
            "force_index": round(float(current_force), 2),
            "bull_power": round(float(current_bull), 2),
            "bear_power": round(float(current_bear), 2),
            "impulse": int(current_impulse),
            "impulse_label": {1: "green_buy", -1: "red_sell", 0: "blue_neutral"}.get(
                int(current_impulse), "unknown"
            ),
        },
    )
