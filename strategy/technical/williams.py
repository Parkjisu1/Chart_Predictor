"""Larry Williams 전략 - Williams %R + Large Range Day.

출처: "Long-Term Secrets to Short-Term Trading" (Larry Williams)
- Williams %R: 일정 기간의 최고가 대비 현재가 위치 (과매수/과매도)
- Large Range Day: 평균 대비 비정상적 큰 캔들 → 추세 전환 신호
- Trap Day: 전일 고가/저가 돌파 후 반전 → 역추세 진입
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals import StrategyParameters, SignalOutput


def compute_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Williams %R: (최고가 - 현재가) / (최고가 - 최저가) * -100
    범위: -100 (과매도) ~ 0 (과매수)
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = (highest_high - close) / (highest_high - lowest_low) * -100
    return wr


def detect_large_range_day(
    df: pd.DataFrame, multiplier: float = 1.5, lookback: int = 20
) -> dict:
    """대폭 변동일 감지 (Larry Williams).
    현재 캔들의 range가 평균 range의 multiplier배 초과 시 감지.
    """
    ranges = df["high"] - df["low"]
    avg_range = ranges.rolling(window=lookback).mean()

    if ranges.empty or avg_range.isna().all():
        return {"detected": False, "ratio": 0, "direction": "none"}

    current_range = ranges.iloc[-1]
    current_avg = avg_range.iloc[-1]
    ratio = current_range / current_avg if current_avg > 0 else 0

    detected = ratio > multiplier
    # 방향: 종가가 캔들의 상단이면 강세, 하단이면 약세
    body_position = (df["close"].iloc[-1] - df["low"].iloc[-1]) / (current_range + 1e-10)
    direction = "bullish" if body_position > 0.6 else "bearish" if body_position < 0.4 else "neutral"

    return {"detected": detected, "ratio": round(ratio, 2), "direction": direction}


def detect_trap_day(df: pd.DataFrame) -> dict:
    """트랩 데이 감지: 전일 고가/저가 돌파 후 반전 마감.
    - Bull Trap: 전일 고가 돌파했으나 전일 종가 아래 마감
    - Bear Trap: 전일 저가 이탈했으나 전일 종가 위 마감
    """
    if len(df) < 3:
        return {"trap": "none"}

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    # Bull Trap
    if curr["high"] > prev["high"] and curr["close"] < prev["close"]:
        return {"trap": "bull_trap", "signal": "short"}

    # Bear Trap
    if curr["low"] < prev["low"] and curr["close"] > prev["close"]:
        return {"trap": "bear_trap", "signal": "long"}

    return {"trap": "none"}


def analyze_williams(df: pd.DataFrame, params: StrategyParameters) -> SignalOutput:
    """Williams 종합 분석."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    wr = compute_williams_r(high, low, close, 14)
    if wr.isna().all():
        return SignalOutput(name="williams", value=0.0, confidence=0.0)

    current_wr = wr.iloc[-1]
    lrd = detect_large_range_day(df)
    trap = detect_trap_day(df)

    # Williams %R 신호
    if current_wr < -80:  # 과매도
        value = (-80 - current_wr) / 20 * 0.6  # 0 ~ 0.6
        confidence = 0.6
    elif current_wr > -20:  # 과매수
        value = -(current_wr + 20) / 20 * 0.6  # 0 ~ -0.6
        confidence = 0.6
    else:
        value = (-50 - current_wr) / 50 * 0.2
        confidence = 0.3

    # Large Range Day 보정
    if lrd["detected"]:
        if lrd["direction"] == "bullish":
            value = min(value + 0.25, 1.0)
        elif lrd["direction"] == "bearish":
            value = max(value - 0.25, -1.0)
        confidence = min(confidence + 0.15, 1.0)

    # Trap Day 보정 (강력 역추세 신호)
    if trap["trap"] == "bear_trap":
        value = min(value + 0.3, 1.0)
        confidence = min(confidence + 0.2, 1.0)
    elif trap["trap"] == "bull_trap":
        value = max(value - 0.3, -1.0)
        confidence = min(confidence + 0.2, 1.0)

    return SignalOutput(
        name="williams",
        value=max(-1.0, min(1.0, value)),
        confidence=min(confidence, 1.0),
        details={
            "williams_r": round(current_wr, 2),
            "large_range_day": lrd,
            "trap_day": trap,
        },
    )
