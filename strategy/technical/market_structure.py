"""시장 구조 분석 - 지지/저항, 피보나치, 추세 구조.

출처: "Technical Analysis of the Financial Markets" (John J. Murphy)
- 지지/저항선: 피봇 포인트 기반 자동 감지
- 피보나치 되돌림: 38.2%, 50%, 61.8% 수준
- 시장 구조: Higher High/Higher Low (상승추세), Lower High/Lower Low (하락추세)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals import StrategyParameters, SignalOutput


def find_pivot_points(
    high: pd.Series, low: pd.Series, window: int = 5
) -> tuple[list[float], list[float]]:
    """피봇 포인트로 지지/저항 수준 감지."""
    resistances = []
    supports = []

    for i in range(window, len(high) - window):
        # 저항: 좌우 window 캔들보다 높은 고가
        if high.iloc[i] == high.iloc[i - window:i + window + 1].max():
            resistances.append(float(high.iloc[i]))
        # 지지: 좌우 window 캔들보다 낮은 저가
        if low.iloc[i] == low.iloc[i - window:i + window + 1].min():
            supports.append(float(low.iloc[i]))

    return resistances, supports


def find_nearest_levels(
    price: float, supports: list[float], resistances: list[float], n: int = 3
) -> dict:
    """현재가 기준 가장 가까운 지지/저항 수준 n개."""
    above = sorted([r for r in resistances if r > price])[:n]
    below = sorted([s for s in supports if s < price], reverse=True)[:n]

    nearest_resistance = above[0] if above else None
    nearest_support = below[0] if below else None

    return {
        "nearest_resistance": nearest_resistance,
        "nearest_support": nearest_support,
        "resistance_distance_pct": round((nearest_resistance / price - 1) * 100, 3) if nearest_resistance else None,
        "support_distance_pct": round((1 - nearest_support / price) * 100, 3) if nearest_support else None,
        "resistances": above,
        "supports": below,
    }


def compute_fibonacci_levels(swing_high: float, swing_low: float) -> dict[str, float]:
    """피보나치 되돌림 수준 계산."""
    diff = swing_high - swing_low
    return {
        "0.0": swing_high,
        "0.236": swing_high - diff * 0.236,
        "0.382": swing_high - diff * 0.382,
        "0.500": swing_high - diff * 0.500,
        "0.618": swing_high - diff * 0.618,
        "0.786": swing_high - diff * 0.786,
        "1.0": swing_low,
    }


def detect_market_structure(
    high: pd.Series, low: pd.Series, lookback: int = 20
) -> dict:
    """시장 구조 판단: HH/HL (상승), LH/LL (하락), 혼조.

    John Murphy의 추세 정의: 연속적인 Higher High + Higher Low = 상승 추세
    """
    if len(high) < lookback * 2:
        return {"structure": "unknown", "trend": "unknown"}

    # 최근 피봇 포인트 찾기
    pivot_window = 5
    swing_highs = []
    swing_lows = []

    recent = slice(-lookback * 2, None)
    h = high.iloc[recent].reset_index(drop=True)
    l = low.iloc[recent].reset_index(drop=True)

    for i in range(pivot_window, len(h) - pivot_window):
        if h.iloc[i] == h.iloc[i - pivot_window:i + pivot_window + 1].max():
            swing_highs.append(float(h.iloc[i]))
        if l.iloc[i] == l.iloc[i - pivot_window:i + pivot_window + 1].min():
            swing_lows.append(float(l.iloc[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"structure": "insufficient", "trend": "unknown"}

    # 최근 2개 스윙 비교
    hh = swing_highs[-1] > swing_highs[-2]  # Higher High
    hl = swing_lows[-1] > swing_lows[-2]    # Higher Low
    lh = swing_highs[-1] < swing_highs[-2]  # Lower High
    ll = swing_lows[-1] < swing_lows[-2]    # Lower Low

    if hh and hl:
        structure = "uptrend"
        trend = "bullish"
    elif lh and ll:
        structure = "downtrend"
        trend = "bearish"
    elif hh and ll:
        structure = "expanding"
        trend = "volatile"
    elif lh and hl:
        structure = "contracting"
        trend = "consolidation"
    else:
        structure = "mixed"
        trend = "neutral"

    return {
        "structure": structure,
        "trend": trend,
        "higher_high": hh,
        "higher_low": hl,
        "recent_swing_highs": swing_highs[-3:],
        "recent_swing_lows": swing_lows[-3:],
    }


def analyze_market_structure(df: pd.DataFrame, params: StrategyParameters) -> SignalOutput:
    """시장 구조 종합 분석."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    price = close.iloc[-1]

    # 지지/저항
    resistances, supports = find_pivot_points(high, low, window=5)
    levels = find_nearest_levels(price, supports, resistances)

    # 피보나치
    lookback = min(100, len(df))
    swing_high = float(high.iloc[-lookback:].max())
    swing_low = float(low.iloc[-lookback:].min())
    fib = compute_fibonacci_levels(swing_high, swing_low)

    # 시장 구조
    structure = detect_market_structure(high, low)

    # 종합 신호
    value = 0.0
    confidence = 0.4

    # 1. 추세 구조
    if structure["trend"] == "bullish":
        value += 0.35
        confidence = 0.6
    elif structure["trend"] == "bearish":
        value -= 0.35
        confidence = 0.6
    elif structure["trend"] == "consolidation":
        confidence = 0.3  # 횡보 시 신뢰도 감소

    # 2. 지지/저항 근접도
    if levels["support_distance_pct"] is not None and levels["support_distance_pct"] < 1.0:
        value += 0.2  # 지지선 근처 → 반등 기대
        confidence += 0.1
    if levels["resistance_distance_pct"] is not None and levels["resistance_distance_pct"] < 1.0:
        value -= 0.2  # 저항선 근처 → 저항 기대
        confidence += 0.1

    # 3. 피보나치 수준 근접
    for level_name, level_price in fib.items():
        if abs(price - level_price) / price < 0.005:  # 0.5% 이내
            confidence += 0.1
            break

    return SignalOutput(
        name="market_structure",
        value=max(-1.0, min(1.0, value)),
        confidence=min(confidence, 1.0),
        details={
            "structure": structure,
            "levels": {
                "nearest_support": levels["nearest_support"],
                "nearest_resistance": levels["nearest_resistance"],
                "support_dist_pct": levels["support_distance_pct"],
                "resistance_dist_pct": levels["resistance_distance_pct"],
            },
            "fibonacci": {k: round(v, 2) for k, v in fib.items()},
        },
    )
