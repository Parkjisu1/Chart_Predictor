"""일목균형표 (Ichimoku Kinko Hyo) - 일목산인.

전환선(9), 기준선(26), 선행스팬A/B(26/52), 후행스팬(26)
- 삼역호전: 전환선>기준선 + 가격>구름 + 후행스팬>가격 → 강력 매수
- 삼역역전: 전환선<기준선 + 가격<구름 + 후행스팬<가격 → 강력 매도
- 구름대 두께: 변동성/지지저항 강도 판단
- 구름 비틀림: 추세 전환 신호
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals import StrategyParameters, SignalOutput


def compute_ichimoku(
    high: pd.Series, low: pd.Series, close: pd.Series,
    tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
) -> dict[str, pd.Series]:
    """일목균형표 5개 라인 계산."""
    # 전환선 (Tenkan-sen): 9일 최고가+최저가 / 2
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2

    # 기준선 (Kijun-sen): 26일 최고가+최저가 / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2

    # 선행스팬 A (Senkou Span A): (전환선+기준선)/2, 26일 선행
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)

    # 선행스팬 B (Senkou Span B): 52일 최고+최저/2, 26일 선행
    senkou_b_line = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)

    # 후행스팬 (Chikou Span): 현재 종가를 26일 후행
    chikou = close.shift(-kijun)

    return {
        "tenkan": tenkan_sen,
        "kijun": kijun_sen,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b_line,
        "chikou": chikou,
    }


def check_sangyaku(
    close: pd.Series, ichimoku: dict[str, pd.Series]
) -> str:
    """삼역호전/삼역역전 판단.
    Returns: 'sangyaku_koten' (강력매수), 'sangyaku_gyakuten' (강력매도), 'none'
    """
    if any(v.isna().all() for v in ichimoku.values()):
        return "none"

    idx = -1
    tenkan = ichimoku["tenkan"].iloc[idx]
    kijun = ichimoku["kijun"].iloc[idx]
    senkou_a = ichimoku["senkou_a"].iloc[idx]
    senkou_b = ichimoku["senkou_b"].iloc[idx]
    price = close.iloc[idx]

    # 후행스팬은 26일 전 가격과 비교
    chikou_idx = max(0, len(close) - 27)
    chikou_price = close.iloc[chikou_idx] if chikou_idx < len(close) else price
    current_price_for_chikou = close.iloc[idx]

    cloud_top = max(senkou_a, senkou_b) if not (np.isnan(senkou_a) or np.isnan(senkou_b)) else 0
    cloud_bottom = min(senkou_a, senkou_b) if not (np.isnan(senkou_a) or np.isnan(senkou_b)) else 0

    # 삼역호전: 전환선>기준선 AND 가격>구름 AND 후행스팬>가격(26일전)
    if (tenkan > kijun and price > cloud_top and
            current_price_for_chikou > chikou_price):
        return "sangyaku_koten"

    # 삼역역전: 전환선<기준선 AND 가격<구름 AND 후행스팬<가격(26일전)
    if (tenkan < kijun and price < cloud_bottom and
            current_price_for_chikou < chikou_price):
        return "sangyaku_gyakuten"

    return "none"


def analyze_ichimoku(df: pd.DataFrame, params: StrategyParameters) -> SignalOutput:
    """일목균형표 종합 분석."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ich = compute_ichimoku(high, low, close)
    sangyaku = check_sangyaku(close, ich)

    if ich["tenkan"].isna().all() or ich["kijun"].isna().all():
        return SignalOutput(name="ichimoku", value=0.0, confidence=0.0)

    tenkan = ich["tenkan"].iloc[-1]
    kijun = ich["kijun"].iloc[-1]
    senkou_a = ich["senkou_a"].iloc[-1] if not np.isnan(ich["senkou_a"].iloc[-1]) else 0
    senkou_b = ich["senkou_b"].iloc[-1] if not np.isnan(ich["senkou_b"].iloc[-1]) else 0
    price = close.iloc[-1]

    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)

    value = 0.0
    confidence = 0.4

    # 1. 전환선/기준선 교차
    if tenkan > kijun:
        value += 0.2
    elif tenkan < kijun:
        value -= 0.2

    # 2. 가격과 구름 관계
    if price > cloud_top:
        value += 0.25  # 구름 위: 강세
    elif price < cloud_bottom:
        value -= 0.25  # 구름 아래: 약세
    else:
        value *= 0.5   # 구름 안: 신호 약화
        confidence *= 0.7

    # 3. 구름 미래 방향 (센코우A vs B)
    if senkou_a > senkou_b:
        value += 0.1  # 양운 (강세 구름)
    else:
        value -= 0.1  # 음운 (약세 구름)

    # 4. 삼역호전/역전 (가장 강력한 신호)
    if sangyaku == "sangyaku_koten":
        value = min(value + 0.4, 1.0)
        confidence = 0.85
    elif sangyaku == "sangyaku_gyakuten":
        value = max(value - 0.4, -1.0)
        confidence = 0.85

    # 5. 구름 두께 → 지지/저항 강도
    cloud_thickness = abs(senkou_a - senkou_b) / price if price > 0 else 0
    if cloud_thickness > 0.02:  # 두꺼운 구름 = 강한 지지/저항
        confidence = min(confidence + 0.1, 1.0)

    return SignalOutput(
        name="ichimoku",
        value=max(-1.0, min(1.0, value)),
        confidence=confidence,
        details={
            "tenkan": round(tenkan, 2),
            "kijun": round(kijun, 2),
            "senkou_a": round(senkou_a, 2),
            "senkou_b": round(senkou_b, 2),
            "price_vs_cloud": (
                "above" if price > cloud_top
                else "below" if price < cloud_bottom
                else "inside"
            ),
            "sangyaku": sangyaku,
            "cloud_thickness_pct": round(cloud_thickness * 100, 3),
        },
    )
