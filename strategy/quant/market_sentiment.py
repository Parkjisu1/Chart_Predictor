"""시장 감성 지표 - 롱숏비율, 공포탐욕, 청산 데이터.

퀀트 매니저 핵심 원칙: 극단적 군중 심리의 반대편에 서라.
- 극단적 탐욕 → 역추세 매도 신호
- 극단적 공포 → 역추세 매수 신호
- 롱숏비율 극단 → 역추세 신호
"""

from __future__ import annotations

import numpy as np

from strategy.signals import SignalOutput


def analyze_fear_greed(fg_data: dict) -> SignalOutput:
    """공포탐욕지수 분석.
    0-24: Extreme Fear → 매수 기회 (Warren Buffett: "공포 속에 매수")
    25-44: Fear → 약한 매수
    45-55: Neutral
    56-75: Greed → 약한 매도
    76-100: Extreme Greed → 매도 기회
    """
    value_raw = fg_data.get("value", 50)
    history = fg_data.get("history", [])

    # 역추세 신호: 극단값의 반대
    normalized = (value_raw - 50) / 50  # -1 ~ +1
    contrarian_signal = -normalized * 0.6  # 역추세

    confidence = 0.3
    if value_raw <= 20 or value_raw >= 80:
        confidence = 0.7  # 극단값 높은 신뢰도
    elif value_raw <= 30 or value_raw >= 70:
        confidence = 0.5

    # 추세 확인: 연속 극단값
    if len(history) >= 7:
        recent_avg = np.mean([h["value"] for h in history[:7]])
        if recent_avg <= 25 or recent_avg >= 75:
            confidence = min(confidence + 0.15, 0.85)

    label = fg_data.get("label", "neutral")

    return SignalOutput(
        name="fear_greed",
        value=float(np.clip(contrarian_signal, -1.0, 1.0)),
        confidence=confidence,
        details={
            "index": value_raw,
            "label": label,
            "signal_type": "contrarian",
            "zone": (
                "extreme_fear" if value_raw <= 20
                else "fear" if value_raw <= 44
                else "neutral" if value_raw <= 55
                else "greed" if value_raw <= 75
                else "extreme_greed"
            ),
        },
    )


def analyze_long_short_ratio(ls_data: dict) -> SignalOutput:
    """롱숏비율 분석 (역추세).
    비율 > 2.0: 롱 과밀 → 매도 신호
    비율 < 0.5: 숏 과밀 → 매수 신호
    """
    ratio = ls_data.get("ratio", 1.0)
    extreme = ls_data.get("extreme", False)

    # 역추세 신호
    if ratio > 1:
        # 롱 과밀 → 매도 방향
        value = -min((ratio - 1) / 2, 1.0) * 0.5
    else:
        # 숏 과밀 → 매수 방향
        value = min((1 - ratio) / 0.5, 1.0) * 0.5

    confidence = 0.4
    if extreme:
        confidence = 0.7
        value *= 1.3

    return SignalOutput(
        name="long_short_ratio",
        value=float(np.clip(value, -1.0, 1.0)),
        confidence=confidence,
        details={
            "ratio": ratio,
            "long_pct": ls_data.get("long_pct", 50),
            "short_pct": ls_data.get("short_pct", 50),
            "extreme": extreme,
            "signal_type": "contrarian",
        },
    )


def analyze_liquidation_pressure(liq_data: dict) -> SignalOutput:
    """청산 데이터 기반 압력 분석.
    24h 가격 변동이 크면 청산 캐스케이드 후 반전 가능성.
    """
    if not liq_data.get("available", False):
        return SignalOutput(name="liquidation", value=0.0, confidence=0.0)

    price_change = liq_data.get("price_change_pct", 0)
    volume_24h = liq_data.get("volume_24h", 0)

    # 급격한 하락 후 → 롱 청산 완료 → 반등 가능
    if price_change < -5:
        value = min(abs(price_change) / 20, 0.6)  # 매수 방향
        confidence = 0.5
    # 급격한 상승 후 → 숏 청산 완료 → 조정 가능
    elif price_change > 5:
        value = -min(price_change / 20, 0.6)  # 매도 방향
        confidence = 0.5
    else:
        value = 0.0
        confidence = 0.2

    return SignalOutput(
        name="liquidation",
        value=float(np.clip(value, -1.0, 1.0)),
        confidence=confidence,
        details={
            "price_change_24h": round(price_change, 2),
            "interpretation": (
                "post_long_liquidation_bounce" if price_change < -5
                else "post_short_squeeze_pullback" if price_change > 5
                else "normal"
            ),
        },
    )
