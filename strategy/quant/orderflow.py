"""오더플로우 분석 - 오더북 불균형, OI 변화.

퀀트 매니저들의 핵심 데이터:
- 오더북 불균형: 매수벽/매도벽 비율 → 단기 가격 압력
- OI 변화 + 가격 방향: 추세 확인/부정
- 거래량 프로필: 가격대별 거래량 → 지지/저항 강도
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals import SignalOutput


def analyze_orderbook_imbalance(orderbook_data: dict) -> SignalOutput:
    """오더북 불균형 분석.
    매수량 >> 매도량: 단기 상승 압력
    매도량 >> 매수량: 단기 하락 압력
    """
    imbalance = orderbook_data.get("imbalance", 0)
    bid_vol = orderbook_data.get("bid_volume", 0)
    ask_vol = orderbook_data.get("ask_volume", 0)

    if bid_vol == 0 and ask_vol == 0:
        return SignalOutput(name="orderflow", value=0.0, confidence=0.0)

    # 불균형 기반 신호
    value = np.clip(imbalance * 2, -1.0, 1.0)

    # 극단적 불균형 시 높은 신뢰도
    confidence = min(abs(imbalance) * 2 + 0.3, 0.8)

    return SignalOutput(
        name="orderflow",
        value=float(value),
        confidence=confidence,
        details={
            "imbalance": imbalance,
            "bid_wall_pct": orderbook_data.get("bid_wall", 50),
            "interpretation": (
                "strong_buy_pressure" if imbalance > 0.3
                else "strong_sell_pressure" if imbalance < -0.3
                else "balanced"
            ),
        },
    )


def analyze_oi_price_divergence(
    oi_data: dict,
    price_change_pct: float,
) -> SignalOutput:
    """OI와 가격 방향 분석.

    | OI    | Price | 해석              |
    |-------|-------|-------------------|
    | 증가  | 상승  | 새 롱 진입 → 강세  |
    | 증가  | 하락  | 새 숏 진입 → 약세  |
    | 감소  | 상승  | 숏 커버 → 약한 강세 |
    | 감소  | 하락  | 롱 청산 → 약한 약세 |
    """
    oi_change = oi_data.get("change_24h_pct", 0)

    if oi_change > 5 and price_change_pct > 1:
        value = 0.5   # 새 롱 진입 → 강세
        interpretation = "new_longs_bullish"
    elif oi_change > 5 and price_change_pct < -1:
        value = -0.5  # 새 숏 진입 → 약세
        interpretation = "new_shorts_bearish"
    elif oi_change < -5 and price_change_pct > 1:
        value = 0.2   # 숏 커버 → 약한 강세
        interpretation = "short_covering_weak_bull"
    elif oi_change < -5 and price_change_pct < -1:
        value = -0.2  # 롱 청산 → 약한 약세
        interpretation = "long_liquidation_weak_bear"
    else:
        value = 0.0
        interpretation = "neutral"

    confidence = min(abs(oi_change) / 20 + 0.3, 0.7)

    return SignalOutput(
        name="oi_divergence",
        value=value,
        confidence=confidence,
        details={
            "oi_change_pct": oi_change,
            "price_change_pct": round(price_change_pct, 2),
            "interpretation": interpretation,
        },
    )
