"""고래 활동 감지 - 대량 거래, 이상 거래량.

거래량에서 고래 활동을 추론:
- 이상 거래량 급등: 고래 진입/청산 가능성
- 가격 미변동 + 대량 거래: 축적 또는 분배
- 가격 급등 + 대량 거래: 추세 확인
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals import SignalOutput


def detect_whale_activity(
    df: pd.DataFrame, lookback: int = 20, threshold: float = 3.0
) -> SignalOutput:
    """거래량 기반 고래 활동 감지."""
    if len(df) < lookback + 5:
        return SignalOutput(name="whale", value=0.0, confidence=0.0)

    volume = df["volume"]
    close = df["close"]

    vol_ma = volume.rolling(lookback).mean()
    vol_std = volume.rolling(lookback).std()
    current_vol = volume.iloc[-1]
    avg_vol = vol_ma.iloc[-1]
    std_vol = vol_std.iloc[-1]

    if avg_vol == 0 or np.isnan(avg_vol):
        return SignalOutput(name="whale", value=0.0, confidence=0.0)

    # Z-score
    z_score = (current_vol - avg_vol) / (std_vol + 1e-10)
    is_whale = z_score > threshold

    # 가격 변화
    price_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]

    value = 0.0
    confidence = 0.2

    if is_whale:
        confidence = 0.6
        if price_change > 0.01:
            # 대량 거래 + 가격 상승 → 고래 매수
            value = min(z_score / 10, 0.6)
        elif price_change < -0.01:
            # 대량 거래 + 가격 하락 → 고래 매도
            value = max(-z_score / 10, -0.6)
        else:
            # 대량 거래 + 가격 미변동 → 축적/분배 (방향 불명)
            value = 0.0
            confidence = 0.4

    # 연속 이상 거래량 (3봉 연속) → 더 강한 신호
    recent_z = [(volume.iloc[-i] - avg_vol) / (std_vol + 1e-10) for i in range(1, 4)]
    consecutive_whale = sum(1 for z in recent_z if z > threshold * 0.7)
    if consecutive_whale >= 2:
        confidence = min(confidence + 0.15, 0.85)
        value *= 1.2

    return SignalOutput(
        name="whale",
        value=float(np.clip(value, -1.0, 1.0)),
        confidence=confidence,
        details={
            "volume_z_score": round(float(z_score), 2),
            "is_whale_activity": is_whale,
            "volume_ratio": round(float(current_vol / avg_vol), 2) if avg_vol > 0 else 0,
            "price_change_pct": round(float(price_change * 100), 3),
            "consecutive_spikes": consecutive_whale,
            "interpretation": (
                "whale_buying" if is_whale and price_change > 0.01
                else "whale_selling" if is_whale and price_change < -0.01
                else "whale_accumulation" if is_whale
                else "normal"
            ),
        },
    )
