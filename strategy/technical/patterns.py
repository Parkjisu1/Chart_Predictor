"""차트 패턴 인식 - VCP, Wyckoff, Stage Analysis.

출처:
- Mark Minervini: "Trade Like a Stock Market Wizard" → VCP 패턴
- Stan Weinstein: "Secrets for Profiting in Bull and Bear Markets" → 4단계 분석
- Richard Wyckoff: Wyckoff Method → 축적/분배 감지
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.signals import StrategyParameters, SignalOutput


def detect_vcp(
    df: pd.DataFrame, contractions: int = 3, min_contraction_pct: float = 0.3
) -> dict:
    """VCP (Volatility Contraction Pattern) 감지 - Mark Minervini.

    연속적인 변동성 수축 → 브레이크아웃 직전 패턴.
    각 수축의 범위가 이전 대비 줄어들어야 함.
    """
    if len(df) < 60:
        return {"detected": False}

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # 최근 60봉에서 스윙 범위 측정
    windows = [20, 15, 10, 8]
    ranges_pct = []

    for i, w in enumerate(windows):
        if len(df) < sum(windows[:i + 1]):
            break
        start = -sum(windows[:i + 1])
        end = -sum(windows[:i]) if i > 0 else None
        segment = df.iloc[start:end] if end else df.iloc[start:]
        if segment.empty:
            continue
        seg_range = (segment["high"].max() - segment["low"].min()) / segment["close"].mean()
        ranges_pct.append(seg_range)

    if len(ranges_pct) < 3:
        return {"detected": False}

    # 연속 수축 확인
    contracting = all(
        ranges_pct[i] > ranges_pct[i + 1] * min_contraction_pct
        for i in range(len(ranges_pct) - 1)
    )

    # 최종 수축 범위가 충분히 작은지
    final_tight = ranges_pct[-1] < 0.03  # 3% 이내 타이트 통합

    detected = contracting and final_tight and len(ranges_pct) >= contractions

    return {
        "detected": detected,
        "contractions": len(ranges_pct),
        "ranges_pct": [round(r * 100, 2) for r in ranges_pct],
        "final_range_pct": round(ranges_pct[-1] * 100, 2) if ranges_pct else 0,
    }


def detect_stage(
    close: pd.Series, ma_period: int = 150
) -> dict:
    """Weinstein 4단계 분석.
    Stage 1: 바닥 형성 (MA 횡보, 가격 MA 주변)
    Stage 2: 상승 추세 (가격 > MA, MA 상승)
    Stage 3: 천장 형성 (MA 횡보, 가격 MA 주변, 거래량 증가)
    Stage 4: 하락 추세 (가격 < MA, MA 하락)
    """
    if len(close) < ma_period + 20:
        return {"stage": 0, "description": "insufficient_data"}

    ma = close.rolling(window=ma_period).mean()
    ma_slope = ma.diff(20).iloc[-1]  # 20봉 변화로 추세 판단
    price = close.iloc[-1]
    current_ma = ma.iloc[-1]

    if np.isnan(current_ma) or np.isnan(ma_slope):
        return {"stage": 0, "description": "insufficient_data"}

    price_above_ma = price > current_ma
    ma_rising = ma_slope > 0
    ma_flat = abs(ma_slope / current_ma) < 0.001  # 0.1% 이내 변화 = 횡보

    if price_above_ma and ma_rising:
        stage = 2
        desc = "uptrend"
    elif not price_above_ma and not ma_rising:
        stage = 4
        desc = "downtrend"
    elif ma_flat and abs(price - current_ma) / current_ma < 0.02:
        if close.iloc[-20:].mean() > close.iloc[-60:-20].mean():
            stage = 1
            desc = "basing"
        else:
            stage = 3
            desc = "topping"
    elif price_above_ma:
        stage = 2
        desc = "uptrend_weak"
    else:
        stage = 4
        desc = "downtrend_weak"

    return {
        "stage": stage,
        "description": desc,
        "price_vs_ma": round((price / current_ma - 1) * 100, 2),
        "ma_slope_pct": round(ma_slope / current_ma * 100, 4) if current_ma > 0 else 0,
    }


def detect_wyckoff(df: pd.DataFrame, lookback: int = 50) -> dict:
    """Wyckoff 축적/분배 감지 (간소화).

    축적: 가격 하락 둔화 + 거래량 감소 + 바닥 지지
    분배: 가격 상승 둔화 + 거래량 증가 + 천장 저항
    """
    if len(df) < lookback:
        return {"phase": "unknown"}

    recent = df.iloc[-lookback:]
    close = recent["close"]
    volume = recent["volume"]

    # 가격 추세
    price_first_half = close.iloc[:lookback // 2].mean()
    price_second_half = close.iloc[lookback // 2:].mean()
    price_trend = price_second_half / price_first_half - 1

    # 거래량 추세
    vol_first_half = volume.iloc[:lookback // 2].mean()
    vol_second_half = volume.iloc[lookback // 2:].mean()
    vol_trend = vol_second_half / vol_first_half - 1 if vol_first_half > 0 else 0

    # 가격 변동성 추세
    volatility_first = close.iloc[:lookback // 2].std() / price_first_half
    volatility_second = close.iloc[lookback // 2:].std() / price_second_half
    vol_contracting = volatility_second < volatility_first * 0.8

    # 축적: 하락 둔화 + 변동성 축소 + 거래량 감소
    if price_trend < 0.01 and price_trend > -0.05 and vol_contracting and vol_trend < 0:
        phase = "accumulation"
    # 분배: 상승 둔화 + 거래량 증가
    elif price_trend > -0.01 and price_trend < 0.05 and vol_trend > 0.2:
        phase = "distribution"
    # Spring: 축적 마지막 단계의 하방 이탈 후 반전
    elif price_trend < -0.03 and vol_contracting:
        phase = "potential_spring"
    elif price_trend > 0.05:
        phase = "markup"
    elif price_trend < -0.05:
        phase = "markdown"
    else:
        phase = "neutral"

    return {
        "phase": phase,
        "price_trend_pct": round(price_trend * 100, 2),
        "volume_trend_pct": round(vol_trend * 100, 2),
        "volatility_contracting": vol_contracting,
    }


def analyze_patterns(df: pd.DataFrame, params: StrategyParameters) -> SignalOutput:
    """패턴 종합 분석: VCP + Stage + Wyckoff."""
    close = df["close"]
    vcp = detect_vcp(df)
    stage = detect_stage(close)
    wyckoff = detect_wyckoff(df)

    value = 0.0
    confidence = 0.3

    # Stage Analysis 신호
    if stage["stage"] == 2:
        value += 0.3
        confidence = 0.55
    elif stage["stage"] == 4:
        value -= 0.3
        confidence = 0.55
    elif stage["stage"] == 1:
        value += 0.15  # 바닥 형성 → 약한 매수
    elif stage["stage"] == 3:
        value -= 0.15  # 천장 형성 → 약한 매도

    # VCP 감지 시 강력 매수 신호
    if vcp["detected"]:
        value += 0.35
        confidence = min(confidence + 0.25, 1.0)

    # Wyckoff 신호
    if wyckoff["phase"] == "accumulation":
        value += 0.2
        confidence += 0.1
    elif wyckoff["phase"] == "distribution":
        value -= 0.2
        confidence += 0.1
    elif wyckoff["phase"] == "potential_spring":
        value += 0.25
        confidence += 0.15

    return SignalOutput(
        name="patterns",
        value=max(-1.0, min(1.0, value)),
        confidence=min(confidence, 1.0),
        details={
            "vcp": vcp,
            "stage": stage,
            "wyckoff": wyckoff,
        },
    )
