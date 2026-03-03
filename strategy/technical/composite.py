"""Enhanced composite signal - 기존 + 서적 전략 + 퀀트 데이터 통합.

12개 신호 소스:
- 기존 5개: RSI, Bollinger, Volume, GARCH, Momentum
- 서적 5개: Williams, Elder, Ichimoku, Market Structure, Patterns
- 퀀트 1개: 퀀트 종합 (OI, 롱숏, 공포탐욕, 고래)
- 센티먼트 1개: Claude CLI (선택적)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.constants import Signal
from strategy.signals import StrategyParameters, SignalOutput, CompositeSignal

# 기존 지표
from strategy.technical.rsi import analyze_rsi
from strategy.technical.bollinger import analyze_bollinger
from strategy.technical.volume import analyze_volume
from strategy.technical.garch import analyze_garch
from strategy.technical.momentum import analyze_momentum

# 서적 기반 전략
from strategy.technical.williams import analyze_williams
from strategy.technical.elder import analyze_elder
from strategy.technical.ichimoku import analyze_ichimoku
from strategy.technical.market_structure import analyze_market_structure
from strategy.technical.patterns import analyze_patterns


def compute_composite_signal(
    df: pd.DataFrame,
    params: StrategyParameters,
    sentiment_signal: SignalOutput | None = None,
    quant_signals: list[SignalOutput] | None = None,
) -> CompositeSignal:
    """12개 소스를 가중 합산하여 최종 신호 생성."""
    params.normalize_weights()

    # --- 기존 기술지표 (5개) ---
    rsi_out = analyze_rsi(df, params)
    bb_out = analyze_bollinger(df, params)
    vol_out = analyze_volume(df, params)
    garch_out = analyze_garch(df, params)
    mom_out = analyze_momentum(df, params)

    # --- 서적 기반 전략 (5개) ---
    williams_out = analyze_williams(df, params)
    elder_out = analyze_elder(df, params)
    ichimoku_out = analyze_ichimoku(df, params)
    structure_out = analyze_market_structure(df, params)
    patterns_out = analyze_patterns(df, params)

    components = [
        rsi_out, bb_out, vol_out, garch_out, mom_out,
        williams_out, elder_out, ichimoku_out, structure_out, patterns_out,
    ]

    weights = {
        "rsi": params.weight_rsi,
        "bollinger": params.weight_bollinger,
        "volume": params.weight_volume,
        "garch": params.weight_garch,
        "momentum": params.weight_momentum,
        "williams": params.weight_williams,
        "elder": params.weight_elder,
        "ichimoku": params.weight_ichimoku,
        "market_structure": params.weight_market_structure,
        "patterns": params.weight_patterns,
    }

    # GARCH 변동성 체제에 따른 신뢰도 조정
    vol_regime = garch_out.details.get("regime", "normal")
    confidence_modifier = 1.0
    if vol_regime == "high":
        confidence_modifier = 0.7
    elif vol_regime == "low":
        confidence_modifier = 1.1

    # 가중 합산
    weighted_sum = 0.0
    total_weight = 0.0
    for comp in components:
        w = weights.get(comp.name, 0)
        adjusted_confidence = min(comp.confidence * confidence_modifier, 1.0)
        weighted_sum += comp.value * adjusted_confidence * w
        total_weight += w * adjusted_confidence

    # 센티먼트 (Claude CLI)
    if sentiment_signal and sentiment_signal.confidence > 0:
        components.append(sentiment_signal)
        s_w = params.weight_sentiment
        weighted_sum += sentiment_signal.value * sentiment_signal.confidence * s_w
        total_weight += s_w * sentiment_signal.confidence

    # 퀀트 데이터 (OI, 롱숏, 공포탐욕, 고래 등)
    if quant_signals:
        quant_combined = _combine_quant_signals(quant_signals)
        components.append(quant_combined)
        q_w = params.weight_quant
        weighted_sum += quant_combined.value * quant_combined.confidence * q_w
        total_weight += q_w * quant_combined.confidence

    score = weighted_sum / total_weight if total_weight > 0 else 0.0
    score = max(-1.0, min(1.0, score))

    # --- 신호 합의도 체크 (다수결 강화) ---
    bullish_count = sum(1 for c in components if c.value > 0.1)
    bearish_count = sum(1 for c in components if c.value < -0.1)
    total_indicators = len(components)
    consensus = max(bullish_count, bearish_count) / total_indicators if total_indicators > 0 else 0

    # 합의도가 높을수록 신뢰도 상승
    if consensus > 0.7:
        score *= 1.15  # 70% 이상 합의 → 신호 강화

    score = max(-1.0, min(1.0, score))

    # 신호 레벨 결정
    if score >= params.strong_signal_threshold:
        signal = Signal.STRONG_LONG
    elif score >= params.signal_threshold:
        signal = Signal.LONG
    elif score <= -params.strong_signal_threshold:
        signal = Signal.STRONG_SHORT
    elif score <= -params.signal_threshold:
        signal = Signal.SHORT
    else:
        signal = Signal.NEUTRAL

    # 종합 신뢰도
    avg_confidence = sum(c.confidence for c in components) / len(components)
    overall_confidence = avg_confidence * abs(score) * (1 + consensus * 0.3)

    return CompositeSignal(
        score=round(score, 4),
        signal=signal.value,
        confidence=round(min(overall_confidence, 1.0), 4),
        components=components,
        metadata={
            "vol_regime": vol_regime,
            "confidence_modifier": confidence_modifier,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "consensus": round(consensus, 2),
            "total_indicators": total_indicators,
        },
    )


def _combine_quant_signals(quant_signals: list[SignalOutput]) -> SignalOutput:
    """퀀트 신호들을 하나로 합산."""
    if not quant_signals:
        return SignalOutput(name="quant", value=0.0, confidence=0.0)

    total_value = 0.0
    total_weight = 0.0
    all_details = {}

    for sig in quant_signals:
        w = sig.confidence
        total_value += sig.value * w
        total_weight += w
        all_details[sig.name] = {
            "value": round(sig.value, 4),
            "confidence": round(sig.confidence, 4),
        }

    combined_value = total_value / total_weight if total_weight > 0 else 0.0
    avg_confidence = total_weight / len(quant_signals) if quant_signals else 0.0

    return SignalOutput(
        name="quant",
        value=max(-1.0, min(1.0, combined_value)),
        confidence=min(avg_confidence, 1.0),
        details=all_details,
    )
