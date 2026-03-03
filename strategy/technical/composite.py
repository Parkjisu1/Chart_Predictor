"""Composite signal aggregation from all technical indicators."""

from __future__ import annotations

import pandas as pd

from config.constants import Signal
from strategy.signals import StrategyParameters, SignalOutput, CompositeSignal
from strategy.technical.rsi import analyze_rsi
from strategy.technical.bollinger import analyze_bollinger
from strategy.technical.volume import analyze_volume
from strategy.technical.garch import analyze_garch
from strategy.technical.momentum import analyze_momentum


def compute_composite_signal(
    df: pd.DataFrame,
    params: StrategyParameters,
    sentiment_signal: SignalOutput | None = None,
) -> CompositeSignal:
    """Compute weighted composite signal from all indicators."""
    params.normalize_weights()

    rsi_out = analyze_rsi(df, params)
    bb_out = analyze_bollinger(df, params)
    vol_out = analyze_volume(df, params)
    garch_out = analyze_garch(df, params)
    mom_out = analyze_momentum(df, params)

    components = [rsi_out, bb_out, vol_out, garch_out, mom_out]

    weights = {
        "rsi": params.weight_rsi,
        "bollinger": params.weight_bollinger,
        "volume": params.weight_volume,
        "garch": params.weight_garch,
        "momentum": params.weight_momentum,
    }

    # GARCH modifies confidence of other signals
    vol_regime = garch_out.details.get("regime", "normal")
    confidence_modifier = 1.0
    if vol_regime == "high":
        confidence_modifier = 0.7
    elif vol_regime == "low":
        confidence_modifier = 1.1

    # Weighted score
    weighted_sum = 0.0
    total_weight = 0.0
    for comp in components:
        w = weights.get(comp.name, 0)
        adjusted_confidence = min(comp.confidence * confidence_modifier, 1.0)
        weighted_sum += comp.value * adjusted_confidence * w
        total_weight += w * adjusted_confidence

    # Add sentiment if provided
    if sentiment_signal:
        components.append(sentiment_signal)
        s_w = params.weight_sentiment
        weighted_sum += sentiment_signal.value * sentiment_signal.confidence * s_w
        total_weight += s_w * sentiment_signal.confidence

    score = weighted_sum / total_weight if total_weight > 0 else 0.0
    score = max(-1.0, min(1.0, score))

    # Determine signal level
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

    # Overall confidence
    avg_confidence = sum(c.confidence for c in components) / len(components)
    overall_confidence = avg_confidence * abs(score)

    return CompositeSignal(
        score=round(score, 4),
        signal=signal.value,
        confidence=round(min(overall_confidence, 1.0), 4),
        components=components,
        metadata={
            "vol_regime": vol_regime,
            "confidence_modifier": confidence_modifier,
        },
    )
