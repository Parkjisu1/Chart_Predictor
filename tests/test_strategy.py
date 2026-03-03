"""Tests for strategy indicators and signals."""

import numpy as np
import pandas as pd
import pytest

from strategy.signals import StrategyParameters, SignalOutput, CompositeSignal
from strategy.technical.rsi import compute_rsi, detect_divergence, analyze_rsi
from strategy.technical.bollinger import compute_bollinger_bands, analyze_bollinger
from strategy.technical.volume import compute_obv, compute_vwap, analyze_volume
from strategy.technical.momentum import compute_macd, compute_adx, analyze_momentum
from strategy.technical.composite import compute_composite_signal


def make_ohlcv_df(n: int = 200, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    base = 50000
    if trend == "up":
        prices = base + np.cumsum(np.random.randn(n) * 100 + 5)
    elif trend == "down":
        prices = base + np.cumsum(np.random.randn(n) * 100 - 5)
    else:
        prices = base + np.cumsum(np.random.randn(n) * 100)

    df = pd.DataFrame({
        "open": prices + np.random.randn(n) * 50,
        "high": prices + abs(np.random.randn(n) * 100),
        "low": prices - abs(np.random.randn(n) * 100),
        "close": prices,
        "volume": np.random.uniform(100, 1000, n),
    })
    return df


class TestStrategyParameters:
    def test_defaults(self):
        params = StrategyParameters()
        assert params.rsi_period == 14
        assert params.bb_period == 20
        # 12개 가중치 합계 = 1.0
        total = (params.weight_rsi + params.weight_bollinger +
                 params.weight_volume + params.weight_garch +
                 params.weight_momentum + params.weight_sentiment +
                 params.weight_williams + params.weight_elder +
                 params.weight_ichimoku + params.weight_market_structure +
                 params.weight_patterns + params.weight_quant)
        assert 0.99 < total < 1.01

    def test_json_roundtrip(self):
        params = StrategyParameters(rsi_period=21)
        json_str = params.to_json()
        restored = StrategyParameters.from_json(json_str)
        assert restored.rsi_period == 21

    def test_normalize_weights(self):
        params = StrategyParameters(weight_rsi=1.0, weight_bollinger=1.0,
                                     weight_volume=1.0, weight_garch=1.0,
                                     weight_momentum=1.0, weight_sentiment=1.0)
        params.normalize_weights()
        weight_names = [
            "weight_rsi", "weight_bollinger", "weight_volume",
            "weight_garch", "weight_momentum", "weight_sentiment",
            "weight_williams", "weight_elder", "weight_ichimoku",
            "weight_market_structure", "weight_patterns", "weight_quant",
        ]
        total = sum(getattr(params, w) for w in weight_names)
        assert abs(total - 1.0) < 0.001

    def test_clone(self):
        params = StrategyParameters(rsi_period=21)
        clone = params.clone()
        assert clone.rsi_period == 21
        clone.rsi_period = 7
        assert params.rsi_period == 21  # Original unchanged


class TestRSI:
    def test_compute_rsi(self):
        df = make_ohlcv_df(200)
        rsi = compute_rsi(df["close"], 14)
        assert len(rsi) == 200
        valid_rsi = rsi.dropna()
        assert all(0 <= v <= 100 for v in valid_rsi)

    def test_analyze_rsi(self):
        df = make_ohlcv_df(200)
        params = StrategyParameters()
        result = analyze_rsi(df, params)
        assert isinstance(result, SignalOutput)
        assert result.name == "rsi"
        assert -1.0 <= result.value <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_divergence_detection(self):
        df = make_ohlcv_df(200)
        rsi = compute_rsi(df["close"], 14)
        div = detect_divergence(df["close"], rsi, 20)
        assert div in ("bullish", "bearish", "none")


class TestBollinger:
    def test_compute_bands(self):
        df = make_ohlcv_df(200)
        mid, upper, lower, pct_b = compute_bollinger_bands(df["close"], 20, 2.0)
        assert len(mid) == 200
        valid = upper.dropna()
        assert all(upper.dropna() >= lower.dropna())

    def test_analyze_bollinger(self):
        df = make_ohlcv_df(200)
        params = StrategyParameters()
        result = analyze_bollinger(df, params)
        assert isinstance(result, SignalOutput)
        assert -1.0 <= result.value <= 1.0


class TestVolume:
    def test_compute_obv(self):
        df = make_ohlcv_df(200)
        obv = compute_obv(df["close"], df["volume"])
        assert len(obv) == 200

    def test_analyze_volume(self):
        df = make_ohlcv_df(200)
        params = StrategyParameters()
        result = analyze_volume(df, params)
        assert isinstance(result, SignalOutput)
        assert -1.0 <= result.value <= 1.0


class TestMomentum:
    def test_compute_macd(self):
        df = make_ohlcv_df(200)
        macd_line, signal_line, histogram = compute_macd(df["close"])
        assert len(macd_line) == 200

    def test_compute_adx(self):
        df = make_ohlcv_df(200)
        adx = compute_adx(df, 14)
        valid = adx.dropna()
        assert all(v >= 0 for v in valid)

    def test_analyze_momentum(self):
        df = make_ohlcv_df(200)
        params = StrategyParameters()
        result = analyze_momentum(df, params)
        assert isinstance(result, SignalOutput)
        assert -1.0 <= result.value <= 1.0


class TestComposite:
    def test_composite_signal(self):
        df = make_ohlcv_df(200)
        params = StrategyParameters()
        result = compute_composite_signal(df, params)
        assert isinstance(result, CompositeSignal)
        assert -1.0 <= result.score <= 1.0
        assert result.signal in ("strong_long", "long", "neutral", "short", "strong_short")
        assert len(result.components) >= 10  # 5 기존 + 5 서적

    def test_composite_with_sentiment(self):
        df = make_ohlcv_df(200)
        params = StrategyParameters()
        sentiment = SignalOutput(name="sentiment", value=0.5, confidence=0.7)
        result = compute_composite_signal(df, params, sentiment)
        assert len(result.components) >= 11
