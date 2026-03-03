"""서적 기반 전략 + 퀀트 지표 테스트."""

import numpy as np
import pandas as pd
import pytest

from strategy.signals import StrategyParameters, SignalOutput
from strategy.technical.williams import compute_williams_r, detect_large_range_day, analyze_williams
from strategy.technical.elder import compute_force_index, compute_elder_ray, analyze_elder
from strategy.technical.ichimoku import compute_ichimoku, analyze_ichimoku
from strategy.technical.market_structure import (
    find_pivot_points, compute_fibonacci_levels, detect_market_structure, analyze_market_structure
)
from strategy.technical.patterns import detect_vcp, detect_stage, detect_wyckoff, analyze_patterns
from strategy.technical.composite import compute_composite_signal
from strategy.quant.market_sentiment import analyze_fear_greed, analyze_long_short_ratio
from strategy.quant.orderflow import analyze_orderbook_imbalance, analyze_oi_price_divergence
from strategy.quant.whale_detection import detect_whale_activity


def make_df(n=200, trend="up"):
    np.random.seed(42)
    base = 50000
    if trend == "up":
        prices = base + np.cumsum(np.random.randn(n) * 100 + 5)
    elif trend == "down":
        prices = base + np.cumsum(np.random.randn(n) * 100 - 5)
    else:
        prices = base + np.cumsum(np.random.randn(n) * 50)
    return pd.DataFrame({
        "open": prices + np.random.randn(n) * 50,
        "high": prices + abs(np.random.randn(n) * 100),
        "low": prices - abs(np.random.randn(n) * 100),
        "close": prices,
        "volume": np.random.uniform(100, 1000, n),
    })


class TestWilliams:
    def test_williams_r_range(self):
        df = make_df(200)
        wr = compute_williams_r(df["high"], df["low"], df["close"])
        valid = wr.dropna()
        assert all(-100 <= v <= 0 for v in valid)

    def test_large_range_day(self):
        df = make_df(200)
        lrd = detect_large_range_day(df)
        assert "detected" in lrd
        assert "direction" in lrd

    def test_analyze_williams(self):
        df = make_df(200)
        result = analyze_williams(df, StrategyParameters())
        assert -1.0 <= result.value <= 1.0
        assert result.name == "williams"


class TestElder:
    def test_force_index(self):
        df = make_df(200)
        fi = compute_force_index(df["close"], df["volume"])
        assert len(fi) == 200

    def test_elder_ray(self):
        df = make_df(200)
        bull, bear = compute_elder_ray(df["high"], df["low"], df["close"])
        assert len(bull) == 200

    def test_analyze_elder(self):
        df = make_df(200)
        result = analyze_elder(df, StrategyParameters())
        assert -1.0 <= result.value <= 1.0
        assert "impulse" in result.details


class TestIchimoku:
    def test_compute_ichimoku(self):
        df = make_df(200)
        ich = compute_ichimoku(df["high"], df["low"], df["close"])
        assert "tenkan" in ich
        assert "kijun" in ich
        assert "senkou_a" in ich

    def test_analyze_ichimoku(self):
        df = make_df(200)
        result = analyze_ichimoku(df, StrategyParameters())
        assert -1.0 <= result.value <= 1.0
        assert "price_vs_cloud" in result.details


class TestMarketStructure:
    def test_pivot_points(self):
        df = make_df(200)
        res, sup = find_pivot_points(df["high"], df["low"])
        assert isinstance(res, list)
        assert isinstance(sup, list)

    def test_fibonacci(self):
        fib = compute_fibonacci_levels(55000, 45000)
        assert fib["0.382"] == pytest.approx(55000 - 10000 * 0.382, abs=1)
        assert fib["0.618"] == pytest.approx(55000 - 10000 * 0.618, abs=1)

    def test_market_structure(self):
        df = make_df(200, trend="up")
        result = detect_market_structure(df["high"], df["low"])
        assert "structure" in result
        assert "trend" in result

    def test_analyze(self):
        df = make_df(200)
        result = analyze_market_structure(df, StrategyParameters())
        assert -1.0 <= result.value <= 1.0


class TestPatterns:
    def test_stage_detection(self):
        df = make_df(300, trend="up")
        stage = detect_stage(df["close"], ma_period=150)
        assert stage["stage"] in [0, 1, 2, 3, 4]

    def test_wyckoff(self):
        df = make_df(200)
        wyckoff = detect_wyckoff(df)
        assert "phase" in wyckoff

    def test_analyze_patterns(self):
        df = make_df(200)
        result = analyze_patterns(df, StrategyParameters())
        assert -1.0 <= result.value <= 1.0
        assert "vcp" in result.details


class TestQuantSignals:
    def test_fear_greed(self):
        result = analyze_fear_greed({"value": 15, "label": "Extreme Fear"})
        assert result.value > 0  # 극단적 공포 → 매수 신호 (역추세)
        assert result.confidence > 0.5

    def test_fear_greed_extreme_greed(self):
        result = analyze_fear_greed({"value": 90, "label": "Extreme Greed"})
        assert result.value < 0  # 극단적 탐욕 → 매도 신호

    def test_long_short_ratio_long_heavy(self):
        result = analyze_long_short_ratio({"ratio": 2.5, "extreme": True,
                                            "long_pct": 71, "short_pct": 29})
        assert result.value < 0  # 롱 과밀 → 역추세 매도

    def test_orderbook_imbalance(self):
        result = analyze_orderbook_imbalance({
            "imbalance": 0.4, "bid_volume": 70, "ask_volume": 30, "bid_wall": 70,
        })
        assert result.value > 0  # 매수 압력 우세

    def test_oi_price_divergence(self):
        result = analyze_oi_price_divergence(
            {"change_24h_pct": 10}, price_change_pct=3
        )
        assert result.value > 0  # OI증가 + 가격상승 → 강세

    def test_whale_detection(self):
        df = make_df(100)
        result = detect_whale_activity(df)
        assert -1.0 <= result.value <= 1.0


class TestEnhancedComposite:
    def test_composite_with_all(self):
        df = make_df(200)
        params = StrategyParameters()
        result = compute_composite_signal(df, params)
        assert -1.0 <= result.score <= 1.0
        # 최소 10개 컴포넌트 (기존 5 + 서적 5)
        assert len(result.components) >= 10
        assert "consensus" in result.metadata

    def test_composite_with_quant(self):
        df = make_df(200)
        params = StrategyParameters()
        quant = [
            SignalOutput(name="fear_greed", value=0.5, confidence=0.7),
            SignalOutput(name="orderflow", value=0.3, confidence=0.5),
        ]
        result = compute_composite_signal(df, params, quant_signals=quant)
        assert len(result.components) >= 11  # 10 + quant combined

    def test_consensus_metric(self):
        df = make_df(200, trend="up")
        params = StrategyParameters()
        result = compute_composite_signal(df, params)
        assert 0 <= result.metadata["consensus"] <= 1.0
