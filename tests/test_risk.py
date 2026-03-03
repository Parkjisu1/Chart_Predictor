"""Tests for risk management modules."""

import numpy as np
import pandas as pd
import pytest

from risk.position_sizer import PositionSizer
from risk.kill_switch import KillSwitch
from risk.correlation import CorrelationAnalyzer
from risk.cvar import CVaRCalculator
from risk.slippage import SlippageModel
from risk.limits import TradingLimits


class TestPositionSizer:
    def test_basic_sizing(self):
        sizer = PositionSizer()
        result = sizer.calculate(
            capital=100000,
            price=50000,
            signal_confidence=0.7,
            kelly_pct=0.15,
        )
        assert result.size_pct == 0.15
        assert result.quantity > 0

    def test_respects_max(self):
        sizer = PositionSizer(max_position_pct=0.10)
        result = sizer.calculate(
            capital=100000,
            price=50000,
            signal_confidence=0.7,
            kelly_pct=0.20,
        )
        assert result.size_pct <= 0.10

    def test_zero_price(self):
        sizer = PositionSizer()
        result = sizer.calculate(capital=100000, price=0,
                                  signal_confidence=0.7, kelly_pct=0.1)
        assert result.quantity == 0


class TestKillSwitch:
    def test_normal_conditions(self):
        ks = KillSwitch()
        status = ks.check(
            daily_pnl_pct=-0.02,
            total_drawdown_pct=0.05,
            equity=95000,
            initial_capital=100000,
        )
        assert not status.triggered

    def test_daily_loss_trigger(self):
        ks = KillSwitch(max_daily_loss=0.05)
        status = ks.check(
            daily_pnl_pct=-0.06,
            total_drawdown_pct=0.02,
            equity=94000,
            initial_capital=100000,
        )
        assert status.triggered

    def test_drawdown_trigger(self):
        ks = KillSwitch(max_drawdown=0.15)
        status = ks.check(
            daily_pnl_pct=-0.01,
            total_drawdown_pct=0.20,
            equity=80000,
            initial_capital=100000,
        )
        assert status.triggered

    def test_reset(self):
        ks = KillSwitch()
        ks.check(daily_pnl_pct=-0.10, total_drawdown_pct=0.20,
                 equity=50000, initial_capital=100000)
        assert ks.is_active
        ks.reset()
        assert not ks.is_active


class TestCorrelation:
    def test_high_correlation(self):
        np.random.seed(42)
        base = np.random.randn(100)
        a = pd.Series(base + np.random.randn(100) * 0.1)
        b = pd.Series(base + np.random.randn(100) * 0.1)
        analyzer = CorrelationAnalyzer(threshold=0.7)
        result = analyzer.check_pair_correlation(a, b)
        assert result["too_high"]

    def test_low_correlation(self):
        np.random.seed(42)
        a = pd.Series(np.random.randn(100))
        b = pd.Series(np.random.randn(100))
        analyzer = CorrelationAnalyzer(threshold=0.7)
        result = analyzer.check_pair_correlation(a, b)
        assert not result["too_high"]


class TestCVaR:
    def test_var_cvar(self):
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)
        calc = CVaRCalculator(confidence_level=0.95)
        risk = calc.assess_risk(returns, position_value=10000)
        assert risk["var_pct"] > 0
        assert risk["cvar_pct"] >= risk["var_pct"]


class TestSlippage:
    def test_estimate(self):
        model = SlippageModel()
        bps = model.estimate(1000, 1_000_000_000)
        assert bps > 0

    def test_large_order_more_slippage(self):
        model = SlippageModel()
        small = model.estimate(1000, 1_000_000_000)
        large = model.estimate(1_000_000, 1_000_000_000)
        assert large > small


class TestTradingLimits:
    def test_all_clear(self):
        limits = TradingLimits()
        check = limits.check_new_trade(
            position_pct=0.10,
            current_exposure_pct=0.20,
            daily_pnl_pct=-0.01,
            drawdown_pct=0.05,
            open_positions=1,
            leverage=3,
        )
        assert check.passed

    def test_exceeds_position(self):
        limits = TradingLimits(max_position_pct=0.10)
        check = limits.check_new_trade(
            position_pct=0.15,
            current_exposure_pct=0.0,
            daily_pnl_pct=0.0,
            drawdown_pct=0.0,
            open_positions=0,
            leverage=1,
        )
        assert not check.passed
        assert len(check.violations) > 0
