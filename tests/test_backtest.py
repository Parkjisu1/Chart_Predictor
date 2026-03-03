"""Tests for backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from backtest.cost_model import CostModel
from backtest.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
)
from backtest.data_splitter import DataSplitter
from backtest.monte_carlo import MonteCarloSimulator
from strategy.signals import StrategyParameters


def make_df(n=500):
    np.random.seed(42)
    base = 50000
    prices = base + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        "open": prices + np.random.randn(n) * 50,
        "high": prices + abs(np.random.randn(n) * 100),
        "low": prices - abs(np.random.randn(n) * 100),
        "close": prices,
        "volume": np.random.uniform(100, 1000, n),
    })


class TestCostModel:
    def test_entry_cost(self):
        model = CostModel()
        costs = model.calculate_entry_cost(10000)
        assert costs["fee"] > 0
        assert costs["slippage"] >= 0
        assert costs["total"] > 0

    def test_round_trip(self):
        model = CostModel()
        costs = model.calculate_total_round_trip(10000, 10100)
        assert costs["total_cost"] > 0
        assert costs["cost_pct"] > 0


class TestMetrics:
    def test_sharpe_ratio(self):
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)
        sharpe = calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)

    def test_sortino_ratio(self):
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.001)
        sortino = calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)

    def test_max_drawdown(self):
        equity = [100, 110, 105, 95, 100, 115]
        dd, dd_pct = calculate_max_drawdown(equity)
        assert dd == 15  # 110 - 95
        assert abs(dd_pct - 15 / 110) < 0.001

    def test_max_drawdown_empty(self):
        dd, dd_pct = calculate_max_drawdown([])
        assert dd == 0

    def test_profit_factor(self):
        wins = [100, 200, 150]
        losses = [-50, -75, -100]
        pf = calculate_profit_factor(wins, losses)
        assert pf == 450 / 225
        assert pf == 2.0


class TestDataSplitter:
    def test_simple_split(self):
        df = make_df(1000)
        splitter = DataSplitter()
        split = splitter.simple_split(df)
        assert len(split.in_sample) == 700
        assert len(split.out_of_sample) == 300

    def test_walk_forward(self):
        df = make_df(1000)
        splitter = DataSplitter()
        splits = splitter.walk_forward_splits(df, n_folds=3)
        assert len(splits) >= 1
        for s in splits:
            assert len(s.in_sample) > 0
            assert len(s.out_of_sample) > 0


class TestMonteCarlo:
    def test_simulation(self):
        returns = list(np.random.randn(100) * 0.01)
        mc = MonteCarloSimulator(n_simulations=100)
        result = mc.simulate(returns)
        assert result.simulations == 100
        assert 0 <= result.probability_of_loss <= 1

    def test_empty_trades(self):
        mc = MonteCarloSimulator()
        result = mc.simulate([])
        assert result.simulations == 0


class TestBacktestEngine:
    def test_run_basic(self):
        df = make_df(500)
        engine = BacktestEngine(initial_capital=100000)
        result = engine.run(df, lookback=100)
        assert result.total_trades >= 0
        # Engine may produce 0 trades with synthetic data
        assert 0.0 <= result.win_rate <= 1.0
        # Internal equity curve always has entries
        assert len(engine.equity_curve) > 0

    def test_parameter_update(self):
        engine = BacktestEngine()
        new_params = StrategyParameters(rsi_period=21)
        engine.update_parameters(new_params)
        assert engine.params.rsi_period == 21
