"""Tests for learning loop components."""

import pytest

from strategy.signals import StrategyParameters
from learning.parameter_tuner import ParameterTuner
from learning.iteration_tracker import IterationTracker, IterationRecord
from learning.trade_analyzer import TradeAnalyzer
from data.models import Trade


class TestParameterTuner:
    def test_apply_adjustments(self):
        tuner = ParameterTuner()
        params = StrategyParameters()
        original_rsi = params.rsi_period

        adjusted = tuner.apply_adjustments(params, {
            "rsi_period": 21,
        })
        assert adjusted.rsi_period == 21
        assert params.rsi_period == original_rsi  # Original unchanged

    def test_directive_parsing(self):
        tuner = ParameterTuner()
        params = StrategyParameters(signal_threshold=0.3)
        adjusted = tuner.apply_adjustments(params, {
            "signal_threshold": "increase_10pct",
        })
        assert adjusted.signal_threshold > 0.3
        assert abs(adjusted.signal_threshold - 0.33) < 0.01

    def test_boundary_enforcement(self):
        tuner = ParameterTuner()
        params = StrategyParameters()
        adjusted = tuner.apply_adjustments(params, {
            "rsi_period": 100,  # Way beyond boundary
        })
        assert adjusted.rsi_period <= 28  # Max boundary

    def test_random_perturbation(self):
        tuner = ParameterTuner()
        params = StrategyParameters()
        perturbed = tuner.random_perturbation(params)
        # At least one parameter should differ
        changed = False
        for key in params.BOUNDARIES:
            if hasattr(params, key) and hasattr(perturbed, key):
                if getattr(params, key) != getattr(perturbed, key):
                    changed = True
                    break
        assert changed


class TestIterationTracker:
    def test_convergence_detection(self):
        tracker = IterationTracker(target_win_rate=0.90)
        tracker.record(IterationRecord(1, 0.92, 1.5, 2.0, 0.05, 1000, {}))
        assert tracker.has_converged()

    def test_stagnation_detection(self):
        tracker = IterationTracker(stagnation_limit=3)
        tracker.record(IterationRecord(1, 0.60, 1.0, 1.5, 0.1, 500, {}))
        tracker.record(IterationRecord(2, 0.55, 1.0, 1.5, 0.1, 500, {}))
        tracker.record(IterationRecord(3, 0.58, 1.0, 1.5, 0.1, 500, {}))
        tracker.record(IterationRecord(4, 0.57, 1.0, 1.5, 0.1, 500, {}))
        assert tracker.is_stagnant()

    def test_max_iterations(self):
        tracker = IterationTracker(max_iterations=3)
        for i in range(3):
            tracker.record(IterationRecord(i, 0.5, 1.0, 1.5, 0.1, 100, {}))
        should_stop, reason = tracker.should_stop()
        assert should_stop
        assert reason == "max_iterations"

    def test_summary(self):
        tracker = IterationTracker()
        tracker.record(IterationRecord(1, 0.55, 1.0, 1.5, 0.1, 100, {}))
        tracker.record(IterationRecord(2, 0.60, 1.2, 1.6, 0.08, 120, {}))
        summary = tracker.get_summary()
        assert summary["iterations"] == 2
        assert summary["best_win_rate"] == 0.60


class TestTradeAnalyzer:
    def test_classify_winning_trade(self):
        analyzer = TradeAnalyzer()
        trade = Trade(symbol="BTC", side="long", entry_price=50000,
                     quantity=0.1, pnl=100, pnl_pct=0.02)
        mode = analyzer.classify_trade(trade)
        assert mode == ""  # Not a loss

    def test_classify_wrong_direction(self):
        analyzer = TradeAnalyzer()
        trade = Trade(symbol="BTC", side="long", entry_price=50000,
                     exit_price=47000, quantity=0.1, pnl=-300,
                     pnl_pct=-0.06, stop_loss=47000, take_profit=53000)
        mode = analyzer.classify_trade(trade)
        assert mode == "wrong_direction"

    def test_analyze_trades(self):
        analyzer = TradeAnalyzer()
        trades = [
            Trade(symbol="BTC", side="long", entry_price=50000,
                  quantity=0.1, pnl=100, pnl_pct=0.02),
            Trade(symbol="BTC", side="long", entry_price=50000,
                  exit_price=47000, quantity=0.1, pnl=-300,
                  pnl_pct=-0.06, stop_loss=47000),
            Trade(symbol="BTC", side="short", entry_price=50000,
                  exit_price=49500, quantity=0.1, pnl=-50,
                  pnl_pct=-0.005, stop_loss=51000),
        ]
        result = analyzer.analyze_trades(trades)
        assert result["total_losses"] == 2
        assert "failure_modes" in result
