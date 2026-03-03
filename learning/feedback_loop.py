"""Main self-learning feedback loop orchestrator."""

from __future__ import annotations

import json

import pandas as pd

from config.constants import TARGET_WIN_RATE, OOS_MIN_WIN_RATE
from config.logging_config import get_logger
from data.database import Database
from data.models import StrategyParametersRecord
from strategy.signals import StrategyParameters
from backtest.engine import BacktestEngine
from backtest.data_splitter import DataSplitter
from backtest.monte_carlo import MonteCarloSimulator
from backtest.report import ReportGenerator
from learning.trade_analyzer import TradeAnalyzer
from learning.claude_insights import ClaudeInsightsEngine
from learning.parameter_tuner import ParameterTuner
from learning.iteration_tracker import IterationTracker, IterationRecord

logger = get_logger(__name__)


class FeedbackLoop:
    """Main orchestrator for self-learning feedback loop.

    Flow:
    1. Backtest with current params (in-sample)
    2. Analyze losses (8 failure modes)
    3. Get Claude insights OR rule-based adjustments
    4. Tune parameters (within boundaries)
    5. Validate on out-of-sample
    6. Check convergence
    7. Repeat until target or max iterations
    """

    def __init__(
        self,
        db: Database | None = None,
        initial_params: StrategyParameters | None = None,
        initial_capital: float = 100_000,
        use_claude: bool = True,
    ):
        self.db = db or Database()
        self.params = initial_params or StrategyParameters()
        self.initial_capital = initial_capital

        self.engine = BacktestEngine(
            params=self.params,
            initial_capital=initial_capital,
        )
        self.splitter = DataSplitter()
        self.analyzer = TradeAnalyzer()
        self.insights = ClaudeInsightsEngine(enabled=use_claude)
        self.tuner = ParameterTuner()
        self.tracker = IterationTracker()
        self.mc = MonteCarloSimulator()
        self.reporter = ReportGenerator()

        self.best_params = self.params.clone()
        self.best_win_rate = 0.0

    def run(
        self,
        df: pd.DataFrame,
        max_iterations: int | None = None,
    ) -> dict:
        """Run the full feedback loop."""
        if max_iterations:
            self.tracker.max_iterations = max_iterations

        # Split data
        split = self.splitter.simple_split(df)

        logger.info("feedback_loop_start",
                     in_sample_rows=len(split.in_sample),
                     oos_rows=len(split.out_of_sample))

        iteration = 0
        while True:
            iteration += 1
            should_stop, reason = self.tracker.should_stop()
            if should_stop:
                logger.info("feedback_loop_stop", reason=reason,
                           iteration=iteration)
                break

            logger.info("iteration_start", iteration=iteration)

            # Step 1: Backtest on in-sample
            self.engine.update_parameters(self.params)
            is_result = self.engine.run(split.in_sample)

            # Step 2: Analyze losses
            analysis = self.analyzer.analyze_trades(is_result.trades)

            # Step 3: Get insights
            key_params = {
                "signal_threshold": self.params.signal_threshold,
                "strong_signal_threshold": self.params.strong_signal_threshold,
                "stop_loss_atr_multiplier": self.params.stop_loss_atr_multiplier,
                "take_profit_atr_multiplier": self.params.take_profit_atr_multiplier,
                "weight_momentum": self.params.weight_momentum,
                "weight_rsi": self.params.weight_rsi,
            }

            trades_summary = (
                f"Trades: {is_result.total_trades}, "
                f"Win: {is_result.winning_trades}, "
                f"Loss: {is_result.losing_trades}, "
                f"PnL: {is_result.total_pnl:.2f}"
            )

            insights = self.insights.get_insights(
                iteration=iteration,
                win_rate=is_result.win_rate,
                loss_breakdown=analysis,
                current_params=key_params,
                recent_trades_summary=trades_summary,
            )

            # Step 4: Tune parameters
            adjustments = insights.get("parameter_adjustments", {})

            if self.tracker.is_stagnant():
                logger.info("stagnation_detected, applying random perturbation")
                self.params = self.tuner.random_perturbation(self.params)
            elif adjustments:
                self.params = self.tuner.apply_adjustments(self.params, adjustments)

            # Step 5: Record iteration
            record = IterationRecord(
                iteration=iteration,
                win_rate=is_result.win_rate,
                sharpe=is_result.sharpe_ratio,
                sortino=is_result.sortino_ratio,
                max_drawdown=is_result.max_drawdown_pct,
                total_pnl=is_result.total_pnl,
                adjustments=adjustments,
                insights=insights.get("analysis", ""),
            )
            self.tracker.record(record)

            # Track best params
            if is_result.win_rate > self.best_win_rate:
                self.best_win_rate = is_result.win_rate
                self.best_params = self.params.clone()

            # Save to database
            self.db.save_learning_iteration(
                iteration=iteration,
                win_rate=is_result.win_rate,
                loss_rate=1 - is_result.win_rate,
                sharpe=is_result.sharpe_ratio,
                sortino=is_result.sortino_ratio,
                max_drawdown=is_result.max_drawdown_pct,
                total_pnl=is_result.total_pnl,
                adjustments_json=json.dumps(adjustments),
                claude_insights=insights.get("analysis", ""),
            )

            logger.info(
                "iteration_complete",
                iteration=iteration,
                win_rate=f"{is_result.win_rate:.2%}",
                sharpe=is_result.sharpe_ratio,
                pnl=is_result.total_pnl,
            )

        # Final validation with best params
        return self._validate_best(split, df)

    def _validate_best(self, split, full_df: pd.DataFrame) -> dict:
        """4-stage validation of best parameters."""
        logger.info("validation_start", best_win_rate=f"{self.best_win_rate:.2%}")

        self.engine.update_parameters(self.best_params)

        # Stage 1: In-sample (already done, use best)
        is_result = self.engine.run(split.in_sample)

        # Stage 2: Out-of-sample
        oos_result = self.engine.run(split.out_of_sample)

        # Stage 3: Walk-forward
        wf_splits = self.splitter.walk_forward_splits(full_df, n_folds=3)
        wf_results = []
        for wf_split in wf_splits:
            wf_result = self.engine.run(wf_split.out_of_sample)
            wf_results.append(wf_result.win_rate)
        avg_wf_win_rate = sum(wf_results) / len(wf_results) if wf_results else 0

        # Stage 4: Monte Carlo
        trade_returns = [t.pnl_pct for t in oos_result.trades if t.pnl_pct]
        mc_result = self.mc.simulate(trade_returns, self.initial_capital)

        # Generate report
        report_path = self.reporter.save_report(
            oos_result, mc_result, self.best_params.to_json()
        )

        # Save best params
        self.db.save_strategy_parameters(StrategyParametersRecord(
            iteration=len(self.tracker.history),
            parameters_json=self.best_params.to_json(),
            in_sample_win_rate=is_result.win_rate,
            oos_win_rate=oos_result.win_rate,
            sharpe_ratio=oos_result.sharpe_ratio,
            max_drawdown=oos_result.max_drawdown_pct,
            total_trades=oos_result.total_trades,
        ))

        validation = {
            "in_sample_win_rate": round(is_result.win_rate, 4),
            "oos_win_rate": round(oos_result.win_rate, 4),
            "walk_forward_avg_win_rate": round(avg_wf_win_rate, 4),
            "monte_carlo": {
                "median_return": mc_result.median_return,
                "probability_of_loss": mc_result.probability_of_loss,
                "median_max_drawdown": mc_result.median_max_drawdown,
            },
            "passed_oos": oos_result.win_rate >= OOS_MIN_WIN_RATE,
            "ready_for_live": (
                is_result.win_rate >= TARGET_WIN_RATE
                and oos_result.win_rate >= OOS_MIN_WIN_RATE
            ),
            "report_path": report_path,
            "best_params": json.loads(self.best_params.to_json()),
            "iterations": len(self.tracker.history),
            "summary": self.tracker.get_summary(),
        }

        logger.info("validation_complete", **{k: v for k, v in validation.items()
                                               if k != "best_params"})
        return validation
