"""Monte Carlo simulation for strategy robustness testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.constants import MONTE_CARLO_SIMULATIONS
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MonteCarloResult:
    median_return: float
    mean_return: float
    percentile_5: float
    percentile_95: float
    probability_of_loss: float
    worst_case: float
    best_case: float
    median_max_drawdown: float
    simulations: int


class MonteCarloSimulator:
    """Run Monte Carlo simulations on trade results."""

    def __init__(self, n_simulations: int = MONTE_CARLO_SIMULATIONS):
        self.n_simulations = n_simulations

    def simulate(
        self,
        trade_returns: list[float],
        initial_capital: float = 100_000,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation by reshuffling trade order."""
        if not trade_returns or len(trade_returns) < 5:
            return MonteCarloResult(
                median_return=0, mean_return=0, percentile_5=0,
                percentile_95=0, probability_of_loss=0,
                worst_case=0, best_case=0, median_max_drawdown=0,
                simulations=0,
            )

        returns_array = np.array(trade_returns)
        n_trades = len(returns_array)
        final_returns = []
        max_drawdowns = []

        for _ in range(self.n_simulations):
            # Shuffle trade order
            shuffled = np.random.permutation(returns_array)

            # Build equity curve
            equity = initial_capital
            peak = equity
            max_dd = 0.0
            for ret in shuffled:
                equity *= (1 + ret)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            final_return = (equity - initial_capital) / initial_capital
            final_returns.append(final_return)
            max_drawdowns.append(max_dd)

        final_returns = np.array(final_returns)
        max_drawdowns = np.array(max_drawdowns)

        result = MonteCarloResult(
            median_return=round(float(np.median(final_returns)), 4),
            mean_return=round(float(np.mean(final_returns)), 4),
            percentile_5=round(float(np.percentile(final_returns, 5)), 4),
            percentile_95=round(float(np.percentile(final_returns, 95)), 4),
            probability_of_loss=round(float(np.mean(final_returns < 0)), 4),
            worst_case=round(float(np.min(final_returns)), 4),
            best_case=round(float(np.max(final_returns)), 4),
            median_max_drawdown=round(float(np.median(max_drawdowns)), 4),
            simulations=self.n_simulations,
        )

        logger.info(
            "monte_carlo_complete",
            simulations=self.n_simulations,
            median_return=result.median_return,
            prob_loss=result.probability_of_loss,
        )

        return result
