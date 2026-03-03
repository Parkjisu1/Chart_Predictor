"""Track learning iterations and detect convergence."""

from __future__ import annotations

from dataclasses import dataclass, field

from config.constants import TARGET_WIN_RATE, STAGNATION_LIMIT, MAX_ITERATIONS
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class IterationRecord:
    iteration: int
    win_rate: float
    sharpe: float
    sortino: float
    max_drawdown: float
    total_pnl: float
    adjustments: dict
    insights: str = ""


class IterationTracker:
    """Track learning iterations and detect convergence/stagnation."""

    def __init__(
        self,
        target_win_rate: float = TARGET_WIN_RATE,
        stagnation_limit: int = STAGNATION_LIMIT,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.target_win_rate = target_win_rate
        self.stagnation_limit = stagnation_limit
        self.max_iterations = max_iterations
        self.history: list[IterationRecord] = []
        self.best_win_rate = 0.0
        self.stagnation_count = 0

    def record(self, record: IterationRecord) -> None:
        """Record an iteration result."""
        self.history.append(record)

        if record.win_rate > self.best_win_rate:
            self.best_win_rate = record.win_rate
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        logger.info(
            "iteration_recorded",
            iteration=record.iteration,
            win_rate=f"{record.win_rate:.2%}",
            best=f"{self.best_win_rate:.2%}",
            stagnation=self.stagnation_count,
        )

    def has_converged(self) -> bool:
        """Check if target win rate has been reached."""
        if not self.history:
            return False
        return self.history[-1].win_rate >= self.target_win_rate

    def is_stagnant(self) -> bool:
        """Check if learning has stagnated."""
        return self.stagnation_count >= self.stagnation_limit

    def should_stop(self) -> tuple[bool, str]:
        """Check if learning loop should terminate."""
        if self.has_converged():
            return True, "converged"
        if len(self.history) >= self.max_iterations:
            return True, "max_iterations"
        return False, ""

    def get_trend(self, window: int = 5) -> str:
        """Get recent win rate trend."""
        if len(self.history) < window:
            return "insufficient_data"

        recent = [r.win_rate for r in self.history[-window:]]
        avg_change = sum(
            recent[i] - recent[i - 1] for i in range(1, len(recent))
        ) / (len(recent) - 1)

        if avg_change > 0.01:
            return "improving"
        elif avg_change < -0.01:
            return "degrading"
        else:
            return "stable"

    def get_summary(self) -> dict:
        """Get summary of all iterations."""
        if not self.history:
            return {"iterations": 0, "best_win_rate": 0}

        win_rates = [r.win_rate for r in self.history]
        return {
            "iterations": len(self.history),
            "best_win_rate": round(self.best_win_rate, 4),
            "latest_win_rate": round(win_rates[-1], 4),
            "trend": self.get_trend(),
            "stagnation_count": self.stagnation_count,
            "converged": self.has_converged(),
        }
