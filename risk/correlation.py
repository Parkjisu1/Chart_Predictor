"""Correlation analysis between trading pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.constants import CORRELATION_THRESHOLD
from config.logging_config import get_logger

logger = get_logger(__name__)


class CorrelationAnalyzer:
    """Check correlations to avoid concentrated exposure."""

    def __init__(self, threshold: float = CORRELATION_THRESHOLD):
        self.threshold = threshold

    def compute_correlation(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
        window: int = 30,
    ) -> float:
        """Compute rolling correlation between two return series."""
        if len(returns_a) < window or len(returns_b) < window:
            return 0.0

        aligned = pd.concat([returns_a, returns_b], axis=1).dropna()
        if len(aligned) < window:
            return 0.0

        corr = aligned.iloc[-window:].corr().iloc[0, 1]
        return corr if not np.isnan(corr) else 0.0

    def check_pair_correlation(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
        symbol_a: str = "A",
        symbol_b: str = "B",
    ) -> dict:
        """Check if two pairs are too correlated for simultaneous positions."""
        corr = self.compute_correlation(returns_a, returns_b)
        too_high = abs(corr) > self.threshold

        if too_high:
            logger.warning(
                "high_correlation",
                pair_a=symbol_a,
                pair_b=symbol_b,
                correlation=round(corr, 4),
            )

        return {
            "correlation": round(corr, 4),
            "too_high": too_high,
            "threshold": self.threshold,
        }

    def get_portfolio_correlation_risk(
        self,
        returns_dict: dict[str, pd.Series],
    ) -> float:
        """Get maximum pairwise correlation in portfolio."""
        symbols = list(returns_dict.keys())
        if len(symbols) < 2:
            return 0.0

        max_corr = 0.0
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = abs(self.compute_correlation(
                    returns_dict[symbols[i]],
                    returns_dict[symbols[j]],
                ))
                max_corr = max(max_corr, corr)

        return max_corr
