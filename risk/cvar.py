"""Conditional Value at Risk (CVaR) calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd


class CVaRCalculator:
    """Calculate Value at Risk and Conditional Value at Risk."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def calculate_var(self, returns: pd.Series) -> float:
        """Calculate historical VaR at confidence level."""
        if returns.empty or len(returns) < 10:
            return 0.0
        return -np.percentile(returns.dropna(), (1 - self.confidence_level) * 100)

    def calculate_cvar(self, returns: pd.Series) -> float:
        """Calculate CVaR (Expected Shortfall) at confidence level."""
        if returns.empty or len(returns) < 10:
            return 0.0
        var = self.calculate_var(returns)
        tail_returns = returns[returns <= -var]
        if tail_returns.empty:
            return var
        return -tail_returns.mean()

    def assess_risk(
        self,
        returns: pd.Series,
        position_value: float,
    ) -> dict:
        """Full risk assessment with VaR and CVaR."""
        var = self.calculate_var(returns)
        cvar = self.calculate_cvar(returns)

        return {
            "var_pct": round(var, 6),
            "cvar_pct": round(cvar, 6),
            "var_value": round(position_value * var, 2),
            "cvar_value": round(position_value * cvar, 2),
            "confidence": self.confidence_level,
            "observations": len(returns.dropna()),
        }
