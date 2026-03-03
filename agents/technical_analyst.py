"""Technical Analyst Agent - Pure quantitative analysis, no LLM."""

from __future__ import annotations

import time

import pandas as pd

from agents.base import AgentBase, AgentResult
from strategy.signals import StrategyParameters, CompositeSignal
from strategy.technical.composite import compute_composite_signal


class TechnicalAnalystAgent(AgentBase):
    """Agent 1: Quantitative technical analysis.

    Runs all technical indicators and produces a composite signal.
    No LLM dependency - pure computation.
    """

    def __init__(self, params: StrategyParameters | None = None):
        super().__init__("technical_analyst")
        self.params = params or StrategyParameters()

    def analyze(
        self,
        df: pd.DataFrame,
        sentiment_signal=None,
        **kwargs,
    ) -> AgentResult:
        """Analyze OHLCV data using all technical indicators."""
        start = time.time()

        if df is None or df.empty or len(df) < 50:
            return self._make_result(
                error="Insufficient data for analysis",
                confidence=0.0,
            )

        try:
            composite = compute_composite_signal(
                df, self.params, sentiment_signal
            )

            elapsed = time.time() - start
            return self._make_result(
                signal_value=composite.score,
                confidence=composite.confidence,
                reasoning=f"Composite signal: {composite.signal} (score={composite.score:.4f})",
                details={
                    "signal": composite.signal,
                    "score": composite.score,
                    "confidence": composite.confidence,
                    "components": {
                        c.name: {
                            "value": round(c.value, 4),
                            "confidence": round(c.confidence, 4),
                            "details": c.details,
                        }
                        for c in composite.components
                    },
                    "metadata": composite.metadata,
                },
                execution_time=elapsed,
            )
        except Exception as e:
            self.logger.error("technical_analysis_failed", error=str(e))
            return self._make_result(
                error=str(e),
                confidence=0.0,
                execution_time=time.time() - start,
            )

    def update_parameters(self, params: StrategyParameters) -> None:
        self.params = params
