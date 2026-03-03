"""Sentiment Analyst Agent - Uses Claude CLI for market sentiment."""

from __future__ import annotations

import time

import pandas as pd

from agents.base import AgentBase, AgentResult, ClaudeCLIRunner
from strategy.signals import SignalOutput
from strategy.sentiment.onchain_prompts import build_market_sentiment_prompt
from strategy.sentiment.funding_analysis import analyze_funding_sentiment


class SentimentAnalystAgent(AgentBase):
    """Agent 2: Market sentiment analysis.

    Uses Claude CLI for qualitative analysis + funding rate quantitative analysis.
    Disabled during backtesting (speed + avoid look-ahead bias).
    """

    def __init__(self, enabled: bool = True):
        super().__init__("sentiment_analyst")
        self.enabled = enabled
        self.claude = ClaudeCLIRunner() if enabled else None

    def analyze(
        self,
        df: pd.DataFrame | None = None,
        funding_df: pd.DataFrame | None = None,
        symbol: str = "BTC/USDT",
        **kwargs,
    ) -> AgentResult:
        """Analyze market sentiment using funding rates and Claude CLI."""
        start = time.time()

        if not self.enabled:
            return self._make_result(
                reasoning="Sentiment analysis disabled (backtest mode)",
                confidence=0.0,
            )

        # 1. Funding rate analysis (quantitative, always runs)
        funding_signal = SignalOutput(name="funding", value=0.0, confidence=0.0)
        if funding_df is not None and not funding_df.empty:
            funding_signal = analyze_funding_sentiment(funding_df)

        # 2. Claude CLI analysis (qualitative)
        claude_signal = self._run_claude_analysis(df, funding_signal, symbol)

        # Combine signals
        if claude_signal and claude_signal.confidence > 0:
            combined_value = (
                funding_signal.value * 0.4 + claude_signal.value * 0.6
            )
            combined_confidence = (
                funding_signal.confidence * 0.4 + claude_signal.confidence * 0.6
            )
        else:
            combined_value = funding_signal.value
            combined_confidence = funding_signal.confidence * 0.7

        elapsed = time.time() - start
        return self._make_result(
            signal_value=max(-1.0, min(1.0, combined_value)),
            confidence=min(combined_confidence, 1.0),
            reasoning="Sentiment: funding + Claude CLI analysis",
            details={
                "funding": funding_signal.details,
                "claude": claude_signal.details if claude_signal else {},
            },
            execution_time=elapsed,
        )

    def _run_claude_analysis(
        self,
        df: pd.DataFrame | None,
        funding_signal: SignalOutput,
        symbol: str,
    ) -> SignalOutput | None:
        """Run Claude CLI for sentiment analysis."""
        if not self.claude or df is None or df.empty:
            return None

        try:
            close = df["close"]
            price = close.iloc[-1]
            change_24h = ((close.iloc[-1] / close.iloc[-24] - 1) * 100
                          if len(close) >= 24 else 0)
            volume_24h = df["volume"].tail(24).sum() if len(df) >= 24 else 0
            funding_rate = funding_signal.details.get("current_rate", 0)

            # Simple RSI for prompt context
            from strategy.technical.rsi import compute_rsi
            rsi = compute_rsi(close, 14)
            current_rsi = rsi.iloc[-1] if not rsi.isna().all() else 50

            prompt = build_market_sentiment_prompt(
                symbol=symbol,
                price=price,
                price_change_24h=change_24h,
                volume_24h=volume_24h,
                funding_rate=funding_rate,
                rsi=current_rsi,
            )

            result = self.claude.run(prompt)
            if result:
                score = float(result.get("sentiment_score", 0))
                conf = float(result.get("confidence", 0.3))
                return SignalOutput(
                    name="claude_sentiment",
                    value=max(-1.0, min(1.0, score)),
                    confidence=max(0.0, min(1.0, conf)),
                    details=result,
                )
        except Exception as e:
            self.logger.warning("claude_sentiment_failed", error=str(e))

        return None

    def get_signal_output(self, result: AgentResult) -> SignalOutput:
        """Convert agent result to SignalOutput for composite signal."""
        return SignalOutput(
            name="sentiment",
            value=result.signal_value,
            confidence=result.confidence,
            details=result.details,
        )
