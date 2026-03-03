"""Trade analyzer: classify losing trades into 8 failure modes."""

from __future__ import annotations

from collections import Counter

from config.constants import FailureMode
from config.logging_config import get_logger
from data.models import Trade

logger = get_logger(__name__)


class TradeAnalyzer:
    """Classify losing trades by failure mode for targeted improvement."""

    def classify_trade(self, trade: Trade, df_context=None) -> str:
        """Classify a losing trade into one of 8 failure modes."""
        if trade.pnl and trade.pnl >= 0:
            return ""  # Not a loss

        pnl_pct = trade.pnl_pct or 0
        entry = trade.entry_price
        exit_p = trade.exit_price or entry
        stop = trade.stop_loss or 0
        tp = trade.take_profit or 0

        # 1. Wrong direction - price moved significantly opposite
        if abs(pnl_pct) > 0.05:
            return FailureMode.WRONG_DIRECTION.value

        # 2. No stop loss or stop too wide
        if stop == 0:
            return FailureMode.NO_STOP_LOSS.value

        # 3. Overleveraged - hit stop but leverage amplified loss
        leverage = trade.leverage or 1
        if leverage >= 4 and abs(pnl_pct) > 0.03:
            return FailureMode.OVERLEVERAGED.value

        # 4. Stop loss hit precisely - potential early entry
        if trade.side == "long":
            if exit_p <= stop * 1.001:
                # Check if price recovered after stop
                return FailureMode.EARLY_ENTRY.value
        else:
            if exit_p >= stop * 0.999:
                return FailureMode.EARLY_ENTRY.value

        # 5. Take profit almost reached - premature exit or tight TP
        if trade.side == "long" and tp > 0:
            if exit_p > entry and exit_p < tp * 0.95:
                return FailureMode.PREMATURE_EXIT.value
        elif trade.side == "short" and tp > 0:
            if exit_p < entry and exit_p > tp * 1.05:
                return FailureMode.PREMATURE_EXIT.value

        # 6. Late entry - entered at peak/trough
        if abs(pnl_pct) < 0.01:
            return FailureMode.LATE_ENTRY.value

        # 7. High volatility whipsaw
        if abs(pnl_pct) > 0.02 and leverage <= 2:
            return FailureMode.HIGH_VOLATILITY.value

        # 8. Default: trend reversal
        return FailureMode.TREND_REVERSAL.value

    def analyze_trades(self, trades: list[Trade]) -> dict:
        """Analyze all trades and return failure mode breakdown."""
        losses = [t for t in trades if (t.pnl or 0) < 0]

        if not losses:
            return {
                "total_losses": 0,
                "failure_modes": {},
                "dominant_mode": None,
                "suggestions": [],
            }

        modes = []
        for trade in losses:
            mode = self.classify_trade(trade)
            trade.failure_mode = mode
            modes.append(mode)

        counter = Counter(modes)
        total_losses = len(losses)

        failure_breakdown = {
            mode: {
                "count": count,
                "pct": round(count / total_losses, 2),
            }
            for mode, count in counter.most_common()
        }

        dominant = counter.most_common(1)[0][0] if counter else None

        suggestions = self._generate_suggestions(counter, total_losses)

        result = {
            "total_losses": total_losses,
            "failure_modes": failure_breakdown,
            "dominant_mode": dominant,
            "suggestions": suggestions,
        }

        logger.info("trade_analysis_complete",
                     losses=total_losses,
                     dominant=dominant)

        return result

    def _generate_suggestions(
        self, counter: Counter, total: int
    ) -> list[str]:
        """Generate parameter adjustment suggestions based on failure modes."""
        suggestions = []

        for mode, count in counter.most_common(3):
            pct = count / total

            if mode == FailureMode.EARLY_ENTRY.value and pct > 0.2:
                suggestions.append("Widen stop loss (increase stop_loss_atr_multiplier)")
                suggestions.append("Require stronger signal (increase signal_threshold)")

            elif mode == FailureMode.LATE_ENTRY.value and pct > 0.2:
                suggestions.append("Lower signal threshold for faster entry")
                suggestions.append("Reduce RSI overbought/oversold thresholds")

            elif mode == FailureMode.WRONG_DIRECTION.value and pct > 0.2:
                suggestions.append("Increase signal_threshold for higher conviction")
                suggestions.append("Increase momentum weight in composite")

            elif mode == FailureMode.OVERLEVERAGED.value and pct > 0.15:
                suggestions.append("Reduce maximum leverage")
                suggestions.append("Tighten position sizing")

            elif mode == FailureMode.HIGH_VOLATILITY.value and pct > 0.15:
                suggestions.append("Increase GARCH weight to respect vol regimes")
                suggestions.append("Widen stops in high-vol environments")

            elif mode == FailureMode.PREMATURE_EXIT.value and pct > 0.15:
                suggestions.append("Widen take-profit (increase take_profit_atr_multiplier)")

            elif mode == FailureMode.TREND_REVERSAL.value and pct > 0.2:
                suggestions.append("Add trailing stop to lock in profits")
                suggestions.append("Increase ADX trend filter threshold")

        return suggestions
