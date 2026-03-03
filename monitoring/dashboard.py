"""Simple console dashboard for monitoring."""

from __future__ import annotations

from datetime import datetime

from config.logging_config import get_logger

logger = get_logger(__name__)


class ConsoleDashboard:
    """Print formatted status to console."""

    @staticmethod
    def print_status(
        equity: float,
        initial_capital: float,
        daily_pnl: float,
        open_positions: list[dict],
        win_rate: float,
        total_trades: int,
    ) -> str:
        """Generate status display string."""
        pnl_total = equity - initial_capital
        pnl_pct = pnl_total / initial_capital * 100

        lines = [
            "",
            "=" * 50,
            f"  Chart Predictor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            f"  Equity:       ${equity:,.2f} ({pnl_pct:+.2f}%)",
            f"  Daily PnL:    ${daily_pnl:,.2f}",
            f"  Win Rate:     {win_rate:.1%} ({total_trades} trades)",
            "-" * 50,
        ]

        if open_positions:
            lines.append("  Open Positions:")
            for p in open_positions:
                lines.append(
                    f"    {p['symbol']} {p['side'].upper()}: "
                    f"${p.get('current_price', 0):,.2f} "
                    f"({p.get('pnl_pct', 0):+.1f}%)"
                )
        else:
            lines.append("  No open positions")

        lines.append("=" * 50)
        output = "\n".join(lines)
        print(output)
        return output
