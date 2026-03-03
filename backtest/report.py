"""Backtest report generation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from data.models import BacktestResult
from backtest.monte_carlo import MonteCarloResult
from config.logging_config import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generate formatted backtest reports."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_text_report(
        self,
        result: BacktestResult,
        mc_result: MonteCarloResult | None = None,
        params_json: str = "",
    ) -> str:
        """Generate a text-based backtest report."""
        lines = [
            "=" * 60,
            "  BACKTEST REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "--- Performance Summary ---",
            f"  Total Trades:      {result.total_trades}",
            f"  Winning Trades:    {result.winning_trades}",
            f"  Losing Trades:     {result.losing_trades}",
            f"  Win Rate:          {result.win_rate:.2%}",
            f"  Total PnL:         ${result.total_pnl:,.2f} ({result.total_pnl_pct:.2%})",
            f"  Avg Win:           ${result.avg_win:,.2f}",
            f"  Avg Loss:          ${result.avg_loss:,.2f}",
            f"  Profit Factor:     {result.profit_factor:.2f}",
            "",
            "--- Risk Metrics ---",
            f"  Sharpe Ratio:      {result.sharpe_ratio:.4f}",
            f"  Sortino Ratio:     {result.sortino_ratio:.4f}",
            f"  Calmar Ratio:      {result.calmar_ratio:.4f}",
            f"  Max Drawdown:      ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2%})",
        ]

        if mc_result:
            lines.extend([
                "",
                "--- Monte Carlo Analysis ---",
                f"  Simulations:       {mc_result.simulations}",
                f"  Median Return:     {mc_result.median_return:.2%}",
                f"  5th Percentile:    {mc_result.percentile_5:.2%}",
                f"  95th Percentile:   {mc_result.percentile_95:.2%}",
                f"  P(Loss):           {mc_result.probability_of_loss:.2%}",
                f"  Worst Case:        {mc_result.worst_case:.2%}",
                f"  Median Max DD:     {mc_result.median_max_drawdown:.2%}",
            ])

        lines.append("")
        lines.append("=" * 60)

        report = "\n".join(lines)
        return report

    def save_report(
        self,
        result: BacktestResult,
        mc_result: MonteCarloResult | None = None,
        params_json: str = "",
        filename: str | None = None,
    ) -> str:
        """Save report to file and return path."""
        if filename is None:
            filename = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        text_report = self.generate_text_report(result, mc_result, params_json)

        # Save text report
        text_path = self.output_dir / f"{filename}.txt"
        text_path.write_text(text_report, encoding="utf-8")

        # Save JSON data
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "performance": {
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "total_pnl": result.total_pnl,
                "sharpe": result.sharpe_ratio,
                "sortino": result.sortino_ratio,
                "calmar": result.calmar_ratio,
                "max_drawdown_pct": result.max_drawdown_pct,
                "profit_factor": result.profit_factor,
            },
            "parameters": json.loads(params_json) if params_json else {},
        }
        if mc_result:
            json_data["monte_carlo"] = {
                "median_return": mc_result.median_return,
                "percentile_5": mc_result.percentile_5,
                "percentile_95": mc_result.percentile_95,
                "probability_of_loss": mc_result.probability_of_loss,
                "median_max_drawdown": mc_result.median_max_drawdown,
            }

        json_path = self.output_dir / f"{filename}.json"
        json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

        logger.info("report_saved", path=str(text_path))
        return str(text_path)
