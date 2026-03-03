"""Performance metrics: Sharpe, Sortino, Calmar, MDD, win rate, etc."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.models import BacktestResult


def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252
) -> float:
    """Annualized Sharpe ratio."""
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / periods
    return float(np.sqrt(periods) * excess.mean() / excess.std())


def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252
) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate / periods
    downside = returns[returns < 0]
    if downside.empty or downside.std() == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    return float(np.sqrt(periods) * excess.mean() / downside.std())


def calculate_calmar_ratio(
    total_return: float, max_drawdown: float, years: float = 1.0
) -> float:
    """Calmar ratio = annualized return / max drawdown."""
    if max_drawdown == 0 or years == 0:
        return 0.0
    annualized_return = total_return / years
    return annualized_return / abs(max_drawdown)


def calculate_max_drawdown(equity_curve: list[float]) -> tuple[float, float]:
    """Calculate maximum drawdown (absolute and percentage).

    Returns: (max_dd_absolute, max_dd_pct)
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0, 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    max_dd_pct = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        dd = peak - value
        dd_pct = dd / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    return max_dd, max_dd_pct


def calculate_profit_factor(wins: list[float], losses: list[float]) -> float:
    """Profit factor = gross profits / gross losses."""
    gross_profit = sum(w for w in wins if w > 0)
    gross_loss = abs(sum(l for l in losses if l < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_all_metrics(
    trades: list,
    equity_curve: list[float],
    initial_capital: float,
    trading_days: int = 252,
) -> BacktestResult:
    """Compute all performance metrics from trade list and equity curve."""
    if not trades:
        return BacktestResult()

    winning = [t for t in trades if (t.pnl or 0) > 0]
    losing = [t for t in trades if (t.pnl or 0) <= 0]

    total_pnl = sum(t.pnl or 0 for t in trades)
    total_pnl_pct = total_pnl / initial_capital if initial_capital > 0 else 0

    win_pnls = [t.pnl for t in winning if t.pnl]
    loss_pnls = [t.pnl for t in losing if t.pnl]

    avg_win = np.mean(win_pnls) if win_pnls else 0.0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0.0

    # Build returns series from equity curve
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    max_dd, max_dd_pct = calculate_max_drawdown(equity_curve)
    years = trading_days / 252

    return BacktestResult(
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=len(winning) / len(trades) if trades else 0,
        total_pnl=round(total_pnl, 2),
        total_pnl_pct=round(total_pnl_pct, 4),
        sharpe_ratio=round(calculate_sharpe_ratio(returns), 4),
        sortino_ratio=round(calculate_sortino_ratio(returns), 4),
        calmar_ratio=round(calculate_calmar_ratio(total_pnl_pct, max_dd_pct, years), 4),
        max_drawdown=round(max_dd, 2),
        max_drawdown_pct=round(max_dd_pct, 4),
        avg_win=round(float(avg_win), 2),
        avg_loss=round(float(avg_loss), 2),
        profit_factor=round(calculate_profit_factor(win_pnls, loss_pnls), 4),
        trades=trades,
        equity_curve=equity_curve,
    )
