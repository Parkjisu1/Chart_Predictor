"""Event-driven backtesting engine."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from config.logging_config import get_logger
from config.constants import Signal, Side
from strategy.signals import StrategyParameters
from agents.technical_analyst import TechnicalAnalystAgent
from agents.risk_reviewer import RiskReviewerAgent
from agents.final_decider import FinalDeciderAgent
from backtest.cost_model import CostModel
from backtest.metrics import compute_all_metrics
from data.models import Trade, BacktestResult

logger = get_logger(__name__)


class BacktestEngine:
    """Event-driven backtesting engine.

    Processes candles sequentially, running the agent pipeline
    (without sentiment) to generate and manage trades.
    """

    def __init__(
        self,
        params: StrategyParameters | None = None,
        initial_capital: float = 100_000,
        cost_model: CostModel | None = None,
        max_leverage: int = 5,
    ):
        self.params = params or StrategyParameters()
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.max_leverage = max_leverage

        # Agents (sentiment disabled in backtest)
        self.tech_agent = TechnicalAnalystAgent(self.params)
        self.risk_agent = RiskReviewerAgent()
        self.decider = FinalDeciderAgent(max_leverage=max_leverage)

        # State
        self.equity = initial_capital
        self.equity_curve: list[float] = [initial_capital]
        self.trades: list[Trade] = []
        self.open_trades: list[Trade] = []
        self.peak_equity = initial_capital
        self.daily_pnl = 0.0
        self.win_count = 0
        self.total_count = 0

    def run(
        self,
        df: pd.DataFrame,
        lookback: int = 100,
        funding_rate: float = 0.0001,
    ) -> BacktestResult:
        """Run backtest on OHLCV DataFrame."""
        if df is None or df.empty or len(df) < lookback:
            logger.warning("insufficient_data", rows=len(df) if df is not None else 0)
            return BacktestResult()

        self._reset()

        logger.info("backtest_start", rows=len(df), lookback=lookback)

        for i in range(lookback, len(df)):
            window = df.iloc[i - lookback : i + 1]
            current_bar = df.iloc[i]
            current_price = current_bar["close"]
            timestamp = str(current_bar.name) if hasattr(current_bar, "name") else str(i)

            # 1. Check and close open trades
            self._check_exits(current_bar, timestamp, funding_rate)

            # 2. Run agent pipeline
            decision = self._run_pipeline(window, current_price)

            # 3. Execute trade if approved
            if decision and decision.get("execute", False):
                self._open_trade(decision, current_price, timestamp)

            # 4. Update equity curve
            unrealized = self._calculate_unrealized_pnl(current_price)
            self.equity_curve.append(self.equity + unrealized)

        # Close any remaining open trades
        if self.open_trades and len(df) > 0:
            last_price = df.iloc[-1]["close"]
            last_ts = str(df.iloc[-1].name) if hasattr(df.iloc[-1], "name") else "end"
            for trade in list(self.open_trades):
                self._close_trade(trade, last_price, last_ts, "backtest_end")

        result = compute_all_metrics(
            self.trades, self.equity_curve, self.initial_capital,
            trading_days=len(df) - lookback,
        )

        logger.info(
            "backtest_complete",
            trades=result.total_trades,
            win_rate=f"{result.win_rate:.2%}",
            total_pnl=result.total_pnl,
            sharpe=result.sharpe_ratio,
            max_dd=f"{result.max_drawdown_pct:.2%}",
        )

        return result

    def _reset(self) -> None:
        self.equity = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.trades = []
        self.open_trades = []
        self.peak_equity = self.initial_capital
        self.daily_pnl = 0.0
        self.win_count = 0
        self.total_count = 0

    def _run_pipeline(self, window: pd.DataFrame, current_price: float) -> dict | None:
        """Run the agent pipeline (tech -> risk -> decide)."""
        # Skip if we already have max positions
        if len(self.open_trades) >= 3:
            return None

        # 1. Technical analysis
        tech_result = self.tech_agent.analyze(df=window)
        if tech_result.error:
            return None

        # 2. Risk review
        drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        risk_result = self.risk_agent.analyze(
            technical_result=tech_result,
            current_drawdown=drawdown,
            open_positions=len(self.open_trades),
            daily_pnl=self.daily_pnl / self.initial_capital if self.initial_capital > 0 else 0,
        )

        # 3. Final decision
        win_rate = self.win_count / self.total_count if self.total_count > 0 else 0.5
        avg_win_loss = 1.5  # Default
        if self.trades:
            wins = [t.pnl for t in self.trades if (t.pnl or 0) > 0]
            losses = [abs(t.pnl) for t in self.trades if (t.pnl or 0) < 0]
            if wins and losses:
                avg_win_loss = (sum(wins) / len(wins)) / (sum(losses) / len(losses))

        decision_result = self.decider.analyze(
            risk_result=risk_result,
            current_capital=self.equity,
            current_price=current_price,
            historical_win_rate=win_rate,
            avg_win_loss_ratio=avg_win_loss,
        )

        return decision_result.details if decision_result else None

    def _open_trade(self, decision: dict, price: float, timestamp: str) -> None:
        """Open a new trade."""
        side = decision["side"]
        position_pct = decision["position_size_pct"]
        leverage = decision["leverage"]
        stop_loss_pct = decision["stop_loss_pct"]
        take_profit_pct = decision["take_profit_pct"]

        notional = self.equity * position_pct * leverage
        quantity = notional / price

        # Entry costs
        costs = self.cost_model.calculate_entry_cost(notional)
        self.equity -= costs["total"]

        # Calculate stop/take-profit prices
        if side == "long":
            stop_loss = price * (1 - stop_loss_pct)
            take_profit = price * (1 + take_profit_pct)
        else:
            stop_loss = price * (1 + stop_loss_pct)
            take_profit = price * (1 - take_profit_pct)

        trade = Trade(
            symbol="backtest",
            side=side,
            entry_price=price,
            quantity=quantity,
            leverage=leverage,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signals_json=json.dumps(decision),
        )
        self.open_trades.append(trade)

    def _check_exits(self, bar, timestamp: str, funding_rate: float) -> None:
        """Check stop loss, take profit, and trailing stops."""
        high = bar["high"]
        low = bar["low"]
        close = bar["close"]

        for trade in list(self.open_trades):
            exit_price = None
            exit_reason = None

            if trade.side == "long":
                if low <= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = "stop_loss"
                elif high >= trade.take_profit:
                    exit_price = trade.take_profit
                    exit_reason = "take_profit"
            else:  # short
                if high >= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = "stop_loss"
                elif low <= trade.take_profit:
                    exit_price = trade.take_profit
                    exit_reason = "take_profit"

            if exit_price:
                self._close_trade(trade, exit_price, timestamp, exit_reason)

    def _close_trade(self, trade: Trade, exit_price: float, timestamp: str, reason: str) -> None:
        """Close a trade and calculate P&L."""
        notional_entry = trade.entry_price * trade.quantity
        notional_exit = exit_price * trade.quantity

        # P&L
        if trade.side == "long":
            raw_pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            raw_pnl = (trade.entry_price - exit_price) * trade.quantity

        # Exit costs
        exit_costs = self.cost_model.calculate_exit_cost(notional_exit)

        # Funding cost (estimate based on hold time)
        funding_cost = 0  # Simplified for backtest

        net_pnl = raw_pnl - exit_costs["total"] - abs(funding_cost)

        trade.exit_price = exit_price
        trade.exit_time = timestamp
        trade.pnl = round(net_pnl, 4)
        trade.pnl_pct = round(net_pnl / notional_entry, 4) if notional_entry > 0 else 0
        trade.status = "closed"

        self.equity += net_pnl
        self.daily_pnl += net_pnl
        self.total_count += 1
        if net_pnl > 0:
            self.win_count += 1
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.trades.append(trade)
        self.open_trades.remove(trade)

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for all open trades."""
        total = 0.0
        for trade in self.open_trades:
            if trade.side == "long":
                total += (current_price - trade.entry_price) * trade.quantity
            else:
                total += (trade.entry_price - current_price) * trade.quantity
        return total

    def update_parameters(self, params: StrategyParameters) -> None:
        """Update strategy parameters for next run."""
        self.params = params
        self.tech_agent.update_parameters(params)
