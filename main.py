"""Chart Predictor - AI-powered crypto futures trading system.

CLI entry point supporting: collect, backtest, learn, live modes.
"""

from __future__ import annotations

import asyncio
import sys
import json

import click

from config.settings import get_settings
from config.logging_config import setup_logging, get_logger
from config.constants import DEFAULT_TIMEFRAMES, COLLECT_START_DATE, COLLECT_END_DATE


@click.group()
@click.option("--log-level", default="INFO", help="Log level")
def cli(log_level: str):
    """Chart Predictor - AI Crypto Futures Trading System."""
    setup_logging(level=log_level)


@cli.command()
@click.option("--start", default=COLLECT_START_DATE, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=COLLECT_END_DATE, help="End date (YYYY-MM-DD)")
@click.option("--symbols", default=None, help="Comma-separated symbols (e.g., BTC/USDT:USDT)")
def collect(start: str, end: str, symbols: str | None):
    """Collect historical OHLCV and funding rate data."""
    logger = get_logger("collect")
    settings = get_settings()

    from data.database import Database
    from data.collector import DataCollector
    from data.funding_rates import FundingRateCollector

    db = Database(settings.db_path)

    symbol_list = None
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",")]

    # Collect OHLCV
    logger.info("Starting OHLCV collection...")
    collector = DataCollector(db)
    ohlcv_results = collector.collect_all(
        symbols=symbol_list, start_date=start, end_date=end
    )

    click.echo("\n--- OHLCV Collection Results ---")
    for key, count in ohlcv_results.items():
        click.echo(f"  {key}: {count} candles")

    # Collect funding rates
    logger.info("Starting funding rate collection...")
    fr_collector = FundingRateCollector(db)
    fr_results = fr_collector.collect_all(
        symbols=symbol_list, start_date=start, end_date=end
    )

    click.echo("\n--- Funding Rate Results ---")
    for key, count in fr_results.items():
        click.echo(f"  {key}: {count} rates")

    click.echo("\nData collection complete!")


@cli.command()
@click.option("--symbol", default="BTC/USDT:USDT", help="Trading pair")
@click.option("--timeframe", default="1h", help="Candle timeframe")
@click.option("--capital", default=100000, help="Initial capital (KRW)")
@click.option("--leverage", default=5, help="Max leverage")
def backtest(symbol: str, timeframe: str, capital: float, leverage: int):
    """Run backtest with current strategy parameters."""
    logger = get_logger("backtest")
    settings = get_settings()

    from data.database import Database
    from data.collector import DataCollector
    from strategy.signals import StrategyParameters
    from backtest.engine import BacktestEngine
    from backtest.monte_carlo import MonteCarloSimulator
    from backtest.report import ReportGenerator

    db = Database(settings.db_path)
    collector = DataCollector(db)

    logger.info(f"Loading data for {symbol} {timeframe}...")
    df = collector.get_dataframe(symbol, timeframe)

    if df.empty:
        click.echo(f"No data found for {symbol} {timeframe}. Run 'collect' first.")
        return

    click.echo(f"Loaded {len(df)} candles for {symbol} {timeframe}")

    params = StrategyParameters()
    engine = BacktestEngine(
        params=params, initial_capital=capital, max_leverage=leverage
    )

    click.echo("Running backtest...")
    result = engine.run(df)

    # Monte Carlo
    trade_returns = [t.pnl_pct for t in result.trades if t.pnl_pct]
    mc = MonteCarloSimulator()
    mc_result = mc.simulate(trade_returns, capital)

    # Generate report
    reporter = ReportGenerator()
    report_text = reporter.generate_text_report(result, mc_result, params.to_json())
    click.echo(report_text)

    report_path = reporter.save_report(result, mc_result, params.to_json())
    click.echo(f"\nReport saved: {report_path}")


@cli.command()
@click.option("--symbol", default="BTC/USDT:USDT", help="Trading pair")
@click.option("--timeframe", default="1h", help="Candle timeframe")
@click.option("--max-iter", default=50, help="Maximum learning iterations")
@click.option("--capital", default=100000, help="Initial capital (KRW)")
@click.option("--no-claude", is_flag=True, help="Disable Claude CLI (rule-based only)")
def learn(symbol: str, timeframe: str, max_iter: int, capital: float, no_claude: bool):
    """Run self-learning feedback loop."""
    logger = get_logger("learn")
    settings = get_settings()

    from data.database import Database
    from data.collector import DataCollector
    from strategy.signals import StrategyParameters
    from learning.feedback_loop import FeedbackLoop

    db = Database(settings.db_path)
    collector = DataCollector(db)

    logger.info(f"Loading data for {symbol} {timeframe}...")
    df = collector.get_dataframe(symbol, timeframe)

    if df.empty or len(df) < 500:
        click.echo(f"Insufficient data ({len(df)} rows). Need 500+. Run 'collect' first.")
        return

    click.echo(f"Loaded {len(df)} candles. Starting learning loop...")
    click.echo(f"Max iterations: {max_iter}, Claude: {'disabled' if no_claude else 'enabled'}")

    loop = FeedbackLoop(
        db=db,
        initial_capital=capital,
        use_claude=not no_claude,
    )

    validation = loop.run(df, max_iterations=max_iter)

    click.echo("\n" + "=" * 60)
    click.echo("  LEARNING LOOP RESULTS")
    click.echo("=" * 60)
    click.echo(f"  Iterations:        {validation['iterations']}")
    click.echo(f"  In-Sample WR:      {validation['in_sample_win_rate']:.2%}")
    click.echo(f"  Out-of-Sample WR:  {validation['oos_win_rate']:.2%}")
    click.echo(f"  Walk-Forward WR:   {validation['walk_forward_avg_win_rate']:.2%}")
    click.echo(f"  MC P(Loss):        {validation['monte_carlo']['probability_of_loss']:.2%}")
    click.echo(f"  Ready for Live:    {'YES' if validation['ready_for_live'] else 'NO'}")
    click.echo(f"  Report:            {validation['report_path']}")
    click.echo("=" * 60)


@cli.command()
@click.option("--symbol", default="BTC/USDT:USDT", help="Trading pair")
@click.option("--timeframe", default="15", help="Kline interval (minutes)")
@click.option("--capital", default=100000, help="Initial capital (KRW)")
@click.option("--dry-run", is_flag=True, help="Dry run (no real orders)")
def live(symbol: str, timeframe: str, capital: float, dry_run: bool):
    """Start live trading (requires validated strategy)."""
    logger = get_logger("live")
    settings = get_settings()

    from data.database import Database
    from data.live_feed import LiveFeed
    from execution.order_manager import OrderManager
    from execution.position_tracker import PositionTracker
    from agents.technical_analyst import TechnicalAnalystAgent
    from agents.sentiment_analyst import SentimentAnalystAgent
    from agents.risk_reviewer import RiskReviewerAgent
    from agents.final_decider import FinalDeciderAgent
    from agents.supervisor import SupervisorAgent
    from risk.kill_switch import KillSwitch
    from monitoring.health_check import HealthChecker
    from monitoring.dashboard import ConsoleDashboard
    from strategy.signals import StrategyParameters

    if dry_run:
        click.echo("DRY RUN MODE - No real orders will be placed")

    click.echo(f"Starting live trading for {symbol}...")
    click.echo(f"Capital: {capital:,.0f} KRW, Timeframe: {timeframe}m")

    db = Database(settings.db_path)
    kill_switch = KillSwitch()
    tracker = PositionTracker()
    health = HealthChecker()

    # Initialize agents
    params = StrategyParameters()
    tech_agent = TechnicalAnalystAgent(params)
    sentiment_agent = SentimentAnalystAgent(enabled=True)
    risk_agent = RiskReviewerAgent()
    decider = FinalDeciderAgent(max_leverage=settings.trading.max_leverage)
    supervisor = SupervisorAgent(kill_switch=kill_switch)

    order_mgr = None
    if not dry_run:
        order_mgr = OrderManager(db)

    feed = LiveFeed(testnet=settings.bybit.testnet)
    candle_buffer = []

    def on_kline(data):
        """Handle new candle data."""
        if not data:
            return
        health.update_data_time()
        candle_buffer.extend(data)

        # Process when we have enough data
        if len(candle_buffer) >= 100:
            import pandas as pd
            df = pd.DataFrame(candle_buffer[-200:])
            if "close" not in df.columns and len(df.columns) >= 5:
                df.columns = ["timestamp", "open", "high", "low", "close", "volume"][:len(df.columns)]
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Run pipeline
            tech_result = tech_agent.analyze(df=df)
            risk_result = risk_agent.analyze(
                technical_result=tech_result,
                open_positions=len(tracker.positions),
            )
            decision = decider.analyze(
                risk_result=risk_result,
                current_capital=capital,
            )

            details = decision.details
            if details.get("execute") and not kill_switch.is_active:
                logger.info("trade_signal", **details)
                if order_mgr and not dry_run:
                    order_mgr.place_market_order(
                        symbol=symbol,
                        side=details["side"],
                        quantity=abs(details.get("position_value", 0)) / float(df["close"].iloc[-1]),
                        leverage=details.get("leverage", 1),
                        stop_loss=None,
                        take_profit=None,
                    )

            # Update dashboard
            ConsoleDashboard.print_status(
                equity=capital,
                initial_capital=capital,
                daily_pnl=0,
                open_positions=tracker.get_all_positions(),
                win_rate=0.5,
                total_trades=0,
            )

    feed.on_kline(on_kline)

    click.echo("Connecting to WebSocket...")
    try:
        asyncio.run(feed.start([symbol], interval=timeframe))
    except KeyboardInterrupt:
        click.echo("\nStopping live trading...")
        asyncio.run(feed.stop())


if __name__ == "__main__":
    cli()
