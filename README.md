# Chart Predictor

AI-powered crypto futures trading system for Bybit. Multi-agent architecture with self-learning feedback loop.

## Architecture

5-agent pipeline:
1. **Technical Analyst** - RSI, Bollinger, MACD, ADX, OBV/VWAP, GARCH(1,1)
2. **Sentiment Analyst** - Funding rates + Claude CLI macro analysis
3. **Risk Reviewer** - Decision matrix cross-validation
4. **Final Decider** - Modified Half-Kelly position sizing
5. **Supervisor** - Kill switch authority + daily reviews

## Self-Learning Loop

```
Backtest → Loss Classification (8 modes) → Claude Insights → Parameter Tuning → Validation → Repeat
```

4-stage validation: In-Sample → Out-of-Sample → Walk-Forward → Monte Carlo

Target: 90% in-sample win rate, 85% OOS minimum.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Bybit API keys
```

## Usage

```bash
# 1. Collect historical data (2021-2024)
python main.py collect

# 2. Run backtest
python main.py backtest --symbol "BTC/USDT:USDT" --timeframe 1h

# 3. Run self-learning loop
python main.py learn --symbol "BTC/USDT:USDT" --max-iter 50

# 4. Start live trading (after validation)
python main.py live --symbol "BTC/USDT:USDT" --dry-run
```

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
├── main.py                     # CLI entry point
├── config/                     # Settings, constants, logging
├── data/                       # Database, collectors, live feed
├── agents/                     # 5 trading agents
├── strategy/
│   ├── technical/              # RSI, Bollinger, Volume, GARCH, Momentum
│   └── sentiment/              # Funding analysis, Claude prompts
├── risk/                       # Position sizing, kill switch, CVaR
├── backtest/                   # Engine, metrics, Monte Carlo
├── learning/                   # Self-learning feedback loop
├── execution/                  # Order management, rate limiting
├── monitoring/                 # Telegram, health checks
└── tests/                      # Unit & integration tests
```

## Risk Management

- Max 25% capital per position
- Max 50% total exposure
- 5% daily loss → kill switch
- 15% drawdown → kill switch
- Correlation check between pairs (>0.7 blocked)
- CVaR risk assessment

## Tech Stack

- Python 3.13, ccxt, pandas, numpy
- GARCH via `arch`, indicators via custom implementation
- Claude CLI for sentiment analysis (no API costs)
- SQLite WAL mode for concurrent access
- Bybit WebSocket for live data
