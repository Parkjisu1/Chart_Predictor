"""SQLite database table definitions and dataclasses."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime


# --- SQL Schema ---

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_ts
    ON ohlcv(symbol, timeframe, timestamp);

CREATE TABLE IF NOT EXISTS funding_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    funding_rate REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_funding_symbol_ts
    ON funding_rates(symbol, timestamp);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity REAL NOT NULL,
    leverage INTEGER DEFAULT 1,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    pnl REAL,
    pnl_pct REAL,
    status TEXT DEFAULT 'open',
    stop_loss REAL,
    take_profit REAL,
    failure_mode TEXT,
    signals_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS strategy_parameters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration INTEGER NOT NULL,
    parameters_json TEXT NOT NULL,
    in_sample_win_rate REAL,
    oos_win_rate REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    total_trades INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS learning_iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration INTEGER NOT NULL,
    win_rate REAL,
    loss_rate REAL,
    sharpe REAL,
    sortino REAL,
    max_drawdown REAL,
    total_pnl REAL,
    adjustments_json TEXT,
    claude_insights TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kill_switch_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reason TEXT NOT NULL,
    details_json TEXT,
    triggered_at TEXT DEFAULT (datetime('now'))
);
"""


# --- Dataclasses ---

@dataclass
class OHLCV:
    symbol: str
    timeframe: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_tuple(self) -> tuple:
        return (self.symbol, self.timeframe, self.timestamp,
                self.open, self.high, self.low, self.close, self.volume)


@dataclass
class FundingRate:
    symbol: str
    timestamp: int
    funding_rate: float

    def to_tuple(self) -> tuple:
        return (self.symbol, self.timestamp, self.funding_rate)


@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    leverage: int = 1
    entry_time: str = ""
    exit_price: float | None = None
    exit_time: str | None = None
    pnl: float | None = None
    pnl_pct: float | None = None
    status: str = "open"
    stop_loss: float | None = None
    take_profit: float | None = None
    failure_mode: str | None = None
    signals_json: str | None = None
    id: int | None = None


@dataclass
class BacktestResult:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


@dataclass
class StrategyParametersRecord:
    iteration: int
    parameters_json: str
    in_sample_win_rate: float | None = None
    oos_win_rate: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    total_trades: int | None = None
