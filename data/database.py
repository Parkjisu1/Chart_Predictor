"""SQLite database manager with WAL mode for concurrent reads."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

import pandas as pd

from config.logging_config import get_logger
from data.models import SCHEMA_SQL

logger = get_logger(__name__)


class Database:
    def __init__(self, db_path: str = "data/chart_predictor.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            logger.info("database_initialized", path=str(self.db_path))

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def insert_ohlcv_batch(self, records: list[tuple]) -> int:
        sql = """INSERT OR IGNORE INTO ohlcv
                 (symbol, timeframe, timestamp, open, high, low, close, volume)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
        with self._connect() as conn:
            cursor = conn.executemany(sql, records)
            count = cursor.rowcount
            logger.info("ohlcv_inserted", count=count)
            return count

    def insert_funding_rates(self, records: list[tuple]) -> int:
        sql = """INSERT OR IGNORE INTO funding_rates
                 (symbol, timestamp, funding_rate)
                 VALUES (?, ?, ?)"""
        with self._connect() as conn:
            cursor = conn.executemany(sql, records)
            count = cursor.rowcount
            logger.info("funding_rates_inserted", count=count)
            return count

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> pd.DataFrame:
        query = "SELECT * FROM ohlcv WHERE symbol=? AND timeframe=?"
        params: list = [symbol, timeframe]
        if start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(start_ts)
        if end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(end_ts)
        query += " ORDER BY timestamp ASC"

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_funding_rates(
        self,
        symbol: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> pd.DataFrame:
        query = "SELECT * FROM funding_rates WHERE symbol=?"
        params: list = [symbol]
        if start_ts is not None:
            query += " AND timestamp >= ?"
            params.append(start_ts)
        if end_ts is not None:
            query += " AND timestamp <= ?"
            params.append(end_ts)
        query += " ORDER BY timestamp ASC"

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def insert_trade(self, trade) -> int:
        sql = """INSERT INTO trades
                 (symbol, side, entry_price, exit_price, quantity, leverage,
                  entry_time, exit_time, pnl, pnl_pct, status,
                  stop_loss, take_profit, failure_mode, signals_json)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        with self._connect() as conn:
            cursor = conn.execute(sql, (
                trade.symbol, trade.side, trade.entry_price, trade.exit_price,
                trade.quantity, trade.leverage, trade.entry_time, trade.exit_time,
                trade.pnl, trade.pnl_pct, trade.status,
                trade.stop_loss, trade.take_profit,
                trade.failure_mode, trade.signals_json,
            ))
            return cursor.lastrowid

    def get_trades(self, status: str | None = None) -> list[dict]:
        query = "SELECT * FROM trades"
        params = []
        if status:
            query += " WHERE status=?"
            params.append(status)
        query += " ORDER BY entry_time DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def save_strategy_parameters(self, record) -> int:
        sql = """INSERT INTO strategy_parameters
                 (iteration, parameters_json, in_sample_win_rate,
                  oos_win_rate, sharpe_ratio, max_drawdown, total_trades)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"""
        with self._connect() as conn:
            cursor = conn.execute(sql, (
                record.iteration, record.parameters_json,
                record.in_sample_win_rate, record.oos_win_rate,
                record.sharpe_ratio, record.max_drawdown, record.total_trades,
            ))
            return cursor.lastrowid

    def save_learning_iteration(
        self,
        iteration: int,
        win_rate: float,
        loss_rate: float,
        sharpe: float,
        sortino: float,
        max_drawdown: float,
        total_pnl: float,
        adjustments_json: str,
        claude_insights: str,
    ) -> int:
        sql = """INSERT INTO learning_iterations
                 (iteration, win_rate, loss_rate, sharpe, sortino,
                  max_drawdown, total_pnl, adjustments_json, claude_insights)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        with self._connect() as conn:
            cursor = conn.execute(sql, (
                iteration, win_rate, loss_rate, sharpe, sortino,
                max_drawdown, total_pnl, adjustments_json, claude_insights,
            ))
            return cursor.lastrowid

    def save_kill_switch_event(self, reason: str, details_json: str) -> int:
        sql = "INSERT INTO kill_switch_events (reason, details_json) VALUES (?, ?)"
        with self._connect() as conn:
            cursor = conn.execute(sql, (reason, details_json))
            return cursor.lastrowid

    def get_ohlcv_count(self, symbol: str, timeframe: str) -> int:
        sql = "SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?"
        with self._connect() as conn:
            row = conn.execute(sql, (symbol, timeframe)).fetchone()
            return row[0]

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> int | None:
        sql = "SELECT MAX(timestamp) FROM ohlcv WHERE symbol=? AND timeframe=?"
        with self._connect() as conn:
            row = conn.execute(sql, (symbol, timeframe)).fetchone()
            return row[0]
