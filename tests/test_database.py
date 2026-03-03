"""Tests for database operations."""

import os
import tempfile
import pytest

from data.database import Database
from data.models import Trade, StrategyParametersRecord


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    database = Database(db_path)
    yield database
    os.unlink(db_path)
    # Clean up WAL files
    for suffix in ["-wal", "-shm"]:
        try:
            os.unlink(db_path + suffix)
        except FileNotFoundError:
            pass


class TestDatabase:
    def test_insert_ohlcv(self, db):
        records = [
            ("BTCUSDT", "1h", 1700000000000, 50000, 50500, 49500, 50200, 100),
            ("BTCUSDT", "1h", 1700003600000, 50200, 50700, 50100, 50500, 150),
        ]
        count = db.insert_ohlcv_batch(records)
        assert count == 2

    def test_get_ohlcv(self, db):
        records = [
            ("BTCUSDT", "1h", 1700000000000, 50000, 50500, 49500, 50200, 100),
        ]
        db.insert_ohlcv_batch(records)
        df = db.get_ohlcv("BTCUSDT", "1h")
        assert len(df) == 1
        assert df.iloc[0]["close"] == 50200

    def test_insert_funding_rates(self, db):
        records = [
            ("BTCUSDT", 1700000000000, 0.0001),
            ("BTCUSDT", 1700028800000, -0.0002),
        ]
        count = db.insert_funding_rates(records)
        assert count == 2

    def test_insert_trade(self, db):
        trade = Trade(
            symbol="BTCUSDT",
            side="long",
            entry_price=50000,
            quantity=0.1,
            leverage=3,
            entry_time="2024-01-01T00:00:00",
        )
        trade_id = db.insert_trade(trade)
        assert trade_id > 0

    def test_get_trades(self, db):
        trade = Trade(
            symbol="BTCUSDT", side="long",
            entry_price=50000, quantity=0.1,
            entry_time="2024-01-01",
        )
        db.insert_trade(trade)
        trades = db.get_trades()
        assert len(trades) == 1

    def test_duplicate_ohlcv_ignored(self, db):
        records = [
            ("BTCUSDT", "1h", 1700000000000, 50000, 50500, 49500, 50200, 100),
        ]
        db.insert_ohlcv_batch(records)
        db.insert_ohlcv_batch(records)  # Duplicate
        df = db.get_ohlcv("BTCUSDT", "1h")
        assert len(df) == 1

    def test_ohlcv_count(self, db):
        records = [
            ("BTCUSDT", "1h", 1700000000000, 50000, 50500, 49500, 50200, 100),
            ("BTCUSDT", "1h", 1700003600000, 50200, 50700, 50100, 50500, 150),
        ]
        db.insert_ohlcv_batch(records)
        count = db.get_ohlcv_count("BTCUSDT", "1h")
        assert count == 2

    def test_save_strategy_parameters(self, db):
        record = StrategyParametersRecord(
            iteration=1,
            parameters_json='{"rsi_period": 14}',
            in_sample_win_rate=0.65,
        )
        rid = db.save_strategy_parameters(record)
        assert rid > 0
