"""OHLCV data collector using ccxt for Bybit futures."""

from __future__ import annotations

import time
from datetime import datetime

import ccxt
import pandas as pd

from config.logging_config import get_logger
from config.settings import get_settings
from config.constants import Timeframe, DEFAULT_TIMEFRAMES
from data.database import Database
from data.models import OHLCV

logger = get_logger(__name__)

# Timeframe -> milliseconds
TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class DataCollector:
    def __init__(self, db: Database | None = None):
        settings = get_settings()
        self.exchange = ccxt.bybit({
            "apiKey": settings.bybit.api_key or None,
            "secret": settings.bybit.api_secret or None,
            "options": {"defaultType": "linear"},
            "enableRateLimit": True,
        })
        if settings.bybit.testnet:
            self.exchange.set_sandbox_mode(True)
        self.db = db or Database(settings.db_path)

    def collect_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> int:
        """Collect OHLCV data for a symbol/timeframe range. Returns count inserted."""
        since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        until = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        tf_ms = TIMEFRAME_MS.get(timeframe, 60_000)
        limit = 200
        total_inserted = 0
        current = since

        # Check if we already have data, resume from latest
        latest_ts = self.db.get_latest_timestamp(symbol, timeframe)
        if latest_ts and latest_ts > since:
            current = latest_ts + tf_ms
            logger.info("resuming_collection", symbol=symbol, timeframe=timeframe,
                        from_ts=current)

        logger.info("collecting_ohlcv", symbol=symbol, timeframe=timeframe,
                     start=start_date, end=end_date)

        while current < until:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=current, limit=limit
                )
                if not candles:
                    break

                records = []
                for c in candles:
                    ts, o, h, l, cl, v = c
                    if ts > until:
                        break
                    ohlcv = OHLCV(symbol=symbol, timeframe=timeframe,
                                  timestamp=ts, open=o, high=h, low=l,
                                  close=cl, volume=v)
                    records.append(ohlcv.to_tuple())

                if records:
                    inserted = self.db.insert_ohlcv_batch(records)
                    total_inserted += max(inserted, 0)

                current = candles[-1][0] + tf_ms
                time.sleep(0.1)  # Rate limit courtesy

            except ccxt.RateLimitExceeded:
                logger.warning("rate_limit_hit", symbol=symbol)
                time.sleep(5)
            except ccxt.NetworkError as e:
                logger.warning("network_error", error=str(e))
                time.sleep(3)
            except Exception as e:
                logger.error("collection_error", symbol=symbol, error=str(e))
                break

        logger.info("collection_complete", symbol=symbol, timeframe=timeframe,
                     total_inserted=total_inserted)
        return total_inserted

    def collect_all(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        start_date: str = "2021-01-01",
        end_date: str = "2024-12-31",
    ) -> dict[str, int]:
        """Collect data for all symbol/timeframe combinations."""
        settings = get_settings()
        symbols = symbols or [f"{s}/USDT:USDT" for s in
                              [sym.replace("USDT", "") for sym in settings.trading.symbols]]
        timeframes = timeframes or [tf.value for tf in DEFAULT_TIMEFRAMES]

        results = {}
        for symbol in symbols:
            for tf in timeframes:
                key = f"{symbol}_{tf}"
                count = self.collect_ohlcv(symbol, tf, start_date, end_date)
                results[key] = count
                logger.info("collected", key=key, count=count)
        return results

    def get_dataframe(
        self,
        symbol: str,
        timeframe: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Get OHLCV data as a pandas DataFrame."""
        start_ts = (int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
                    if start_date else None)
        end_ts = (int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
                  if end_date else None)

        df = self.db.get_ohlcv(symbol, timeframe, start_ts, end_ts)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("datetime", inplace=True)
        return df
