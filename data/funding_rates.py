"""Funding rate collector for Bybit perpetual futures."""

from __future__ import annotations

import time
from datetime import datetime

import ccxt

from config.logging_config import get_logger
from config.settings import get_settings
from data.database import Database
from data.models import FundingRate

logger = get_logger(__name__)


class FundingRateCollector:
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

    def collect_funding_rates(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> int:
        """Collect historical funding rates for a symbol."""
        since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        until = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        total_inserted = 0
        current = since

        logger.info("collecting_funding_rates", symbol=symbol,
                     start=start_date, end=end_date)

        while current < until:
            try:
                rates = self.exchange.fetch_funding_rate_history(
                    symbol, since=current, limit=200
                )
                if not rates:
                    break

                records = []
                for r in rates:
                    ts = r.get("timestamp", 0)
                    rate = r.get("fundingRate", 0.0)
                    if ts > until:
                        break
                    fr = FundingRate(symbol=symbol, timestamp=ts,
                                    funding_rate=rate)
                    records.append(fr.to_tuple())

                if records:
                    inserted = self.db.insert_funding_rates(records)
                    total_inserted += max(inserted, 0)

                current = rates[-1]["timestamp"] + 1
                time.sleep(0.2)

            except ccxt.RateLimitExceeded:
                logger.warning("rate_limit_hit", symbol=symbol)
                time.sleep(5)
            except ccxt.NetworkError as e:
                logger.warning("network_error", error=str(e))
                time.sleep(3)
            except Exception as e:
                logger.error("funding_collection_error", symbol=symbol,
                             error=str(e))
                break

        logger.info("funding_collection_complete", symbol=symbol,
                     total_inserted=total_inserted)
        return total_inserted

    def collect_all(
        self,
        symbols: list[str] | None = None,
        start_date: str = "2021-01-01",
        end_date: str = "2024-12-31",
    ) -> dict[str, int]:
        settings = get_settings()
        symbols = symbols or [f"{s}/USDT:USDT" for s in
                              [sym.replace("USDT", "") for sym in settings.trading.symbols]]
        results = {}
        for symbol in symbols:
            count = self.collect_funding_rates(symbol, start_date, end_date)
            results[symbol] = count
        return results
