"""System health checks for the trading bot."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class HealthStatus:
    healthy: bool
    checks: dict[str, bool]
    details: dict[str, str]
    timestamp: str


class HealthChecker:
    """Run health checks on system components."""

    def __init__(self):
        self.last_check: HealthStatus | None = None
        self.last_trade_time: float = 0
        self.last_data_time: float = 0
        self.error_count: int = 0
        self.max_errors: int = 10

    def update_trade_time(self) -> None:
        self.last_trade_time = time.time()

    def update_data_time(self) -> None:
        self.last_data_time = time.time()

    def record_error(self) -> None:
        self.error_count += 1

    def reset_errors(self) -> None:
        self.error_count = 0

    def check(
        self,
        exchange_connected: bool = True,
        db_connected: bool = True,
        websocket_connected: bool = True,
    ) -> HealthStatus:
        """Run all health checks."""
        now = time.time()
        checks = {}
        details = {}

        # Exchange connectivity
        checks["exchange"] = exchange_connected
        details["exchange"] = "connected" if exchange_connected else "disconnected"

        # Database
        checks["database"] = db_connected
        details["database"] = "connected" if db_connected else "disconnected"

        # WebSocket
        checks["websocket"] = websocket_connected
        details["websocket"] = "connected" if websocket_connected else "disconnected"

        # Data freshness (warn if no data for 5 min)
        data_age = now - self.last_data_time if self.last_data_time > 0 else 0
        checks["data_fresh"] = data_age < 300 or self.last_data_time == 0
        details["data_age_sec"] = f"{data_age:.0f}" if self.last_data_time > 0 else "N/A"

        # Error rate
        checks["error_rate"] = self.error_count < self.max_errors
        details["error_count"] = str(self.error_count)

        overall = all(checks.values())
        status = HealthStatus(
            healthy=overall,
            checks=checks,
            details=details,
            timestamp=datetime.now().isoformat(),
        )

        self.last_check = status

        if not overall:
            logger.warning("health_check_failed", checks=checks)
        else:
            logger.debug("health_check_passed")

        return status
