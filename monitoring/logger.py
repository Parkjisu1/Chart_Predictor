"""Trade logger for persistent trade recording."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from config.logging_config import get_logger

logger = get_logger(__name__)


class TradeLogger:
    """Log all trades to a persistent JSON lines file."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def log_trade(self, trade_data: dict) -> None:
        """Append trade record to log file."""
        trade_data["logged_at"] = datetime.now().isoformat()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trade_data) + "\n")

    def log_signal(self, signal_data: dict) -> None:
        """Log signal decision."""
        signal_data["type"] = "signal"
        signal_data["logged_at"] = datetime.now().isoformat()
        signal_file = self.log_dir / f"signals_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(signal_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(signal_data) + "\n")
