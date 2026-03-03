"""Enumerations, fee constants, and project-wide constants."""

from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class Signal(str, Enum):
    STRONG_LONG = "strong_long"
    LONG = "long"
    NEUTRAL = "neutral"
    SHORT = "short"
    STRONG_SHORT = "strong_short"


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"


class FailureMode(str, Enum):
    """8 failure modes for trade analysis."""
    EARLY_ENTRY = "early_entry"
    LATE_ENTRY = "late_entry"
    WRONG_DIRECTION = "wrong_direction"
    OVERLEVERAGED = "overleveraged"
    NO_STOP_LOSS = "no_stop_loss"
    PREMATURE_EXIT = "premature_exit"
    TREND_REVERSAL = "trend_reversal"
    HIGH_VOLATILITY = "high_volatility"


# --- Fee Constants (Bybit Futures) ---
MAKER_FEE = 0.0002    # 0.02%
TAKER_FEE = 0.00055   # 0.055%
FUNDING_INTERVAL_HOURS = 8

# --- Risk Limits ---
MAX_POSITION_PCT = 0.25          # 25% of capital per position
MAX_TOTAL_EXPOSURE_PCT = 0.50    # 50% total exposure
MAX_DAILY_LOSS_PCT = 0.05        # 5% daily loss -> kill switch
MAX_DRAWDOWN_PCT = 0.15          # 15% drawdown -> kill switch
CORRELATION_THRESHOLD = 0.7      # Reject if asset correlation > 0.7

# --- Backtest ---
MIN_BARS_REQUIRED = 500
WALK_FORWARD_IN_SAMPLE_PCT = 0.7
WALK_FORWARD_OUT_SAMPLE_PCT = 0.3
MONTE_CARLO_SIMULATIONS = 1000

# --- Learning Loop ---
TARGET_WIN_RATE = 0.90
OOS_MIN_WIN_RATE = 0.85
MAX_ITERATIONS = 50
STAGNATION_LIMIT = 5
PARAMETER_PERTURBATION = 0.10    # 10% random perturbation on stagnation

# --- Data Collection ---
DEFAULT_TIMEFRAMES = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]
COLLECT_START_DATE = "2021-01-01"
COLLECT_END_DATE = "2024-12-31"
