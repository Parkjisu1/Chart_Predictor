"""Strategy parameters and signal dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json


@dataclass
class StrategyParameters:
    """All tunable strategy parameters with sensible defaults."""

    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_divergence_lookback: int = 20

    # Bollinger Bands
    bb_period: int = 20
    bb_std_dev: float = 2.0
    bb_squeeze_threshold: float = 0.05

    # Volume
    obv_ma_period: int = 20
    vwap_deviation_threshold: float = 2.0
    volume_spike_multiplier: float = 2.0

    # GARCH
    garch_p: int = 1
    garch_q: int = 1
    garch_high_vol_threshold: float = 0.03
    garch_low_vol_threshold: float = 0.01

    # Momentum
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_period: int = 14
    adx_strong_trend: float = 25.0
    adx_weak_trend: float = 20.0

    # Composite Weights (기존 기술지표)
    weight_rsi: float = 0.10
    weight_bollinger: float = 0.08
    weight_volume: float = 0.07
    weight_garch: float = 0.05
    weight_momentum: float = 0.12
    weight_sentiment: float = 0.08

    # Composite Weights (서적 기반 전략)
    weight_williams: float = 0.08       # Larry Williams
    weight_elder: float = 0.08          # Alexander Elder
    weight_ichimoku: float = 0.08       # 일목균형표
    weight_market_structure: float = 0.08  # John Murphy
    weight_patterns: float = 0.08       # Minervini/Weinstein/Wyckoff

    # Composite Weights (퀀트 데이터)
    weight_quant: float = 0.10          # 퀀트 종합 (OI, 롱숏, 공포탐욕, 고래)

    # Signal Thresholds
    strong_signal_threshold: float = 0.7
    signal_threshold: float = 0.3

    # Risk
    max_position_pct: float = 0.25
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    trailing_stop_activation: float = 0.02  # 2% profit
    trailing_stop_distance: float = 0.01    # 1% trailing

    # Parameter boundaries for learning loop
    BOUNDARIES: dict = field(default_factory=lambda: {
        "rsi_period": (7, 28),
        "rsi_overbought": (60, 85),
        "rsi_oversold": (15, 40),
        "bb_period": (10, 40),
        "bb_std_dev": (1.5, 3.0),
        "macd_fast": (8, 20),
        "macd_slow": (20, 40),
        "macd_signal": (5, 15),
        "adx_strong_trend": (20, 35),
        "weight_rsi": (0.05, 0.40),
        "weight_bollinger": (0.05, 0.30),
        "weight_volume": (0.05, 0.30),
        "weight_garch": (0.05, 0.20),
        "weight_momentum": (0.10, 0.40),
        "weight_sentiment": (0.02, 0.20),
        "weight_williams": (0.02, 0.20),
        "weight_elder": (0.02, 0.20),
        "weight_ichimoku": (0.02, 0.20),
        "weight_market_structure": (0.02, 0.20),
        "weight_patterns": (0.02, 0.20),
        "weight_quant": (0.02, 0.20),
        "strong_signal_threshold": (0.5, 0.9),
        "signal_threshold": (0.1, 0.5),
        "stop_loss_atr_multiplier": (1.0, 4.0),
        "take_profit_atr_multiplier": (1.5, 6.0),
    })

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("BOUNDARIES", None)
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_str: str) -> StrategyParameters:
        d = json.loads(json_str)
        d.pop("BOUNDARIES", None)
        return cls(**d)

    def clone(self) -> StrategyParameters:
        return StrategyParameters.from_json(self.to_json())

    def normalize_weights(self) -> None:
        """Ensure all composite weights sum to 1.0."""
        weight_names = [
            "weight_rsi", "weight_bollinger", "weight_volume",
            "weight_garch", "weight_momentum", "weight_sentiment",
            "weight_williams", "weight_elder", "weight_ichimoku",
            "weight_market_structure", "weight_patterns", "weight_quant",
        ]
        total = sum(getattr(self, w) for w in weight_names)
        if total > 0:
            for w in weight_names:
                setattr(self, w, getattr(self, w) / total)


@dataclass
class SignalOutput:
    """Output from a single indicator or agent."""
    name: str
    value: float          # -1.0 (strong short) to +1.0 (strong long)
    confidence: float     # 0.0 to 1.0
    details: dict = field(default_factory=dict)


@dataclass
class CompositeSignal:
    """Aggregated signal from all indicators."""
    score: float          # -1.0 to +1.0
    signal: str           # Signal enum value
    confidence: float     # 0.0 to 1.0
    components: list[SignalOutput] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
