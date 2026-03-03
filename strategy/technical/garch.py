"""GARCH(1,1) volatility model for regime detection."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from strategy.signals import StrategyParameters, SignalOutput

warnings.filterwarnings("ignore", category=RuntimeWarning)


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> dict | None:
    """Fit GARCH(p,q) model and return forecast volatility."""
    try:
        from arch import arch_model

        clean_returns = returns.dropna() * 100  # Scale for numerical stability
        if len(clean_returns) < 100:
            return None

        model = arch_model(clean_returns, vol="Garch", p=p, q=q,
                           mean="Constant", dist="normal")
        result = model.fit(disp="off", show_warning=False)

        forecast = result.forecast(horizon=1)
        forecast_var = forecast.variance.iloc[-1, 0]
        forecast_vol = np.sqrt(forecast_var) / 100  # Unscale

        return {
            "forecast_vol": forecast_vol,
            "current_vol": result.conditional_volatility.iloc[-1] / 100,
            "params": {
                "omega": result.params.get("omega", 0),
                "alpha": result.params.get("alpha[1]", 0),
                "beta": result.params.get("beta[1]", 0),
            },
        }
    except Exception:
        return None


def analyze_garch(df: pd.DataFrame, params: StrategyParameters) -> SignalOutput:
    """Analyze volatility regime using GARCH model."""
    close = df["close"]
    returns = close.pct_change().dropna()

    garch_result = fit_garch(returns, params.garch_p, params.garch_q)

    if garch_result is None:
        # Fallback to simple volatility
        simple_vol = returns.rolling(window=20).std().iloc[-1] if len(returns) > 20 else 0
        return SignalOutput(
            name="garch",
            value=0.0,
            confidence=0.2,
            details={"fallback": True, "simple_vol": round(simple_vol, 6)},
        )

    forecast_vol = garch_result["forecast_vol"]
    current_vol = garch_result["current_vol"]

    # High volatility -> reduce signal / caution
    # Low volatility -> potential breakout setup
    if forecast_vol > params.garch_high_vol_threshold:
        value = 0.0  # Neutral: high vol = uncertain direction
        confidence = 0.3  # Low confidence in any direction
    elif forecast_vol < params.garch_low_vol_threshold:
        value = 0.0  # Neutral but high confidence (range-bound)
        confidence = 0.7
    else:
        # Normal vol regime
        value = 0.0
        confidence = 0.5

    # Vol expansion/contraction signal
    if current_vol > 0:
        vol_change = (forecast_vol - current_vol) / current_vol
        if vol_change > 0.2:
            confidence *= 0.7  # Expanding vol -> reduce confidence
        elif vol_change < -0.2:
            confidence = min(confidence * 1.2, 1.0)  # Contracting vol -> breakout likely

    return SignalOutput(
        name="garch",
        value=value,
        confidence=min(confidence, 1.0),
        details={
            "forecast_vol": round(forecast_vol, 6),
            "current_vol": round(current_vol, 6),
            "regime": (
                "high" if forecast_vol > params.garch_high_vol_threshold
                else "low" if forecast_vol < params.garch_low_vol_threshold
                else "normal"
            ),
        },
    )
