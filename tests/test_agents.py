"""Tests for agent pipeline."""

import numpy as np
import pandas as pd
import pytest

from agents.base import AgentResult
from agents.technical_analyst import TechnicalAnalystAgent
from agents.risk_reviewer import RiskReviewerAgent
from agents.final_decider import FinalDeciderAgent
from strategy.signals import StrategyParameters


def make_df(n=200, trend="up"):
    np.random.seed(42)
    base = 50000
    if trend == "up":
        prices = base + np.cumsum(np.random.randn(n) * 100 + 5)
    else:
        prices = base + np.cumsum(np.random.randn(n) * 100 - 5)
    return pd.DataFrame({
        "open": prices + np.random.randn(n) * 50,
        "high": prices + abs(np.random.randn(n) * 100),
        "low": prices - abs(np.random.randn(n) * 100),
        "close": prices,
        "volume": np.random.uniform(100, 1000, n),
    })


class TestTechnicalAnalyst:
    def test_analyze(self):
        agent = TechnicalAnalystAgent()
        df = make_df(200)
        result = agent.analyze(df=df)
        assert isinstance(result, AgentResult)
        assert result.error is None
        assert -1.0 <= result.signal_value <= 1.0

    def test_insufficient_data(self):
        agent = TechnicalAnalystAgent()
        df = make_df(10)
        result = agent.analyze(df=df)
        assert result.error is not None

    def test_parameter_update(self):
        agent = TechnicalAnalystAgent()
        new_params = StrategyParameters(rsi_period=21)
        agent.update_parameters(new_params)
        assert agent.params.rsi_period == 21


class TestRiskReviewer:
    def test_approve_strong_signal(self):
        agent = RiskReviewerAgent()
        tech_result = AgentResult(
            agent_name="tech",
            signal_value=0.6,
            confidence=0.7,
        )
        result = agent.analyze(technical_result=tech_result)
        assert result.details["approved"]

    def test_reject_weak_signal(self):
        agent = RiskReviewerAgent()
        tech_result = AgentResult(
            agent_name="tech",
            signal_value=0.05,
            confidence=0.2,
        )
        result = agent.analyze(technical_result=tech_result)
        assert not result.details["approved"]

    def test_reject_high_drawdown(self):
        agent = RiskReviewerAgent()
        tech_result = AgentResult(
            agent_name="tech",
            signal_value=0.6,
            confidence=0.7,
        )
        result = agent.analyze(
            technical_result=tech_result,
            current_drawdown=0.20,
        )
        assert not result.details["approved"]

    def test_no_input(self):
        agent = RiskReviewerAgent()
        result = agent.analyze()
        assert result.error is not None


class TestFinalDecider:
    def test_execute_approved(self):
        decider = FinalDeciderAgent(max_leverage=5)
        risk_result = AgentResult(
            agent_name="risk",
            signal_value=0.6,
            confidence=0.6,
            details={
                "approved": True,
                "max_position_pct": 0.2,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.03,
            },
        )
        result = decider.analyze(
            risk_result=risk_result,
            current_capital=100000,
            current_price=50000,
            historical_win_rate=0.6,
        )
        details = result.details
        assert details["execute"]
        assert details["side"] in ("long", "short")
        assert details["leverage"] >= 1

    def test_reject_not_approved(self):
        decider = FinalDeciderAgent()
        risk_result = AgentResult(
            agent_name="risk",
            signal_value=0.0,
            confidence=0.0,
            details={"approved": False},
        )
        result = decider.analyze(risk_result=risk_result)
        assert not result.details.get("execute", False)

    def test_half_kelly(self):
        decider = FinalDeciderAgent()
        kelly = decider._half_kelly(0.6, 1.5, 0.8)
        assert 0 < kelly < 0.25

    def test_kelly_negative(self):
        decider = FinalDeciderAgent()
        kelly = decider._half_kelly(0.2, 0.5, 0.8)
        assert kelly == 0.0
