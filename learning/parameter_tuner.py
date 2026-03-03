"""Parameter tuner with boundary constraints."""

from __future__ import annotations

import random
from copy import deepcopy

from config.constants import PARAMETER_PERTURBATION
from config.logging_config import get_logger
from strategy.signals import StrategyParameters

logger = get_logger(__name__)


class ParameterTuner:
    """Adjust strategy parameters within defined boundaries."""

    def __init__(self, perturbation: float = PARAMETER_PERTURBATION):
        self.perturbation = perturbation

    def apply_adjustments(
        self,
        params: StrategyParameters,
        adjustments: dict[str, any],
    ) -> StrategyParameters:
        """Apply parameter adjustments from Claude insights or rules."""
        new_params = params.clone()
        boundaries = params.BOUNDARIES

        for param_name, value in adjustments.items():
            if not hasattr(new_params, param_name):
                continue

            current = getattr(new_params, param_name)

            if isinstance(value, str):
                # Parse directive strings like "increase_10pct"
                new_value = self._parse_directive(current, value)
            elif isinstance(value, (int, float)):
                new_value = value
            else:
                continue

            # Enforce boundaries
            if param_name in boundaries:
                low, high = boundaries[param_name]
                new_value = max(low, min(high, new_value))
                if isinstance(current, int):
                    new_value = int(round(new_value))

            setattr(new_params, param_name, new_value)
            logger.info("param_adjusted", param=param_name,
                         old=current, new=new_value)

        # Normalize weights after adjustment
        new_params.normalize_weights()

        return new_params

    def random_perturbation(
        self, params: StrategyParameters
    ) -> StrategyParameters:
        """Apply random perturbation within boundaries (for stagnation escape)."""
        new_params = params.clone()
        boundaries = params.BOUNDARIES

        # Perturb 3-5 random parameters
        tunable = [k for k in boundaries.keys() if hasattr(new_params, k)]
        n_perturb = min(random.randint(3, 5), len(tunable))
        selected = random.sample(tunable, n_perturb)

        for param_name in selected:
            current = getattr(new_params, param_name)
            low, high = boundaries[param_name]

            # Random perturbation within +-perturbation%
            delta = current * self.perturbation * random.uniform(-1, 1)
            new_value = current + delta
            new_value = max(low, min(high, new_value))

            if isinstance(current, int):
                new_value = int(round(new_value))

            setattr(new_params, param_name, new_value)
            logger.info("random_perturbation", param=param_name,
                         old=current, new=new_value)

        new_params.normalize_weights()
        return new_params

    def _parse_directive(self, current: float, directive: str) -> float:
        """Parse adjustment directives like 'increase_10pct'."""
        parts = directive.lower().split("_")
        if len(parts) < 2:
            return current

        action = parts[0]
        pct_str = parts[-1].replace("pct", "")
        try:
            pct = float(pct_str) / 100
        except ValueError:
            pct = 0.05

        if action == "increase":
            return current * (1 + pct)
        elif action == "decrease":
            return current * (1 - pct)
        else:
            return current
