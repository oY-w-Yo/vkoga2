"""Selection rules for the rebuilt VKOGA core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .validation import _as_1d_bool_array, _as_1d_finite_array


@dataclass(frozen=True)
class PointSelectionResult:
    """Result of selecting one candidate center."""

    candidate_index: int
    score: float
    log_score: float
    abs_residual: float
    power: float
    rule_name: str
    beta: float


@dataclass(frozen=True)
class BetaGreedyPointSelectionRule:
    """Unified beta-greedy selection rule over a finite candidate set.

    For finite ``beta``:

    ``score = |r(x)|^beta * P(x)^(1 - beta)``

    For ``beta = np.inf``:

    ``score = |r(x)| / P(x)``

    ``power_floor`` protects denominators in the ``beta=np.inf`` case and
    protects fractional powers in mixed residual/power rules.  Selection uses
    log-scores for numerical stability.  ``PointSelectionResult.score`` is the actual
    score for human-readable diagnostics, while ``PointSelectionResult.log_score`` is
    the value used for comparison.
    """

    beta: float
    power_floor: float = 1e-300
    residual_floor: float = 1e-300
    rule_name_override: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "beta", self._validate_beta(self.beta))
        power_floor = float(self.power_floor)
        residual_floor = float(self.residual_floor)
        if not np.isfinite(power_floor) or power_floor <= 0.0:
            raise ValueError("power_floor must be finite and > 0.")
        if not np.isfinite(residual_floor) or residual_floor < 0.0:
            raise ValueError("residual_floor must be finite and >= 0.")
        object.__setattr__(self, "power_floor", power_floor)
        object.__setattr__(self, "residual_floor", residual_floor)

    @staticmethod
    def _validate_beta(beta: float) -> float:
        """Return ``beta`` as a non-negative float or positive infinity."""
        beta = float(beta)
        if np.isinf(beta) and beta > 0:
            return beta
        if not np.isfinite(beta) or beta < 0.0:
            raise ValueError("beta must be non-negative or positive infinity.")
        return beta

    @property
    def rule_name(self) -> str:
        if self.rule_name_override:
            return self.rule_name_override
        if np.isinf(self.beta):
            return "f_over_p_greedy"
        if self.beta == 0.0:
            return "p_greedy"
        if self.beta == 0.5:
            return "f_times_p_greedy"
        if self.beta == 1.0:
            return "f_greedy"
        return "beta_greedy"

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "beta": "inf" if np.isinf(self.beta) else float(self.beta),
            "power_floor": float(self.power_floor),
            "residual_floor": float(self.residual_floor),
        }

    def _compute_scores(self, abs_residual: np.ndarray, power: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if np.isinf(self.beta):
                max_res = np.maximum(abs_residual, self.residual_floor)
                max_pow = np.maximum(power, self.power_floor)
                return max_res / max_pow
            if self.beta == 0.0:
                return np.maximum(power, self.power_floor)
            if self.beta == 0.5:
                max_res = np.maximum(abs_residual, self.residual_floor)
                max_pow = np.maximum(power, self.power_floor)
                return np.sqrt(max_res) * np.sqrt(max_pow)
            if self.beta == 1.0:
                return np.maximum(abs_residual, self.residual_floor)
            residual_term = np.maximum(abs_residual, self.residual_floor) ** self.beta
            power_term = np.maximum(power, self.power_floor) ** (1.0 - self.beta)
            return residual_term * power_term

    def _compute_log_scores(self, abs_residual: np.ndarray, power: np.ndarray) -> np.ndarray:
        residual_safe = np.maximum(abs_residual, self.residual_floor)
        power_safe = np.maximum(power, self.power_floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_residual = np.log(residual_safe)
            log_power = np.log(power_safe)
            if np.isinf(self.beta):
                return log_residual - log_power
            if self.beta == 0.0:
                return log_power
            if self.beta == 0.5:
                return 0.5 * (log_residual + log_power)
            if self.beta == 1.0:
                return log_residual
            return self.beta * log_residual + (1.0 - self.beta) * log_power

    def select(self, abs_residual, power, selected_mask) -> PointSelectionResult:
        """Select the next unselected candidate with the largest beta-greedy score.

        Selection uses log-scores for numerical stability and performance (log is
        cheaper than floating-point power).  Since log is strictly increasing,
        ``argmax(log_score) == argmax(score)``, so the choice is identical to
        using the raw scores.  The raw ``score`` is computed after selection for
        readable history records.
        """
        abs_residual_arr = _as_1d_finite_array(abs_residual, name="abs_residual")
        power_arr = _as_1d_finite_array(power, name="power")
        selected_mask_arr = _as_1d_bool_array(selected_mask, name="selected_mask")

        if (abs_residual_arr.shape != power_arr.shape
                or abs_residual_arr.shape != selected_mask_arr.shape):
            raise ValueError(
                "abs_residual, power, and selected_mask must have the same shape; "
                f"got {abs_residual_arr.shape}, {power_arr.shape}, and {selected_mask_arr.shape}."
            )
        if np.any(abs_residual_arr < 0.0) or np.any(power_arr < 0.0):
            raise ValueError("abs_residual and power must be non-negative.")

        available = ~selected_mask_arr
        if not np.any(available):
            raise ValueError(
                "Cannot select a candidate because "
                "all candidates are already selected."
            )

        # Log-scores: cheaper and more stable than raw scores, identical argmax.
        log_scores = self._compute_log_scores(abs_residual_arr, power_arr)
        if np.any(np.isnan(log_scores)) or np.any(np.isposinf(log_scores)):
            raise ValueError("Selection log-scores must not be NaN or positive infinity.")

        masked_log_scores = np.where(available, log_scores, -np.inf)
        if np.all(np.isneginf(masked_log_scores)):
            candidate_index = int(np.nonzero(available)[0][0])
        else:
            candidate_index = int(np.argmax(masked_log_scores))
        score = self._compute_scores(
            abs_residual_arr[[candidate_index]],
            power_arr[[candidate_index]],
        )[0]
        return PointSelectionResult(
            candidate_index=candidate_index,
            score=float(score),
            log_score=float(log_scores[candidate_index]),
            abs_residual=float(abs_residual_arr[candidate_index]),
            power=float(power_arr[candidate_index]),
            rule_name=self.rule_name,
            beta=float(self.beta),
        )


def point_selection_rule_from_config(
    rule: str, *, beta=None, **kwargs
) -> BetaGreedyPointSelectionRule:
    """Construct a beta-greedy selection rule from a stable config key."""
    key = str(rule).strip().lower()
    if key == "p_greedy":
        return BetaGreedyPointSelectionRule(beta=0.0, rule_name_override="p_greedy", **kwargs)
    if key == "f_times_p_greedy":
        return BetaGreedyPointSelectionRule(
            beta=0.5, rule_name_override="f_times_p_greedy", **kwargs
        )
    if key == "f_greedy":
        return BetaGreedyPointSelectionRule(
            beta=1.0, rule_name_override="f_greedy", **kwargs
        )
    if key == "f_over_p_greedy":
        return BetaGreedyPointSelectionRule(
            beta=np.inf, rule_name_override="f_over_p_greedy", **kwargs
        )
    if key == "beta_greedy":
        if beta is None:
            raise ValueError("point_selection_rule_from_config('beta_greedy') requires beta=...")
        return BetaGreedyPointSelectionRule(beta=beta, rule_name_override="beta_greedy", **kwargs)
    raise ValueError(
        f"Unsupported selection rule '{rule}'. "
        "Supported rules: p_greedy, f_times_p_greedy, f_greedy, f_over_p_greedy, beta_greedy."
    )
