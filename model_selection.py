"""Model-selection rules for rebuilt VKOGA paths.

This module is intentionally independent from the eventual full history object.
For now, rules operate on simple mapping records.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class BaseModelSelectionRule(ABC):
    """Base class for model-selection rules.

    Subclasses must implement ``select`` and ``to_dict`` and provide a
    ``rule_name`` attribute.
    """

    @property
    @abstractmethod
    def rule_name(self) -> str:
        ...

    @abstractmethod
    def select(
        self, records: Sequence[Mapping[str, Any]]
    ) -> ModelSelectionResult:
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ModelSelectionResult:
    retained_iteration: int
    retained_n_centers: int
    score_name: str
    score_value: float
    rule_name: str

    def to_dict(self) -> dict[str, Any]:
        score_value = float(self.score_value)
        return {
            "retained_iteration": int(self.retained_iteration),
            "retained_n_centers": int(self.retained_n_centers),
            "score_name": str(self.score_name),
            "score_value": score_value if np.isfinite(score_value) else None,
            "rule_name": str(self.rule_name),
        }


def _validate_records(records: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    if len(records) == 0:
        raise ValueError("model selection requires at least one history record.")
    return records


def _record_int(record: Mapping[str, Any], key: str) -> int:
    if key not in record:
        raise ValueError(f"history record is missing required key '{key}'.")
    return int(record[key])


@dataclass(frozen=True)
class KeepLastModelSelectionRule(BaseModelSelectionRule):
    """Retain the final greedy path record."""

    rule_name: str = "keep_last"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
        }

    def select(self, records: Sequence[Mapping[str, Any]]) -> ModelSelectionResult:
        records = _validate_records(records)
        record = records[-1]
        return ModelSelectionResult(
            retained_iteration=_record_int(record, "iteration"),
            retained_n_centers=_record_int(record, "n_selected"),
            score_name="last_record",
            score_value=float("nan"),
            rule_name=self.rule_name,
        )


@dataclass(frozen=True)
class BestScoreModelSelectionRule(BaseModelSelectionRule):
    """Retain the earliest record with the minimum named score."""

    score_name: str = "residual_max"
    rule_name: str = "best_score"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "score_name": self.score_name,
        }

    def select(self, records: Sequence[Mapping[str, Any]]) -> ModelSelectionResult:
        records = _validate_records(records)
        best_index = None
        best_score = None
        for i, record in enumerate(records):
            if self.score_name not in record:
                raise ValueError(
                    f"history record at index {i} is missing score '{self.score_name}'."
                )
            score = float(record[self.score_name])
            if not np.isfinite(score):
                raise ValueError(
                    f"score '{self.score_name}' at record index {i} must be finite; got {score!r}."
                )
            if best_score is None or score < best_score:
                best_index = i
                best_score = score

        assert best_index is not None  # guaranteed by _validate_records
        assert best_score is not None
        record = records[best_index]
        return ModelSelectionResult(
            retained_iteration=_record_int(record, "iteration"),
            retained_n_centers=_record_int(record, "n_selected"),
            score_name=self.score_name,
            score_value=best_score,
            rule_name=self.rule_name,
        )


def model_selection_rule_from_config(rule: str, **kwargs):
    """Construct a model-selection rule from a stable config key."""
    key = str(rule).strip().lower()
    if key == "keep_last":
        return KeepLastModelSelectionRule(**kwargs)
    if key == "best_score":
        return BestScoreModelSelectionRule(**kwargs)
    if key == "keep_best_rmse":
        return BestScoreModelSelectionRule(
            score_name="residual_rmse",
            rule_name="keep_best_rmse",
        )
    if key == "keep_best_rmax":
        return BestScoreModelSelectionRule(
            score_name="residual_max",
            rule_name="keep_best_rmax",
        )
    raise ValueError(
        f"Unsupported model selection rule '{rule}'. "
        "Supported rules: keep_last, best_score, keep_best_rmse, "
        "keep_best_rmax."
    )
