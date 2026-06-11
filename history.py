"""Scalar history records for rebuilt VKOGA greedy paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class GreedyStepRecord:
    """Scalar summary for one successful greedy step."""
    # Step identity
    iteration: int  # 1-based greedy step index.
    n_selected: int  # Number of centers selected after this step.

    # Point selection details
    selected_candidate_index: int  # Row index in the training data selected as the new center.
    selection_rule_name: str  # Point-selection rule used at this step.
    selection_score: float  # Raw point-selection score for the selected center.
    selection_log_score: float  # Log-score used for stable score reporting.
    selected_abs_residual: float  # Absolute residual at the selected center before update.
    selected_power: float  # Power-function value at the selected center before update.

    # Model update details
    normalizer: float  # Newton basis normalizer for the selected center.
    coefficient: float  # New Newton coefficient appended at this step.

    # Model indicators after update
    residual_rmse: float  # Residual root-mean-square value after update.
    residual_max: float  # Max absolute residual over training points after update.
    power_max: float  # Max power-function value over all training points after update.

    # Condition number (optional, only when store_cond=True and SPD and n≤2000)
    condition_number: Optional[float] = None  # λ_max/λ_min of K(C, C) after update.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GreedyHistory:
    """Collection of scalar greedy-step records."""

    records: List[GreedyStepRecord]

    def append(self, record: GreedyStepRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def records_to_dicts(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.records]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_steps": len(self),
            "records": self.records_to_dicts(),
        }

    def get_field_values(self, name: str) -> np.ndarray:
        values = []
        for i, record in enumerate(self.records):
            if not hasattr(record, name):
                raise ValueError(f"Unknown history field '{name}' at record index {i}.")
            values.append(getattr(record, name))
        return np.asarray(values, dtype=float)
