"""Training data packaging for the rebuilt VKOGA core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .validation import as_valid_point_matrix


@dataclass(frozen=True)
class GreedyTrainingData:
    """Validated training data for a greedy kernel solve.

    ``X_candidates`` stores the training points from which centers may be
    selected. ``y_candidates`` stores the corresponding target values.
    """

    X_candidates: np.ndarray
    y_candidates: np.ndarray
    n_training_points: int

    @property
    def dim(self) -> int:
        return int(self.X_candidates.shape[1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_training_points": self.n_training_points,
            "dim": self.dim,
        }


def _as_valid_target_vector(y, *, name: str) -> np.ndarray:
    """Return ``y`` as a finite one-dimensional float vector."""
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must have shape (n,); got array with shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_valid_finite_point_matrix(X, *, name: str) -> np.ndarray:
    """Return ``X`` as a finite floating-point matrix with shape ``(n, d)``."""
    arr = as_valid_point_matrix(X, name=name)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def build_greedy_training_data(X, y) -> GreedyTrainingData:
    """Validate and package greedy training data.

    ``X`` contains training points and ``y`` contains the corresponding target
    values. Greedy centers are selected from rows of ``X``.
    """
    X_candidates = _as_valid_finite_point_matrix(X, name="X")
    y_candidates = _as_valid_target_vector(y, name="y")
    n_training_points = int(X_candidates.shape[0])

    if n_training_points <= 0:
        raise ValueError("X must contain at least one data point.")
    if y_candidates.shape[0] != n_training_points:
        raise ValueError(
            "len(y) must match len(X); "
            f"got len(y)={y_candidates.shape[0]} and len(X)={n_training_points}."
        )

    return GreedyTrainingData(
        X_candidates=X_candidates,
        y_candidates=y_candidates,
        n_training_points=n_training_points,
    )
