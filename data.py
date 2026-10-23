"""Training data packaging for the rebuilt VKOGA core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .validation import _as_1d_finite_array, _as_valid_finite_point_matrix


@dataclass(frozen=True)
class TrainingData:
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


def build_training_data(X, y) -> TrainingData:
    """Validate and package greedy training data.

    ``X`` contains training points and ``y`` contains the corresponding target
    values. Greedy centers are selected from rows of ``X``.
    """
    X_candidates = _as_valid_finite_point_matrix(X, name="X")
    y_candidates = _as_1d_finite_array(y, name="y")
    n_training_points = int(X_candidates.shape[0])

    if n_training_points <= 0:
        raise ValueError("X must contain at least one data point.")
    if y_candidates.shape[0] != n_training_points:
        raise ValueError(
            "len(y) must match len(X); "
            f"got len(y)={y_candidates.shape[0]} and len(X)={n_training_points}."
        )

    return TrainingData(
        X_candidates=X_candidates,
        y_candidates=y_candidates,
        n_training_points=n_training_points,
    )


PDE_FUNCTIONAL_INTERIOR = 0
PDE_FUNCTIONAL_DIRICHLET = 1


@dataclass(frozen=True)
class PDETrainingData:
    """Validated collocation data for Phase-1 PDE greedy solves.

    Interior rows enforce ``L[u] = f``.  Dirichlet rows enforce ``u = g_D``.
    The solver consumes the concatenated trial arrays through ``X_all``,
    ``rhs_all``, and ``functional_types``.
    """

    X_int: np.ndarray
    f_vals: np.ndarray
    X_dir: np.ndarray
    g_dir: np.ndarray

    @property
    def n_int(self) -> int:
        return int(self.X_int.shape[0])

    @property
    def n_dir(self) -> int:
        return int(self.X_dir.shape[0])

    @property
    def n_trial(self) -> int:
        return self.n_int + self.n_dir

    @property
    def dim(self) -> int:
        return int(self.X_all.shape[1])

    @property
    def X_all(self) -> np.ndarray:
        return np.vstack((self.X_int, self.X_dir))

    @property
    def rhs_all(self) -> np.ndarray:
        return np.concatenate((self.f_vals, self.g_dir))

    @property
    def functional_types(self) -> np.ndarray:
        return np.concatenate((
            np.full(self.n_int, PDE_FUNCTIONAL_INTERIOR, dtype=int),
            np.full(self.n_dir, PDE_FUNCTIONAL_DIRICHLET, dtype=int),
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_trial": self.n_trial,
            "n_int": self.n_int,
            "n_dir": self.n_dir,
            "dim": self.dim,
            "functional_type_counts": {
                "interior": self.n_int,
                "dirichlet": self.n_dir,
            },
        }


def _empty_point_matrix(dim: int) -> np.ndarray:
    return np.empty((0, int(dim)), dtype=float)


def build_pde_training_data(X_int, f_vals, X_dir=None, g_dir=None) -> PDETrainingData:
    """Validate and package Phase-1 PDE greedy training data."""
    X_int_arr = _as_valid_finite_point_matrix(X_int, name="X_int")
    f_arr = _as_1d_finite_array(f_vals, name="f_vals")
    if f_arr.shape[0] != X_int_arr.shape[0]:
        raise ValueError(
            "len(f_vals) must match len(X_int); "
            f"got len(f_vals)={f_arr.shape[0]} and len(X_int)={X_int_arr.shape[0]}."
        )

    dim = int(X_int_arr.shape[1])
    if X_dir is None:
        X_dir_arr = _empty_point_matrix(dim)
    else:
        X_dir_arr = _as_valid_finite_point_matrix(X_dir, name="X_dir")
        if X_dir_arr.shape[1] != dim:
            raise ValueError(
                "X_int and X_dir must have the same feature dimension; "
                f"got X_int.shape={X_int_arr.shape} and X_dir.shape={X_dir_arr.shape}."
            )

    if g_dir is None:
        if X_dir_arr.shape[0] != 0:
            raise ValueError("g_dir is required when X_dir contains Dirichlet points.")
        g_dir_arr = np.empty((0,), dtype=float)
    else:
        g_dir_arr = _as_1d_finite_array(g_dir, name="g_dir")
    if g_dir_arr.shape[0] != X_dir_arr.shape[0]:
        raise ValueError(
            "len(g_dir) must match len(X_dir); "
            f"got len(g_dir)={g_dir_arr.shape[0]} and len(X_dir)={X_dir_arr.shape[0]}."
        )

    n_trial = int(X_int_arr.shape[0] + X_dir_arr.shape[0])
    if n_trial <= 0:
        raise ValueError("PDE training data must contain at least one trial point.")

    return PDETrainingData(
        X_int=X_int_arr,
        f_vals=f_arr,
        X_dir=X_dir_arr,
        g_dir=g_dir_arr,
    )
