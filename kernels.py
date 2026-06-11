"""Kernel objects for the rebuilt VKOGA core.

The kernels in this module are immutable, pickleable, and use explicit
parameter semantics.  They intentionally do not implement a mutable
``set_params`` interface.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .distances import pairwise_distance
from .validation import (
    as_valid_point_matrix,
    validate_nonnegative_float,
    validate_positive_int,
)


@dataclass(frozen=True, kw_only=True)
class Kernel(ABC):
    """Base protocol for rebuilt kernel objects."""

    is_spd: bool = False
    """Whether this kernel is symmetric positive definite.

    Controls whether the solver computes a condition number in
    ``model_summary()`` (only when ``n_selected <= 2000``).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Concise stable kernel name."""

    @abstractmethod
    def eval(self, X, Y) -> np.ndarray:
        """Evaluate the kernel matrix ``K(X, Y)``."""

    def eval_prod(self, X, Y, v) -> np.ndarray:
        """Evaluate ``K(X, Y) @ v``.

        Faster implementations can override this later.
        """
        return self.eval(X, Y) @ np.asarray(v)

    @abstractmethod
    def diagonal(self, X) -> np.ndarray:
        """Return the kernel diagonal for rows of ``X``."""

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return concise machine-readable reconstruction parameters."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe diagnostic metadata."""


@dataclass(frozen=True, kw_only=True)
class RBFKernel(Kernel):
    """Base class for radial kernels evaluated from pairwise distances."""

    is_spd: bool = True
    distance_metric: str = "euclidean"

    @abstractmethod
    def evaluate_distances(self, r) -> np.ndarray:
        """Evaluate the radial kernel profile at distances ``r``."""

    def eval(self, X, Y) -> np.ndarray:
        """Evaluate ``K(X, Y)`` from pairwise distances.

        The current Euclidean ``pairwise_distance`` path avoids the naive
        ``(n, m, d)`` broadcast tensor and returns a distance block with shape
        ``(n, m)``. ``evaluate_distances`` then applies the radial profile
        elementwise to produce the kernel block with the same shape.
        """
        r = pairwise_distance(X, Y, metric=self.distance_metric)
        return self.evaluate_distances(r)

    def eval_prod(self, X, Y, v, chunk_size=None) -> np.ndarray:
        """Evaluate ``K(X, Y) @ v`` with optional row-chunking.

        The current Euclidean ``pairwise_distance`` path avoids the naive
        ``(n, m, d)`` broadcast tensor, but ``eval_prod`` still materialises a
        kernel block with shape ``(n, m)`` before multiplying by ``v``.
        Row-chunking bounds that block to ``(chunk, m)``.

        ``chunk_size`` resolution (first match wins):
          1. Explicit ``chunk_size`` argument (when not ``None``).
          2. ``VKOGA_EVAL_PROD_CHUNK_SIZE`` environment variable.
          3. Auto-estimated from ``VKOGA_EVAL_PROD_MEMORY_MB``
             (default 256 MB) — each row of the distance tensor costs
             ``m * d * 8`` bytes.
          4. No chunking when none of the above produces a value smaller
             than ``n``.
        """
        X_arr = as_valid_point_matrix(X, name="X")
        Y_arr = as_valid_point_matrix(Y, name="Y")
        v_arr = np.asarray(v, dtype=float).reshape(-1)

        n = X_arr.shape[0]
        m = Y_arr.shape[0]
        d = X_arr.shape[1]

        if d != Y_arr.shape[1]:
            raise ValueError(
                "X and Y must have the same feature dimension; "
                f"got X.shape={X_arr.shape} and Y.shape={Y_arr.shape}."
            )
        if m != v_arr.shape[0]:
            raise ValueError(
                "v length must match Y.shape[0]; "
                f"got len(v)={v_arr.shape[0]} and Y.shape[0]={m}."
            )

        if n == 0:
            return np.empty(0, dtype=float)

        # ---- resolve chunk_size ----
        if chunk_size is None:
            raw = os.environ.get("VKOGA_EVAL_PROD_CHUNK_SIZE", "").strip()
            if raw:
                chunk_size = int(raw)

        if chunk_size is None:
            memory_mb = float(os.environ.get("VKOGA_EVAL_PROD_MEMORY_MB", "256"))
            budget_bytes = int(memory_mb * 1024 * 1024 * 0.8)  # 80 % headroom
            # Each row of the distance intermediate costs m * d * 8 bytes.
            bytes_per_row = max(m * d * 8, 1)
            chunk_size = int(budget_bytes // bytes_per_row)

        if chunk_size is not None:
            chunk_size = max(1, int(chunk_size))

        # ---- evaluate ----
        if chunk_size is None or chunk_size >= n:
            return self.evaluate_distances(
                pairwise_distance(X_arr, Y_arr, metric=self.distance_metric)
            ) @ v_arr

        out = np.empty(n, dtype=float)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            r_chunk = pairwise_distance(
                X_arr[start:end], Y_arr, metric=self.distance_metric,
            )
            out[start:end] = self.evaluate_distances(r_chunk) @ v_arr
        return out

    def diagonal(self, X) -> np.ndarray:
        """Return ``K(X[i], X[i])`` for each row of ``X``.

        For radial kernels, every self-distance is zero, so the diagonal is the
        radial profile evaluated at ``r = 0`` and repeated once per input row.
        """
        X_arr = as_valid_point_matrix(X, name="X")
        value0 = np.asarray(
            self.evaluate_distances(np.asarray([0.0], dtype=float)),
            dtype=float,
        ).reshape(-1)[0]
        return np.full(X_arr.shape[0], value0, dtype=float)


@dataclass(frozen=True, kw_only=True)
class GaussianKernel(RBFKernel):
    """Gaussian RBF with direct-kernel gamma convention.

    ``K(r) = exp(-gamma * r^2)``
    """

    gamma: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "gamma", validate_nonnegative_float(self.gamma, name="gamma"))

    @property
    def name(self) -> str:
        return "gaussian"

    def evaluate_distances(self, r) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        return np.exp(-self.gamma * r_arr * r_arr)

    def get_params(self) -> Dict[str, Any]:
        return {
            "kernel_type": self.name,
            "gamma": self.gamma,
            "distance_metric": self.distance_metric,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_type": self.name,
            "name": self.name,
            "family": "rbf",
            "formula": "exp(-gamma * r^2)",
            "parameter_convention": "gamma is the coefficient in exp(-gamma * r^2)",
            "parameters": {
                "gamma": self.gamma,
                "distance_metric": self.distance_metric,
            },
        }


@dataclass(frozen=True, kw_only=True)
class MaternKernel(RBFKernel):
    """Base class for Matérn-family kernels.

    The rebuilt Matérn convention uses ``gamma`` as inverse length scale.
    """

    gamma: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "gamma", validate_nonnegative_float(self.gamma, name="gamma"))

    @property
    def name(self) -> str:
        return "matern"


@dataclass(frozen=True, kw_only=True)
class Matern32Kernel(MaternKernel):
    """Matérn 3/2 kernel with gamma as inverse length scale.

    ``K(r) = (1 + gamma * r) * exp(-gamma * r)``
    """

    @property
    def name(self) -> str:
        return "matern32"

    def evaluate_distances(self, r) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        return (1.0 + self.gamma * r_arr) * np.exp(- self.gamma * r_arr)

    def get_params(self) -> Dict[str, Any]:
        return {
            "kernel_type": self.name,
            "gamma": self.gamma,
            "distance_metric": self.distance_metric,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_type": self.name,
            "name": self.name,
            "family": "matern",
            "formula": "(1 + gamma * r) * exp(-gamma * r)",
            "parameter_convention": "gamma = inverse length scale",
            "parameters": {
                "gamma": self.gamma,
                "distance_metric": self.distance_metric,
            },
        }


@dataclass(frozen=True, kw_only=True)
class Matern52Kernel(MaternKernel):
    """Matérn 5/2 kernel with gamma as inverse length scale.

    ``K(r) = (1 + gamma*r + (gamma*r)^2/3) * exp(-gamma*r)``
    """

    @property
    def name(self) -> str:
        return "matern52"

    def evaluate_distances(self, r) -> np.ndarray:
        r_arr = np.asarray(r, dtype=float)
        a = self.gamma * r_arr
        return (1.0 + a + a * a / 3.0) * np.exp(-a)

    def get_params(self) -> Dict[str, Any]:
        return {
            "kernel_type": self.name,
            "gamma": self.gamma,
            "distance_metric": self.distance_metric,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_type": self.name,
            "name": self.name,
            "family": "matern",
            "formula": "(1 + gamma*r + (gamma*r)^2/3) * exp(-gamma*r)",
            "parameter_convention": "gamma = inverse length scale",
            "parameters": {
                "gamma": self.gamma,
                "distance_metric": self.distance_metric,
            },
        }


@dataclass(frozen=True, kw_only=True)
class PolynomialKernel(Kernel):
    """Polynomial kernel.

    ``K(x, y) = (offset + x dot y)^degree``
    """

    degree: int = 2
    offset: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "degree", validate_positive_int(self.degree, name="degree"))
        object.__setattr__(
            self,
            "offset",
            validate_nonnegative_float(self.offset, name="offset"),
        )

    @property
    def name(self) -> str:
        return "polynomial"

    def eval(self, X, Y) -> np.ndarray:
        """Evaluate the polynomial kernel matrix between rows of ``X`` and ``Y``."""
        X_arr = as_valid_point_matrix(X, name="X")
        Y_arr = as_valid_point_matrix(Y, name="Y")
        if X_arr.shape[1] != Y_arr.shape[1]:
            raise ValueError(
                "X and Y must have the same feature dimension; "
                f"got X.shape={X_arr.shape} and Y.shape={Y_arr.shape}."
            )
        return (self.offset + X_arr @ Y_arr.T) ** self.degree

    def diagonal(self, X) -> np.ndarray:
        """Return ``K(X[i], X[i])`` for each row of ``X``."""
        X_arr = as_valid_point_matrix(X, name="X")
        return (self.offset + np.sum(X_arr * X_arr, axis=1)) ** self.degree

    def get_params(self) -> Dict[str, Any]:
        return {
            "kernel_type": self.name,
            "degree": self.degree,
            "offset": self.offset,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_type": self.name,
            "name": self.name,
            "family": "polynomial",
            "formula": "(offset + x dot y)^degree",
            "parameter_convention":
                "degree is a positive integer; "
                "offset is added before exponentiation",
            "parameters": {
                "degree": self.degree,
                "offset": self.offset,
            },
        }


def kernel_from_config(kernel_type: str, **params) -> Kernel:
    """Construct a kernel from a concise config dictionary."""
    key = str(kernel_type).strip().lower()
    if key == "gaussian":
        return GaussianKernel(**params)
    if key == "matern32":
        return Matern32Kernel(**params)
    if key == "matern52":
        return Matern52Kernel(**params)
    if key == "polynomial":
        return PolynomialKernel(**params)
    raise ValueError(
        f"Unsupported kernel_type '{kernel_type}'. "
        "Supported kernel types: gaussian, matern32, "
        "matern52, polynomial."
    )
