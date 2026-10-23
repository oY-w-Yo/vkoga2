"""Operator/kernel composition for PDE greedy solves."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .data import PDE_FUNCTIONAL_DIRICHLET, PDE_FUNCTIONAL_INTERIOR
from .kernels import Kernel
from .validation import (
    _as_1d_integer_array,
    _as_valid_finite_point_matrix,
)


CoefficientFunction = Callable[[np.ndarray], np.ndarray]


class OperatorComposer(ABC):
    """Compose kernel differential primitives with PDE operator coefficients."""

    @property
    @abstractmethod
    def n_types(self) -> int:
        """Number of supported functional types."""
        ...

    @abstractmethod
    def power_diagonal(self, types: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Return the initial power-function diagonal ``A[i, i]``."""
        ...

    @abstractmethod
    def eval_candidate(
        self,
        center_type: int,
        trial_X: np.ndarray,
        trial_types: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """Return the generalized kernel column for selected center ``z``."""
        ...

    @abstractmethod
    def eval_predict(
        self,
        center_types: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """Return the mixed Riesz-representer matrix for solution prediction."""
        ...

    @abstractmethod
    def eval_operator_predict(
        self,
        center_types: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """Return ``L_x`` applied to the mixed Riesz-representer matrix."""
        ...


def _zero_coefficient(X: np.ndarray) -> np.ndarray:
    return np.zeros(len(X), dtype=float)


def _validate_center_type(center_type: int) -> int:
    center_type = int(center_type)
    if center_type not in (PDE_FUNCTIONAL_INTERIOR, PDE_FUNCTIONAL_DIRICHLET):
        raise ValueError(f"Unsupported center_type {center_type}; expected 0 or 1.")
    return center_type


def _validate_functional_types(types, *, n_expected: int, name: str) -> np.ndarray:
    arr = _as_1d_integer_array(types, name=name)
    if arr.shape[0] != n_expected:
        raise ValueError(
            f"{name} length must match X rows; got {arr.shape[0]} and {n_expected}."
        )
    invalid = ~np.isin(arr, [PDE_FUNCTIONAL_INTERIOR, PDE_FUNCTIONAL_DIRICHLET])
    if np.any(invalid):
        bad = sorted({int(v) for v in arr[invalid]})
        raise ValueError(f"{name} contains unsupported functional types: {bad}.")
    return arr


@dataclass(frozen=True)
class SecondOrderEllipticComposer(OperatorComposer):
    """Composer for ``L = -nu(x) Δ + sigma(x) I``.

    Phase 1 supports two functional types:

    - ``0``: interior functional ``L[u](x)``
    - ``1``: Dirichlet point-evaluation functional ``u(x)``
    """

    kernel: Kernel
    diffusion: CoefficientFunction
    reaction: CoefficientFunction = _zero_coefficient

    def __post_init__(self) -> None:
        if not callable(self.diffusion):
            raise TypeError("diffusion must be callable.")
        if not callable(self.reaction):
            raise TypeError("reaction must be callable.")

    @property
    def n_types(self) -> int:
        return 2

    def power_diagonal(self, types: np.ndarray, X: np.ndarray) -> np.ndarray:
        X_arr = _as_valid_finite_point_matrix(X, name="X")
        types_arr = _validate_functional_types(types, n_expected=len(X_arr), name="types")

        result = np.empty(len(X_arr), dtype=float)
        mask_int = types_arr == PDE_FUNCTIONAL_INTERIOR
        if np.any(mask_int):
            X_int = X_arr[mask_int]
            result[mask_int] = np.diag(self._L_double(X_int, X_int))

        mask_dir = types_arr == PDE_FUNCTIONAL_DIRICHLET
        if np.any(mask_dir):
            result[mask_dir] = self.kernel.diagonal(X_arr[mask_dir])
        return result

    def eval_candidate(
        self,
        center_type: int,
        trial_X: np.ndarray,
        trial_types: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        center_type = _validate_center_type(center_type)
        X_arr = _as_valid_finite_point_matrix(trial_X, name="trial_X")
        types_arr = _validate_functional_types(
            trial_types,
            n_expected=len(X_arr),
            name="trial_types",
        )
        z_arr = _as_valid_finite_point_matrix(z, name="z")
        if z_arr.shape[0] != 1:
            raise ValueError(f"z must contain exactly one center; got shape {z_arr.shape}.")
        if X_arr.shape[1] != z_arr.shape[1]:
            raise ValueError(
                "trial_X and z must have the same feature dimension; "
                f"got {X_arr.shape} and {z_arr.shape}."
            )

        result = np.empty(len(X_arr), dtype=float)
        mask_int = types_arr == PDE_FUNCTIONAL_INTERIOR
        if np.any(mask_int):
            X_int = X_arr[mask_int]
            if center_type == PDE_FUNCTIONAL_INTERIOR:
                result[mask_int] = self._L_double(X_int, z_arr)[:, 0]
            else:
                result[mask_int] = self._L_first(X_int, z_arr)

        mask_dir = types_arr == PDE_FUNCTIONAL_DIRICHLET
        if np.any(mask_dir):
            X_dir = X_arr[mask_dir]
            if center_type == PDE_FUNCTIONAL_INTERIOR:
                result[mask_dir] = self._L_second(X_dir, z_arr)
            else:
                result[mask_dir] = self.kernel.eval(X_dir, z_arr)[:, 0]
        return result

    def eval_predict(
        self,
        center_types: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        return self._eval_mixed(center_types, X, Z, apply_operator=False)

    def eval_operator_predict(
        self,
        center_types: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        return self._eval_mixed(center_types, X, Z, apply_operator=True)

    def _eval_mixed(
        self,
        center_types,
        X,
        Z,
        *,
        apply_operator: bool,
    ) -> np.ndarray:
        X_arr = _as_valid_finite_point_matrix(X, name="X")
        Z_arr = _as_valid_finite_point_matrix(Z, name="Z")
        if X_arr.shape[1] != Z_arr.shape[1]:
            raise ValueError(
                "X and Z must have the same feature dimension; "
                f"got {X_arr.shape} and {Z_arr.shape}."
            )
        types_arr = _validate_functional_types(
            center_types,
            n_expected=len(Z_arr),
            name="center_types",
        )

        out = np.empty((len(X_arr), len(Z_arr)), dtype=float)
        for j, center_type in enumerate(types_arr):
            z = Z_arr[j:j + 1]
            if apply_operator:
                if center_type == PDE_FUNCTIONAL_INTERIOR:
                    out[:, j] = self._L_double(X_arr, z)[:, 0]
                else:
                    out[:, j] = self._L_first(X_arr, z)
            else:
                if center_type == PDE_FUNCTIONAL_INTERIOR:
                    out[:, j] = self._L_second(X_arr, z)
                else:
                    out[:, j] = self.kernel.eval(X_arr, z)[:, 0]
        return out

    def _coefficient_values(self, fn: CoefficientFunction, X: np.ndarray, *, name: str) -> np.ndarray:
        values = fn(X)
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = np.full(len(X), float(arr), dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name}(X) must return a scalar or one-dimensional array.")
        if arr.shape[0] != len(X):
            raise ValueError(
                f"{name}(X) length must match X rows; got {arr.shape[0]} and {len(X)}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}(X) must contain only finite values.")
        return arr

    def _diffusion(self, X: np.ndarray) -> np.ndarray:
        return self._coefficient_values(self.diffusion, X, name="diffusion")

    def _reaction(self, X: np.ndarray) -> np.ndarray:
        return self._coefficient_values(self.reaction, X, name="reaction")

    def _L_second(self, X: np.ndarray, z: np.ndarray) -> np.ndarray:
        nu_z = self._diffusion(z)[None, :]
        sigma_z = self._reaction(z)[None, :]
        return (
            -nu_z * self.kernel.laplacian_wrt_second(X, z)
            + sigma_z * self.kernel.eval(X, z)
        )[:, 0]

    def _L_first(self, X: np.ndarray, z: np.ndarray) -> np.ndarray:
        nu_X = self._diffusion(X)[:, None]
        sigma_X = self._reaction(X)[:, None]
        return (
            -nu_X * self.kernel.laplacian_wrt_first(X, z)
            + sigma_X * self.kernel.eval(X, z)
        )[:, 0]

    def _L_double(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        nu_X = self._diffusion(X)[:, None]
        sigma_X = self._reaction(X)[:, None]
        nu_Y = self._diffusion(Y)[None, :]
        sigma_Y = self._reaction(Y)[None, :]
        return (
            nu_X * nu_Y * self.kernel.bilaplacian(X, Y)
            - nu_X * sigma_Y * self.kernel.laplacian_wrt_first(X, Y)
            - sigma_X * nu_Y * self.kernel.laplacian_wrt_second(X, Y)
            + sigma_X * sigma_Y * self.kernel.eval(X, Y)
        )
