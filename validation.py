"""Shared input validation helpers for the rebuilt VKOGA core.

Naming convention:

- ``as_*`` — convert input to a validated array of a specific type/shape.
  These accept raw (possibly list) data and return a typed ``np.ndarray``.
- ``validate_*`` — check that a scalar value satisfies its constraints.
  These accept a (possibly raw) value and return a validated Python scalar.

Internal helpers (``_``-prefixed) are used within this package only.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# Array converters (as_*)
# =============================================================================


def as_valid_point_matrix(X, *, name: str) -> np.ndarray:
    """Return ``X`` as a floating-point matrix with shape ``(n, d)``."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (n, d); got array with shape {arr.shape}.")
    return arr


def _as_1d_finite_array(values, *, name: str) -> np.ndarray:
    """Return ``values`` as a one-dimensional finite float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_1d_bool_array(values, *, name: str) -> np.ndarray:
    """Return ``values`` as a one-dimensional bool array."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array; got shape {arr.shape}.")
    if arr.dtype != np.bool_:
        raise ValueError(f"{name} must be a bool array.")
    return arr


# =============================================================================
# Scalar validators (validate_*)
# =============================================================================


def validate_nonnegative_float(value: float, *, name: str) -> float:
    """Return ``value`` as a finite float greater than or equal to zero."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative; got {value!r}.")
    return value


def validate_index_range(index: int, *, lo: int, hi: int, name: str) -> int:
    """Validate and return an integer index in ``[lo, hi)``."""
    index = int(index)
    if index < lo or index >= hi:
        raise IndexError(
            f"{name} must be in [{lo}, {hi}); got {index}."
        )
    return index


def validate_positive_int(value: int, *, name: str) -> int:
    """Return ``value`` as an integer greater than or equal to one."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1; got {value!r}.")
    return value
