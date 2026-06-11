"""Distance helpers for the rebuilt VKOGA core."""

from __future__ import annotations

import numpy as np

from .validation import as_valid_point_matrix


def pairwise_euclidean_distance(X, Y) -> np.ndarray:
    """Compute pairwise Euclidean distances between rows of ``X`` and ``Y``.

    Parameters
    ----------
    X, Y:
        Two-dimensional arrays with shapes ``(n, d)`` and ``(m, d)``.

    Returns
    -------
    numpy.ndarray
        Distance matrix with shape ``(n, m)``.
    """
    X_arr = as_valid_point_matrix(X, name="X")
    Y_arr = as_valid_point_matrix(Y, name="Y")
    if X_arr.shape[1] != Y_arr.shape[1]:
        raise ValueError(
            "X and Y must have the same feature dimension; "
            f"got X.shape={X_arr.shape} and Y.shape={Y_arr.shape}."
        )
    # Compute squared distances by the Gram identity
    #
    #   ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 x_i · y_j.
    #
    # This keeps the main work in the BLAS-backed matrix product X @ Y.T and
    # avoids materializing the naive broadcast tensor X[:, None, :] - Y[None, :, :],
    # which would have shape (n, m, d).
    x_sq = np.sum(X_arr * X_arr, axis=1, keepdims=True)   # (n, 1)
    y_sq = np.sum(Y_arr * Y_arr, axis=1)                   # (m,)
    # Build squared-distance matrix with one alloc then in-place ops.
    #   ||x_i - y_j||² = ||x_i||² + ||y_j||² - 2 x_i·y_j
    # The naive expression `x_sq + y_sq[None, :] - 2.0 * (X @ Y.T)` creates up to
    # three (n, m) temporaries.  Chaining in-place keeps the peak at one.
    dist_sq = X_arr @ Y_arr.T                                # (n, m) — single large alloc
    dist_sq *= -2.0                                          # in-place
    dist_sq += x_sq                                          # in-place broadcast
    dist_sq += y_sq                                          # in-place broadcast
    # Roundoff can make theoretically zero squared distances slightly negative.
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return np.sqrt(dist_sq, out=dist_sq)


def pairwise_distance(X, Y, metric: str = "euclidean") -> np.ndarray:
    """Compute a pairwise distance matrix.

    The first rebuilt layer intentionally supports only Euclidean distance.
    Additional metrics should be added explicitly when they are needed.
    """
    metric_key = str(metric).strip().lower()
    if metric_key == "euclidean":
        return pairwise_euclidean_distance(X, Y)
    raise ValueError(
        f"Unsupported distance metric '{metric}'. "
        "Currently supported metrics: euclidean."
    )
