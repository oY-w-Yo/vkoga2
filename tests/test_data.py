import json

import numpy as np
import pytest

from vkoga2.data import (
    PDE_FUNCTIONAL_DIRICHLET,
    PDE_FUNCTIONAL_INTERIOR,
    PDETrainingData,
    TrainingData,
    build_pde_training_data,
    build_training_data,
)


def test_build_training_data_default_case():
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    y = np.array([1.0, -1.0])

    data = build_training_data(X, y)

    assert isinstance(data, TrainingData)
    np.testing.assert_allclose(data.X_candidates, X)
    np.testing.assert_allclose(data.y_candidates, y)
    assert data.n_training_points == 2
    assert data.dim == 2


def test_greedy_candidates_data_to_dict_default_is_json_safe():
    data = build_training_data(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))

    summary = data.to_dict()

    json.dumps(summary)
    assert summary == {
        "n_training_points": 2,
        "dim": 1,
    }


def test_greedy_candidates_data_is_frozen():
    data = build_training_data(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))

    with pytest.raises(Exception):
        data.n_training_points = 10


def test_invalid_X_rejected():
    with pytest.raises(ValueError, match=r"shape \(n, d\)"):
        build_training_data(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


def test_nonfinite_X_rejected():
    with pytest.raises(ValueError, match="finite"):
        build_training_data(np.array([[0.0], [np.inf]]), np.array([0.0, 1.0]))


def test_invalid_y_rejected():
    with pytest.raises(ValueError, match=r"one-dimensional array"):
        build_training_data(np.array([[0.0], [1.0]]), np.array([[0.0], [1.0]]))


def test_nonfinite_y_rejected():
    with pytest.raises(ValueError, match="finite"):
        build_training_data(np.array([[0.0], [1.0]]), np.array([0.0, np.nan]))


def test_y_length_mismatch_rejected():
    with pytest.raises(ValueError, match="len\\(y\\) must match len\\(X\\)"):
        build_training_data(np.array([[0.0], [1.0]]), np.array([0.0]))


def test_empty_original_data_rejected():
    with pytest.raises(ValueError, match="at least one data point"):
        build_training_data(np.empty((0, 2)), np.empty((0,)))


def test_build_pde_training_data_interior_and_dirichlet():
    X_int = np.array([[0.25], [0.5]])
    f_vals = np.array([1.0, 2.0])
    X_dir = np.array([[0.0], [1.0]])
    g_dir = np.array([0.0, 0.0])

    data = build_pde_training_data(X_int, f_vals, X_dir, g_dir)

    assert isinstance(data, PDETrainingData)
    assert data.n_int == 2
    assert data.n_dir == 2
    assert data.n_trial == 4
    assert data.dim == 1
    np.testing.assert_allclose(data.X_all, np.vstack((X_int, X_dir)))
    np.testing.assert_allclose(data.rhs_all, np.array([1.0, 2.0, 0.0, 0.0]))
    np.testing.assert_array_equal(
        data.functional_types,
        np.array([
            PDE_FUNCTIONAL_INTERIOR,
            PDE_FUNCTIONAL_INTERIOR,
            PDE_FUNCTIONAL_DIRICHLET,
            PDE_FUNCTIONAL_DIRICHLET,
        ]),
    )


def test_pde_training_data_to_dict_is_json_safe():
    data = build_pde_training_data(
        np.array([[0.5]]),
        np.array([2.0]),
        np.array([[0.0], [1.0]]),
        np.array([0.0, 0.0]),
    )

    summary = data.to_dict()

    json.dumps(summary)
    assert summary["n_trial"] == 3
    assert summary["functional_type_counts"] == {
        "interior": 1,
        "dirichlet": 2,
    }


def test_pde_training_data_allows_no_dirichlet_points():
    data = build_pde_training_data(np.array([[0.5]]), np.array([2.0]))

    assert data.n_trial == 1
    assert data.n_dir == 0
    np.testing.assert_allclose(data.X_all, np.array([[0.5]]))
    np.testing.assert_allclose(data.rhs_all, np.array([2.0]))


def test_pde_training_data_rejects_rhs_length_mismatch():
    with pytest.raises(ValueError, match="len\\(f_vals\\)"):
        build_pde_training_data(np.array([[0.5], [0.75]]), np.array([1.0]))

    with pytest.raises(ValueError, match="len\\(g_dir\\)"):
        build_pde_training_data(
            np.array([[0.5]]),
            np.array([1.0]),
            np.array([[0.0], [1.0]]),
            np.array([0.0]),
        )


def test_pde_training_data_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="same feature dimension"):
        build_pde_training_data(
            np.array([[0.5, 0.0]]),
            np.array([1.0]),
            np.array([[0.0]]),
            np.array([0.0]),
        )
