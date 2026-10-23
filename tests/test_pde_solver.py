import json

import numpy as np
import pytest

from vkoga2.data import build_pde_training_data
from vkoga2.kernels import GaussianKernel
from vkoga2.model_selection import KeepLastModelSelectionRule
from vkoga2.pde_composer import SecondOrderEllipticComposer
from vkoga2.pde_solver import (
    PDEGreedySolver,
    PDEGreedyState,
    PDE_STOP_MAX_CENTERS,
)
from vkoga2.point_selection import point_selection_rule_from_config


def _diffusion(X):
    return np.ones(len(X), dtype=float)


def _reaction(X):
    return np.zeros(len(X), dtype=float)


def _manufactured_data():
    X_int = np.array([[0.25], [0.5], [0.75]])
    X_dir = np.array([[0.0], [1.0]])
    u_dir = np.sin(np.pi * X_dir[:, 0])
    f_vals = np.pi**2 * np.sin(np.pi * X_int[:, 0])
    return build_pde_training_data(X_int, f_vals, X_dir, u_dir)


def _composer(gamma=2.0):
    return SecondOrderEllipticComposer(
        GaussianKernel(gamma=gamma),
        diffusion=_diffusion,
        reaction=_reaction,
    )


def _solver(data, gamma=2.0, **kwargs):
    return PDEGreedySolver(
        composer=_composer(gamma=gamma),
        max_centers=data.n_trial,
        point_selection_rule=point_selection_rule_from_config("f_greedy"),
        model_selection_rule=KeepLastModelSelectionRule(),
        **kwargs,
    )


def test_pde_state_create_preallocates_phase1_arrays():
    data = _manufactured_data()
    state = PDEGreedyState.create(data, _composer(), max_centers=data.n_trial)

    assert state.n_selected == 0
    assert state.basis_values.shape == (data.n_trial, data.n_trial)
    assert state.basis_values.flags.f_contiguous
    assert state.selected_indices.shape == (data.n_trial,)
    assert state.selected_types.shape == (data.n_trial,)
    np.testing.assert_allclose(state.initial_residual, data.rhs_all)
    assert np.all(state.initial_power_candidates_sq > 0.0)


def test_pde_greedy_full_solve_satisfies_training_functionals():
    data = _manufactured_data()
    solver = _solver(data, gamma=2.0)

    solver.fit(data)

    np.testing.assert_allclose(
        solver.predict_operator(data.X_int),
        data.f_vals,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        solver.predict(data.X_dir),
        data.g_dir,
        rtol=1e-8,
        atol=1e-8,
    )
    assert solver.is_fitted_
    assert solver.n_selected_ == data.n_trial
    assert solver.stop_reason_ == PDE_STOP_MAX_CENTERS
    assert solver.selected_centers_.shape == data.X_all.shape
    assert solver.selected_center_types_.shape == (data.n_trial,)


def test_pde_greedy_prediction_matches_composer_matrix_times_coefficients():
    data = _manufactured_data()
    solver = _solver(data, gamma=1.5).fit(data)
    X_query = np.linspace(0.1, 0.9, 7)[:, None]

    matrix = solver.composer.eval_predict(
        solver.selected_center_types_,
        X_query,
        solver.selected_centers_,
    )
    expected = matrix @ solver.selected_coefficients_

    np.testing.assert_allclose(solver.predict(X_query), expected, rtol=1e-12, atol=1e-12)


def test_pde_greedy_result_summary_is_json_safe():
    data = _manufactured_data()
    solver = _solver(data, gamma=2.0).fit(data)

    summary = solver.result_summary()

    json.dumps(summary, allow_nan=False)
    assert summary["solver_type"] == "PDEGreedySolver"
    assert summary["model"]["n_selected_centers"] == data.n_trial


def test_pde_greedy_predict_before_fit_raises():
    data = _manufactured_data()
    solver = _solver(data)

    with pytest.raises(RuntimeError, match="before fit"):
        solver.predict(np.array([[0.5]]))

    with pytest.raises(RuntimeError, match="before fit"):
        solver.predict_operator(np.array([[0.5]]))
