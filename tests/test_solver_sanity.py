import numpy as np

from vkoga2.data import build_training_data
from vkoga2.kernels import GaussianKernel, Matern32Kernel
from vkoga2.point_selection import point_selection_rule_from_config
from vkoga2.solver import (
    VKOGARebuilt,
    GreedySolverState,
)


def _solver(kernel, max_centers, rule="f_greedy"):
    return VKOGARebuilt(
        kernel=kernel,
        point_selection_rule=point_selection_rule_from_config(rule),
        max_centers=max_centers,
        power_tol=0.0,
    )


def test_1d_full_center_interpolation_sanity():
    X = np.linspace(-1.0, 1.0, 7)[:, None]
    y = np.sin(np.pi * X[:, 0])
    solver = _solver(Matern32Kernel(gamma=2.0), max_centers=len(X))

    solver.fit(X, y)
    pred = solver.predict(X)

    np.testing.assert_allclose(pred, y, atol=1e-10)


def test_2d_circle_sdf_toy_full_center_interpolation_sanity():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [0.5, 0.5],
        ]
    )
    radius = 0.75
    y = np.linalg.norm(X, axis=1) - radius
    solver = _solver(GaussianKernel(gamma=1.5), max_centers=len(X))

    solver.fit(X, y)
    pred = solver.predict(X)

    np.testing.assert_allclose(pred, y, atol=1e-10)
    assert solver.n_selected_ == len(X)


def test_full_center_vkoga2_matches_direct_kernel_solve_on_candidates():
    X = np.array([[-1.0], [-0.25], [0.5], [1.25]])
    y = np.array([0.5, -1.0, 2.0, -0.25])
    kernel = GaussianKernel(gamma=0.7)
    solver = _solver(kernel, max_centers=len(X))

    solver.fit(X, y)
    direct_coefficients = np.linalg.solve(kernel.eval(X, X), y)
    direct_pred = kernel.eval(X, X) @ direct_coefficients

    np.testing.assert_allclose(solver.predict(X), direct_pred, atol=1e-10)
    np.testing.assert_allclose(direct_pred, y, atol=1e-10)


def test_selected_centers_unique_and_selected_residuals_decay_in_manual_loop():
    X = np.linspace(-1.0, 1.0, 6)[:, None]
    y = np.cos(np.pi * X[:, 0])
    data = build_training_data(X, y)
    kernel = GaussianKernel(gamma=1.2)
    selection_rule = point_selection_rule_from_config("f_greedy")
    state = GreedySolverState.create(data, kernel, max_centers=data.n_training_points)
    solver = VKOGARebuilt(
        kernel=kernel,
        point_selection_rule=selection_rule,
        max_centers=data.n_training_points,
    )
    selected_residual_after_update = []

    for _ in range(data.n_training_points):
        selected_mask = np.zeros(data.n_training_points, dtype=bool)
        selected_mask[state.selected_candidate_indices[:state.n_selected]] = True
        selection = selection_rule.select(
            np.abs(state.residual),
            np.sqrt(np.maximum(state.power_candidates_sq, 0.0)),
            selected_mask,
        )
        _idx, _norm, _bv = solver._compute_new_basis_values(data, state, selection.candidate_index)
        state.update(_idx, _norm, _bv)
        selected_residual_after_update.append(
            abs(float(state.residual[selection.candidate_index]))
        )

    assert len(np.unique(state.selected_candidate_indices[:state.n_selected])) == state.n_selected
    assert max(selected_residual_after_update) < 1e-10
    assert np.linalg.norm(state.residual) < 1e-10
