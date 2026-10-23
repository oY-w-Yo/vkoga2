import numpy as np
import pytest

from vkoga2.data import build_training_data
from vkoga2.kernels import GaussianKernel
from vkoga2.solver import GreedySolverState, VKOGARebuilt
from vkoga2.point_selection import point_selection_rule_from_config


def _solver(kernel, max_centers, power_tol=1e-14):
    return VKOGARebuilt(
        kernel=kernel,
        point_selection_rule=point_selection_rule_from_config("f_greedy"),
        max_centers=max_centers,
        power_tol=power_tol,
    )


def _simple_data():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, -2.0, 3.0])
    return build_training_data(X, y)


def test_training_data_uses_single_candidate_residual_domain():
    data = _simple_data()
    np.testing.assert_allclose(
        data.X_candidates,
        data.X_candidates,
    )
    np.testing.assert_allclose(
        data.y_candidates,
        data.y_candidates,
    )
    assert data.n_training_points == 3


def test_initialize_state_preallocates_full_capacity():
    data = build_training_data(
        np.array([[0.0], [1.0], [2.0]]),
        np.array([1.0, -2.0, 3.0]),
    )
    kernel = GaussianKernel(gamma=1.0)

    workspace = GreedySolverState.create(data, kernel, max_centers=4)

    assert isinstance(workspace, GreedySolverState)
    assert workspace.n_selected == 0
    assert len(workspace.residual) == data.n_training_points
    assert workspace.basis_values.shape == (data.n_training_points, 4)
    assert workspace.basis_values.flags.f_contiguous
    assert workspace.basis_values.flags.f_contiguous
    assert workspace.selected_candidate_indices.shape == (4,)
    assert workspace.newton_coefficients.shape == (4,)
    assert workspace.newton_normalizers.shape == (4,)
    np.testing.assert_allclose(workspace.initial_power_candidates_sq, kernel.diagonal(data.X_candidates))
    np.testing.assert_allclose(workspace.power_candidates_sq, workspace.initial_power_candidates_sq)
    np.testing.assert_allclose(workspace.initial_residual, data.y_candidates)
    np.testing.assert_allclose(workspace.residual, data.y_candidates)


def test_new_basis_eval_values_match_candidate_values():
    data = _simple_data()
    kernel = GaussianKernel(gamma=0.8)
    state = GreedySolverState.create(data, kernel, max_centers=data.n_training_points)

    for idx in [0, 2]:
        _, normalizer, basis_values = _solver(kernel, max_centers=state.selected_candidate_indices.shape[0])._compute_new_basis_values(data, state,idx)
        assert len(basis_values) == data.n_training_points
        state.update(idx, normalizer, basis_values)


def test_first_newton_basis_step_has_expected_values():
    data = _simple_data()
    kernel = GaussianKernel(gamma=0.5)
    state = GreedySolverState.create(data, kernel, max_centers=data.n_training_points)
    selected_index = 1

    _, normalizer, basis_values = _solver(kernel, max_centers=state.selected_candidate_indices.shape[0])._compute_new_basis_values(data, state,selected_index)

    x_selected = data.X_candidates[selected_index : selected_index + 1]
    expected = kernel.eval(data.X_candidates, x_selected)[:, 0] / np.sqrt(
        kernel.diagonal(x_selected)[0]
    )
    np.testing.assert_allclose(basis_values, expected)
    np.testing.assert_allclose(basis_values, expected)
    assert basis_values.shape == (3,)
    assert basis_values.shape == (3,)
    assert np.isclose(basis_values[selected_index], 1.0)
    assert np.isclose(normalizer, 1.0)


def test_second_newton_basis_step_orthogonalizes_against_first_selected_center():
    data = build_training_data(
        np.array([[0.0], [10.0], [20.0]]),
        np.array([1.0, -2.0, 3.0]),
    )
    kernel = GaussianKernel(gamma=1.0)

    state = GreedySolverState.create(data, kernel, max_centers=data.n_training_points)
    idx0, normalizer0, basis_values0 = _solver(kernel, max_centers=state.selected_candidate_indices.shape[0])._compute_new_basis_values(data, state,0)
    state.update(idx0, normalizer0, basis_values0)

    _, _, basis_values1 = _solver(kernel, max_centers=state.selected_candidate_indices.shape[0])._compute_new_basis_values(data, state,1)

    assert abs(basis_values1[0]) < 1e-12
    assert np.isclose(basis_values1[1], 1.0)


def test_state_update_appends_selected_index_coefficient_and_updates_residual_power():
    data = _simple_data()
    kernel = GaussianKernel(gamma=1.0)
    state = GreedySolverState.create(data, kernel, max_centers=data.n_training_points)
    selected_index = 0

    _, normalizer, basis_values = _solver(kernel, max_centers=state.selected_candidate_indices.shape[0])._compute_new_basis_values(data, state,selected_index)
    state.update(_, normalizer, basis_values)

    assert state.n_selected == 1
    assert state.selected_candidate_indices[0] == selected_index
    assert np.isclose(state.newton_normalizers[0], normalizer)
    assert np.isclose(state.newton_coefficients[0], data.y_candidates[selected_index])
    assert abs(state.residual[selected_index]) < 1e-12
    assert abs(state.power_candidates_sq[selected_index]) < 1e-12


def test_workspace_update_matches_state_update_path():
    data = _simple_data()
    kernel = GaussianKernel(gamma=0.8)
    state = GreedySolverState.create(data, kernel, max_centers=data.n_training_points)

    for idx in [0, 2, 1]:
        _idx, _norm, _bv = _solver(kernel, max_centers=state.selected_candidate_indices.shape[0])._compute_new_basis_values(data, state,idx)
        state.update(_idx, _norm, _bv)

    assert state.n_selected == 3
    assert len(np.unique(state.selected_candidate_indices[:3])) == 3
    assert state.newton_coefficients[2] != 0.0
    assert state.newton_normalizers[2] != 0.0
    assert np.isclose(np.linalg.norm(state.residual), 0.0)


def test_state_update_rejects_over_capacity():
    data = _simple_data()
    kernel = GaussianKernel(gamma=1.0)
    state = GreedySolverState.create(data, kernel, max_centers=1)

    idx0, norm0, bv0 = _solver(kernel, max_centers=state.selected_candidate_indices.shape[0])._compute_new_basis_values(data, state,0)
    state.update(idx0, norm0, bv0)
    idx1, norm1, bv1 = _solver(kernel, max_centers=state.selected_candidate_indices.shape[0])._compute_new_basis_values(data, state,1)

    with pytest.raises(ValueError, match="already full"):
        state.update(idx1, norm1, bv1)


def test_compute_new_basis_values_raises_when_selected_power_below_tol():
    data = _simple_data()
    kernel = GaussianKernel(gamma=1.0)
    state = GreedySolverState.create(data, kernel, max_centers=data.n_training_points)
    state.power_candidates_sq[1] = 1e-40

    with pytest.raises(ValueError, match="too small"):
        _solver(kernel, max_centers=state.selected_candidate_indices.shape[0], power_tol=1e-14)._compute_new_basis_values(data, state,1)


def test_slice_state_prefix_reconstructs_residual_and_power_from_initial_quantities():
    data = _simple_data()
    kernel = GaussianKernel(gamma=0.8)
    state = GreedySolverState.create(data, kernel, max_centers=data.n_training_points)
    for idx in [0, 2, 1]:
        _, normalizer, basis_values = _solver(kernel, max_centers=state.selected_candidate_indices.shape[0])._compute_new_basis_values(data, state,idx)
        state.update(idx, normalizer, basis_values)

    prefix = state.slice_prefix(2)

    assert prefix.selected_candidate_indices.shape == (2,)
    assert prefix.newton_coefficients.shape == (2,)
    assert prefix.newton_normalizers.shape == (2,)
    assert prefix.basis_values.shape == (data.n_training_points, 2)

    expected_residual = (
        state.initial_residual
        - state.basis_values[:, :2] @ state.newton_coefficients[:2]
    )
    expected_power = (
        state.initial_power_candidates_sq
        - np.sum(state.basis_values[:, :2] ** 2, axis=1)
    )
    expected_power = np.maximum(expected_power, 0.0)

    np.testing.assert_allclose(prefix.residual, expected_residual)
    np.testing.assert_allclose(prefix.power_candidates_sq, expected_power)


def test_slice_state_prefix_rejects_invalid_prefix_length():
    data = _simple_data()
    kernel = GaussianKernel(gamma=1.0)
    state = GreedySolverState.create(data, kernel, max_centers=data.n_training_points)

    with pytest.raises(ValueError, match="n_selected"):
        state.slice_prefix(1)
