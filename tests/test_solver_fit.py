import json
import pickle

import numpy as np
import pytest

from vkoga2.kernels import GaussianKernel
from vkoga2.model_selection import BestScoreModelSelectionRule
from vkoga2.point_selection import point_selection_rule_from_config
from vkoga2.solver import (
    VKOGARebuilt,
    STOP_ALL_CANDIDATES,
    STOP_MAX_CENTERS,
    STOP_RESIDUAL_TOO_SMALL,
)


def _make_solver(max_centers, *, gamma=1.0, model_selection_rule=None, **extra_kwargs):
    kwargs = {}
    if model_selection_rule is not None:
        kwargs["model_selection_rule"] = model_selection_rule
    kwargs.update(extra_kwargs)
    return VKOGARebuilt(
        kernel=GaussianKernel(gamma=gamma),
        point_selection_rule=point_selection_rule_from_config("f_greedy"),
        max_centers=max_centers,
        **kwargs,
    )


def test_basic_fit_with_one_center_predicts_query_shape():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, -2.0, 3.0])
    solver = _make_solver(max_centers=1)

    solver.fit(X, y)
    pred = solver.predict(np.array([[0.0], [0.5], [2.0], [3.0]]))

    assert solver.is_fitted_ is True
    assert solver.n_selected_ == 1
    assert solver.n_selected_full_ == 1
    assert solver.stop_reason_ == STOP_MAX_CENTERS
    assert not hasattr(solver, "full_state_")
    assert solver.retained_state_ is not None
    assert solver.retained_iteration_ == solver.n_selected_
    assert solver.history_ is not None
    assert len(solver.history_) == solver.n_selected_
    first_record = solver.history_.records[0]
    assert first_record.n_selected == 1
    assert first_record.selected_candidate_index == int(solver.retained_state_.selected_candidate_indices[0])
    assert pred.shape == (4,)


def test_default_verbose_false_produces_no_stdout(capsys):
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, -2.0, 3.0])
    solver = _make_solver(max_centers=2)

    solver.fit(X, y)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_verbose_true_reports_started_progress_and_finished(capsys):
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, -2.0, 3.0])
    solver = _make_solver(max_centers=2, verbose=True, report_every=1)

    solver.fit(X, y)

    out = capsys.readouterr().out
    assert "[VKOGARebuilt] started" in out
    assert "n_training_points=3" in out
    assert "max_centers=2" in out
    assert "point_selection=f_greedy" in out
    assert "model_selection=keep_best_rmax" in out
    assert "[VKOGARebuilt] progress iteration=1/2" in out
    assert "[VKOGARebuilt] progress iteration=2/2" in out
    assert "selected_candidate_index=" in out
    assert "residual_rmse=" in out
    assert "residual_max=" in out
    assert "power_max=" in out
    assert "[VKOGARebuilt] finished" in out
    assert "stop_reason=max_centers" in out
    assert "n_selected_full=2" in out
    assert "retained_iteration=2" in out
    assert "retained_centers=2" in out
    assert "retained_residual_rmse=" in out


@pytest.mark.parametrize("report_every", [0, -1, 1.5, True])
def test_report_every_must_be_positive_integer(report_every):
    with pytest.raises(ValueError, match="report_every"):
        _make_solver(max_centers=1, report_every=report_every)


@pytest.mark.parametrize("residual_tol", [-1.0, float("nan")])
def test_residual_tol_must_be_finite_and_nonnegative(residual_tol):
    with pytest.raises(ValueError, match="residual_tol"):
        _make_solver(max_centers=1, residual_tol=residual_tol)


def test_full_interpolation_on_candidates():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, -2.0, 3.0])
    solver = _make_solver(max_centers=len(X), gamma=0.8)

    solver.fit(X, y)
    pred = solver.predict(X)

    np.testing.assert_allclose(pred, y, atol=1e-10)
    assert solver.n_selected_ == len(X)
    assert solver.stop_reason_ == STOP_MAX_CENTERS
    assert solver.retained_iteration_ == solver.n_selected_full_
    assert len(solver.history_) == solver.n_selected_
    residual_rmse = solver.history_.get_field_values("residual_rmse")
    assert residual_rmse.shape == (solver.n_selected_,)
    assert np.all(np.isfinite(residual_rmse))


def test_predict_before_fit_raises_clear_error():
    solver = _make_solver(max_centers=1)

    with pytest.raises(RuntimeError, match="before fit"):
        solver.predict(np.array([[0.0]]))

    with pytest.raises(RuntimeError, match="before fit"):
        solver.result_summary()


def test_state_normalizers_and_evaluate_basis_at_points_match_state_candidates():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, -2.0, 3.0])
    solver = _make_solver(max_centers=2, gamma=0.8)

    solver.fit(X, y)

    assert solver.retained_state_.newton_normalizers.shape[0] == solver.n_selected_
    assert solver.retained_state_.basis_values.shape == (len(X), solver.n_selected_)


def test_kernel_coefficient_prediction_matches_newton_basis_prediction():
    X = np.linspace(-1.0, 1.0, 9)[:, None]
    y = np.sin(np.pi * X[:, 0])
    X_query = np.linspace(-0.9, 0.9, 13)[:, None]
    solver = _make_solver(max_centers=6, gamma=0.8)

    solver.fit(X, y)

    alpha = solver._selected_kernel_coefficients
    selected_centers = solver.selected_centers_
    kernel_pred = solver.kernel.eval(X_query, selected_centers) @ alpha

    np.testing.assert_allclose(solver.predict(X_query), kernel_pred, atol=1e-12)
    np.testing.assert_allclose(solver._selected_kernel_coefficients, alpha, atol=1e-12)
    np.testing.assert_allclose(solver.selected_centers_, selected_centers, atol=1e-12)


def test_pickled_solver_default_mode_keeps_only_prediction_fields():
    X = np.linspace(-1.0, 1.0, 10)[:, None]
    y = np.sin(np.pi * X[:, 0])
    solver = _make_solver(max_centers=6, gamma=0.9)
    solver.fit(X, y)

    restored = pickle.loads(pickle.dumps(solver))
    X_query = np.linspace(-0.8, 0.8, 7)[:, None]

    assert restored.is_fitted_ is True
    assert not hasattr(restored, "retained_state_")
    assert not hasattr(restored, "training_data_")
    assert not hasattr(restored, "history_")
    assert hasattr(restored, "_selected_centers")
    assert hasattr(restored, "_selected_kernel_coefficients")
    assert restored.selected_centers_.shape == (restored.n_selected_, X.shape[1])
    assert restored._selected_kernel_coefficients.shape == (restored.n_selected_,)
    np.testing.assert_allclose(restored.predict(X_query), solver.predict(X_query), atol=1e-12)


def test_pickled_solver_full_mode_preserves_all_state():
    X = np.linspace(-1.0, 1.0, 10)[:, None]
    y = np.sin(np.pi * X[:, 0])
    solver = _make_solver(max_centers=6, gamma=0.9)
    solver.fit(X, y)

    solver._pickle_mode = "full"
    restored = pickle.loads(pickle.dumps(solver))
    X_query = np.linspace(-0.8, 0.8, 7)[:, None]

    assert restored.is_fitted_ is True
    assert hasattr(restored, "retained_state_")
    assert hasattr(restored, "training_data_")
    assert hasattr(restored, "history_")
    assert restored.retained_state_.basis_values.shape[1] == restored.n_selected_
    np.testing.assert_allclose(restored.predict(X_query), solver.predict(X_query), atol=1e-12)
    json.dumps(restored.result_summary(), allow_nan=False)


def test_fit_stops_when_all_candidates_selected_if_budget_exceeds_candidates():
    X = np.array([[0.0], [1.0]])
    y = np.array([1.0, -1.0])
    solver = _make_solver(max_centers=5)

    solver.fit(X, y)

    assert solver.n_selected_ == 2
    assert solver.stop_reason_ == STOP_ALL_CANDIDATES


def test_fit_stops_when_residual_becomes_small_enough():
    X = np.linspace(-1.0, 1.0, 10)[:, None]
    y = np.sin(np.pi * X[:, 0])
    solver = _make_solver(max_centers=10, gamma=0.9, residual_tol=0.25)

    solver.fit(X, y)

    assert solver.stop_reason_ == STOP_RESIDUAL_TOO_SMALL
    assert solver.n_selected_ < 10
    assert len(solver.history_) >= 1
    assert solver.history_.records[-1].residual_rmse <= solver.residual_tol


def test_best_score_model_selection_builds_retained_prefix_state():
    X = np.linspace(-1.0, 1.0, 6)[:, None]
    y = np.cos(np.pi * X[:, 0])
    solver = _make_solver(
        max_centers=4,
        gamma=1.2,
        model_selection_rule=BestScoreModelSelectionRule(score_name="residual_rmse"),
    )

    solver.fit(X, y)

    rmse = solver.history_.get_field_values("residual_rmse")
    expected_iteration = int(np.argmin(rmse) + 1)
    assert solver.retained_iteration_ == expected_iteration
    assert solver.retained_state_.selected_candidate_indices.shape[0] == solver.retained_iteration_
    assert solver.n_selected_ == solver.retained_iteration_
    assert solver.model_selection_rule.score_name == "residual_rmse"


def test_prediction_uses_retained_prefix_not_full_state():
    X = np.linspace(-1.0, 1.0, 6)[:, None]
    y = np.cos(np.pi * X[:, 0])
    solver = _make_solver(
        max_centers=4,
        gamma=1.2,
        model_selection_rule=BestScoreModelSelectionRule(score_name="residual_rmse"),
    )

    solver.fit(X, y)
    assert solver.retained_iteration_ < solver.n_selected_full_

    X_query = np.array([[-0.5], [0.0], [0.5]])
    retained_pred = solver.kernel.eval(X_query, solver.selected_centers_) @ solver._selected_kernel_coefficients

    np.testing.assert_allclose(solver.predict(X_query), retained_pred)
    assert solver.selected_centers_.shape[0] == solver.retained_iteration_
    assert solver._selected_kernel_coefficients.shape[0] == solver.retained_iteration_


def test_result_summary_is_json_safe_and_deduplicated():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1.0, -2.0, 3.0])
    solver = _make_solver(max_centers=2, gamma=0.8, store_cond=True)

    solver.fit(X, y)
    summary = solver.result_summary()

    json.dumps(summary, allow_nan=False)
    assert list(summary.keys()) == [
        "data",
        "solver_type",
        "config",
        "stop_reason",
        "fit_time_sec",
        "model",
        "training_history",
    ]
    assert summary["data"] == {"n_training_points": 3, "dim": 1}
    assert summary["solver_type"] == "VKOGARebuilt"
    assert summary["config"]["max_centers"] == 2
    assert summary["config"]["kernel"]["kernel_type"] == "gaussian"
    assert summary["config"]["point_selection"]["rule_name"] == "f_greedy"
    assert summary["stop_reason"] == solver.stop_reason_
    assert summary["fit_time_sec"] == solver._fit_time_sec

    model = summary["model"]
    assert list(model.keys()) == [
        "n_selected_centers",
        "selected_candidate_indices",
        "kernel_coefficients",
        "quality",
    ]
    assert model["n_selected_centers"] == solver.n_selected_
    assert model["selected_candidate_indices"] == [
        int(index) for index in solver.retained_state_.selected_candidate_indices
    ]
    assert model["kernel_coefficients"]["count"] == solver.n_selected_
    assert "kernel" not in model
    record = solver.history_.records[solver.retained_iteration_ - 1]
    expected_quality = {
        "residual_rmse": record.residual_rmse,
        "residual_max": record.residual_max,
        "power_max": record.power_max,
        "condition_number": record.condition_number,
    }
    assert model["quality"] == expected_quality
    assert summary["training_history"] == solver.history_.records_to_dicts()
    assert "history" not in summary
    assert "n_selected_full_path" not in summary


def _assert_indicator_dict_allclose(actual, expected, *, atol=1e-12):
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=atol)


def test_fit_produces_fortran_workspace():
    rng = np.random.default_rng(4321)
    X = rng.uniform(-1.0, 1.0, size=(36, 2))
    y = np.sin(2.0 * X[:, 0]) - 0.25 * np.cos(4.0 * X[:, 1])
    X_test = rng.uniform(-1.0, 1.0, size=(20, 2))

    model = _make_solver(max_centers=12, gamma=1.1)
    model.fit(X, y)

    assert model.predict(X_test).shape == (20,)
    assert "workspace" not in model.result_summary()


def test_fit_with_default_order_produces_consistent_results():
    rng = np.random.default_rng(9876)
    X = rng.uniform(-1.0, 1.0, size=(28, 2))
    y = np.cos(3.0 * X[:, 0]) + np.sin(2.0 * X[:, 1])
    X_test = rng.uniform(-1.0, 1.0, size=(12, 2))

    model = _make_solver(max_centers=9, gamma=0.9)
    model.fit(X, y)

    assert model.predict(X_test).shape == (12,)
    assert np.all(np.isfinite(model.predict(X_test)))
