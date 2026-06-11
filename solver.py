"""Newton-basis solver pieces for the rebuilt VKOGA core."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from .data import GreedyTrainingData, build_greedy_training_data
from .history import GreedyHistory, GreedyStepRecord
from .kernels import Kernel
from .model_selection import (
    BaseModelSelectionRule,
    BestScoreModelSelectionRule,
)
from .point_selection import BetaGreedyPointSelectionRule
from .validation import (
    as_valid_point_matrix,
    validate_index_range,
    validate_positive_int,
)
# -- JSON-safe summary helpers -----------------------------------------------


def _json_safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else None

# =============================================================================
# State object
# =============================================================================


@dataclass
class GreedySolverState:
    """State of a partial greedy Newton-basis expansion.

    During the fit loop, arrays are preallocated to ``max_centers`` capacity and
    ``n_selected`` tracks the active prefix.
    """

    # Power function values stored as squares (update avoids sqrt each iteration)
    initial_power_candidates_sq: np.ndarray
    power_candidates_sq: np.ndarray
    # Training-domain residual quantities
    initial_residual: np.ndarray
    residual: np.ndarray
    basis_values: np.ndarray
    # Selected-center / Newton expansion quantities S
    selected_candidate_indices: np.ndarray
    newton_coefficients: np.ndarray
    newton_normalizers: np.ndarray
    n_selected: int = 0

    # ----- Construction -----

    @classmethod
    def create(
        cls,
        data: GreedyTrainingData,
        kernel: Kernel,
        max_centers: int,
    ) -> "GreedySolverState":
        """Initialize a preallocated greedy solver state for fitting.

        Basis values use F-order (column-major) layout.  Benchmarks show it is
        consistently faster than C-order for the Newton-basis write-column /
        prefix-projection access pattern.
        """
        max_centers = validate_positive_int(max_centers, name="max_centers")
        n_training_points = data.n_training_points
        initial_power_candidates_sq = kernel.diagonal(data.X_candidates)
        initial_residual = data.y_candidates
        return cls(
            initial_power_candidates_sq=initial_power_candidates_sq,
            power_candidates_sq=initial_power_candidates_sq.copy(),
            initial_residual=initial_residual,
            residual=initial_residual.copy(),
            basis_values=np.empty(
                (n_training_points, max_centers), dtype=float, order="F",
            ),
            selected_candidate_indices=np.empty((max_centers,), dtype=int),
            newton_coefficients=np.empty((max_centers,), dtype=float),
            newton_normalizers=np.empty((max_centers,), dtype=float),
            n_selected=0,
        )

    # ----- Fit-loop mutation -----

    def update(
        self,
        idx: int,
        normalizer: float,
        basis_values: np.ndarray,
    ) -> float:
        """Append one Newton basis column; return the Newton coefficient."""
        k = int(self.n_selected)
        if k >= self.selected_candidate_indices.shape[0]:
            raise ValueError("Cannot update state because it is already full.")

        coeff_new = float(self.residual[idx] / normalizer)
        self.residual -= coeff_new * basis_values
        self.power_candidates_sq -= basis_values**2
        np.maximum(self.power_candidates_sq, 0.0, out=self.power_candidates_sq)

        self.basis_values[:, k] = basis_values
        self.selected_candidate_indices[k] = idx
        self.newton_coefficients[k] = coeff_new
        self.newton_normalizers[k] = normalizer
        self.n_selected = k + 1
        return coeff_new

    # ----- Post-fit construction -----

    def slice_prefix(self, n_selected: int) -> "GreedySolverState":
        """Construct a retained prefix state from a full greedy path state."""
        n_selected = int(n_selected)
        k_full = int(self.n_selected)
        if n_selected < 0 or n_selected > k_full:
            raise ValueError(
                f"n_selected must be in [0, {k_full}] for this state; got {n_selected}."
            )

        basis_values = self.basis_values[:, :n_selected].copy()
        newton_coefficients = self.newton_coefficients[:n_selected].copy()

        residual = self.initial_residual - basis_values @ newton_coefficients
        power_candidates_sq = (
            self.initial_power_candidates_sq - np.sum(basis_values**2, axis=1)
        )
        power_candidates_sq = np.maximum(power_candidates_sq, 0.0)

        return GreedySolverState(
            initial_power_candidates_sq=self.initial_power_candidates_sq,
            power_candidates_sq=power_candidates_sq,
            initial_residual=self.initial_residual,
            residual=residual,
            basis_values=basis_values,
            selected_candidate_indices=self.selected_candidate_indices[:n_selected].copy(),
            newton_coefficients=newton_coefficients,
            newton_normalizers=self.newton_normalizers[:n_selected].copy(),
            n_selected=int(n_selected),
        )

    # ----- Coefficient conversion -----

    def kernel_coefficients(self) -> np.ndarray:
        """Convert retained Newton coefficients to selected-kernel coefficients.

        With selected-center Newton matrix ``V_S`` and Newton coefficients ``c``,
        predictions satisfy

            f(x) = V_x c = K(x, S) alpha,

        where ``alpha`` solves ``V_S.T alpha = c``.  ``V_S`` is lower triangular,
        so this uses explicit back substitution on the upper-triangular transpose.
        """
        n_selected = int(self.selected_candidate_indices.shape[0])
        if n_selected == 0:
            return np.empty((0,), dtype=float)

        s_idx = self.selected_candidate_indices
        selected_basis = np.asarray(
            self.basis_values[s_idx, :n_selected], dtype=float
        )
        coefficients = np.asarray(self.newton_coefficients, dtype=float).reshape(-1)
        if coefficients.shape[0] != n_selected:
            raise ValueError(
                "newton_coefficients length must match selected centers; "
                f"got {coefficients.shape[0]} and {n_selected}."
            )

        upper = selected_basis.T
        alpha = np.empty(n_selected, dtype=float)
        diag_tol = np.finfo(float).tiny
        for i in range(n_selected - 1, -1, -1):
            diag = float(upper[i, i])
            if not np.isfinite(diag) or abs(diag) <= diag_tol:
                raise ValueError(
                    "Cannot convert Newton coefficients to kernel coefficients: "
                    f"near-zero triangular diagonal at index {i}: {diag!r}."
                )
            rhs = float(coefficients[i])
            if i + 1 < n_selected:
                rhs -= float(upper[i, i + 1:] @ alpha[i + 1:])
            alpha[i] = rhs / diag
        return alpha


# =============================================================================
# VKOGARebuilt public solver
# =============================================================================

# stop_reason values returned by VKOGARebuilt.fit()
STOP_MAX_CENTERS = "max_centers"
STOP_ALL_CANDIDATES = "all_candidates_selected"
STOP_POWER_TOO_SMALL = "power_too_small"
STOP_RESIDUAL_TOO_SMALL = "residual_too_small"


@dataclass
class VKOGARebuilt:
    """Minimal public rebuilt VKOGA-style solver.

    This first public class wires together training data validation,
    beta-greedy selection, Newton-basis updates, and prediction.  It intentionally
    does not implement wrapper/PUM/experiment-engine integration.
    """

    kernel: Kernel
    point_selection_rule: BetaGreedyPointSelectionRule
    max_centers: int
    model_selection_rule: BaseModelSelectionRule = field(
        default_factory=lambda: BestScoreModelSelectionRule(
            score_name="residual_max",
            rule_name="keep_best_rmax",
        )
    )
    power_tol: float = 1e-14
    residual_tol: float = 0.0
    verbose: bool = False
    report_every: int = 10
    store_cond: bool = False

    def __post_init__(self) -> None:
        self.max_centers = validate_positive_int(self.max_centers, name="max_centers")
        self.report_every = validate_positive_int(self.report_every, name="report_every")
        self.power_tol = float(self.power_tol)
        if not np.isfinite(self.power_tol) or self.power_tol < 0.0:
            raise ValueError("power_tol must be finite and >= 0.")
        self.residual_tol = float(self.residual_tol)
        if not np.isfinite(self.residual_tol) or self.residual_tol < 0.0:
            raise ValueError("residual_tol must be finite and >= 0.")
        self.verbose = bool(self.verbose)
        self.is_fitted_ = False
        self.n_selected_ = 0
        self.n_selected_full_ = 0
        self.stop_reason_ = "not_fitted"
        self._selected_kernel_coefficients: np.ndarray | None = None
        self._selected_centers: np.ndarray | None = None
        self._predict_wall_time_sec: float | None = None
        self._predict_n_points: int | None = None

    def __getstate__(self) -> dict:
        """Pickle solver state controlled by ``_pickle_mode``.

        ``_pickle_mode`` is consumed (popped) during serialisation and
        does not appear in the deserialised object.

        Modes
        -----
        ``"full"``
            Preserve all fit-time state including ``retained_state_`` with
            its Newton basis matrix.  Largest pickle, enables full diagnostics
            after deserialisation.
        ``"default"`` (default when ``_pickle_mode`` is unset)
            Prediction-only pickle.  Only the fields needed by ``predict()``
            are serialised — the smallest possible footprint.
        """
        state = dict(self.__dict__)
        pickle_mode = state.pop("_pickle_mode", "default")

        # 未进入 fit 状态时直接返回全部 __dict__
        if not state.get("is_fitted_", False):
            return state

        if pickle_mode == "full":
            return state

        # default: predict 必需字段 + 轻量诊断（fit 时间、终止原因、配置）
        # 规则对象用 rule_name 字符串代替完整对象
        if "point_selection_rule" in state:
            state["_point_selection_rule_name"] = state["point_selection_rule"].rule_name
        if "model_selection_rule" in state:
            state["_model_selection_rule_name"] = state["model_selection_rule"].rule_name

        predict_keys = {
            "is_fitted_", "n_selected_",
            "kernel",
            "_selected_centers", "_selected_kernel_coefficients",
            "_fit_time_sec", "stop_reason_",
            # config（小体积，便于反序列化后查看求解器配置）
            "max_centers", "power_tol", "residual_tol", "store_cond",
            "_point_selection_rule_name", "_model_selection_rule_name",
        }
        return {k: state[k] for k in predict_keys if k in state}

    def _require_fitted(self, method_name: str) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                f"VKOGARebuilt.{method_name}() called before fit()."
            )

    def _print_fit_started(self, training_data: GreedyTrainingData) -> None:
        if not self.verbose:
            return
        print(
            "[VKOGARebuilt] started "
            f"n_training_points={training_data.n_training_points} "
            f"max_centers={self.max_centers} "
            f"point_selection={self.point_selection_rule.rule_name} "
            f"model_selection={self.model_selection_rule.rule_name}"
        )

    def _print_fit_progress(self, record: GreedyStepRecord) -> None:
        if not self.verbose or record.iteration % self.report_every != 0:
            return
        print(
            "[VKOGARebuilt] progress "
            f"iteration={record.iteration}/{self.max_centers} "
            f"selected_candidate_index={record.selected_candidate_index} "
            f"residual_rmse={record.residual_rmse:.6e} "
            f"residual_max={record.residual_max:.6e} "
            f"power_max={record.power_max:.6e}"
        )

    def _print_fit_finished(self) -> None:
        if not self.verbose:
            return
        residual_rmse = float(np.sqrt(np.mean(self.retained_state_.residual**2)))
        print(
            "[VKOGARebuilt] finished "
            f"stop_reason={self.stop_reason_} "
            f"n_selected_full={self.n_selected_full_} "
            f"retained_iteration={self.retained_iteration_} "
            f"retained_centers={self.retained_state_.selected_candidate_indices.size} "
            f"retained_residual_rmse={residual_rmse:.6e}"
        )

    def _compute_new_basis_values(
        self,
        data: GreedyTrainingData,
        state: GreedySolverState,
        selected_candidate_index: int,
    ) -> tuple[int, float, np.ndarray]:
        """Compute one normalized Newton basis function against the active state prefix."""
        idx = validate_index_range(
            selected_candidate_index,
            lo=0,
            hi=len(state.residual),
            name="selected_candidate_index",
        )
        normalizer_sq = float(state.power_candidates_sq[idx])
        normalizer = float(np.sqrt(max(normalizer_sq, 0.0)))
        if normalizer <= float(self.power_tol):
            raise ValueError(
                "Cannot compute new basis values because selected candidate power "
                f"is too small: sqrt(power_candidates_sq[{idx}])={normalizer:.6e} "
                f"<= power_tol={float(self.power_tol):.6e}."
            )

        x_selected = data.X_candidates[idx: idx + 1]
        basis_values_new = self.kernel.eval(data.X_candidates, x_selected)[:, 0]

        # Gram-Schmidt against active prefix: v_new = K(X, x_selected) - V_S @ V_S[idx]
        k = int(state.n_selected)
        if k > 0:
            old_values_at_selected = state.basis_values[idx, :k]
            basis_values_new = basis_values_new - state.basis_values[:, :k] @ old_values_at_selected

        basis_values_new = basis_values_new / normalizer
        return idx, normalizer, np.asarray(basis_values_new, dtype=float)

    def fit(
        self,
        X,
        y,
    ) -> "VKOGARebuilt":
        """Fit a greedy Newton-basis expansion.

        ``X`` contains training points and ``y`` contains the corresponding
        target values. Greedy centers are selected from rows of ``X``.
        """
        # --- 1. Initialisation ---
        _t0 = time.perf_counter()
        training_data = build_greedy_training_data(X, y)
        state = GreedySolverState.create(
            training_data,
            self.kernel,
            max_centers=min(self.max_centers, training_data.n_training_points),
        )
        history = GreedyHistory(records=[])
        stop_reason = STOP_MAX_CENTERS
        self._print_fit_started(training_data)

        # --- 2. Greedy selection loop ---
        for _iteration in range(self.max_centers):
            if state.n_selected == training_data.n_training_points:
                stop_reason = STOP_ALL_CANDIDATES
                break

            abs_residual = np.abs(state.residual)
            power_candidates = np.sqrt(np.maximum(state.power_candidates_sq, 0.0))
            selected_mask = np.zeros(training_data.n_training_points, dtype=bool)
            selected_mask[
                state.selected_candidate_indices[: state.n_selected]
            ] = True

            selection = self.point_selection_rule.select(
                abs_residual,
                power_candidates,
                selected_mask,
            )
            try:
                idx, normalizer, basis_values = self._compute_new_basis_values(
                    training_data,
                    state,
                    selection.candidate_index,
                )
            except ValueError as exc:
                if "too small" not in str(exc):
                    raise
                stop_reason = STOP_POWER_TOO_SMALL
                break

            coeff_new = state.update(idx, normalizer, basis_values)
            n_selected = int(state.n_selected)
            residual_rmse = float(np.sqrt(np.mean(state.residual**2)))
            residual_max = float(np.max(np.abs(state.residual)))
            power_max = float(
                np.max(np.sqrt(np.maximum(state.power_candidates_sq, 0.0)))
            )

            condition_number = None
            if self.store_cond and self.kernel.is_spd and n_selected <= 2000:
                centers = training_data.X_candidates[
                    state.selected_candidate_indices[:n_selected]
                ]
                K = self.kernel.eval(centers, centers)
                K = 0.5 * (K + K.T)
                eigvals = np.linalg.eigvalsh(K)
                condition_number = float(eigvals[-1] / eigvals[0])

            record = GreedyStepRecord(
                iteration=n_selected,
                n_selected=n_selected,
                selected_candidate_index=int(selection.candidate_index),
                selection_rule_name=str(selection.rule_name),
                selection_score=float(selection.score),
                selection_log_score=float(selection.log_score),
                selected_abs_residual=float(selection.abs_residual),
                selected_power=float(selection.power),
                normalizer=float(normalizer),
                coefficient=float(coeff_new),
                residual_rmse=residual_rmse,
                residual_max=residual_max,
                power_max=power_max,
                condition_number=condition_number,
            )
            history.append(record)
            self._print_fit_progress(record)
            if self.residual_tol > 0.0 and residual_rmse <= self.residual_tol:
                stop_reason = STOP_RESIDUAL_TOO_SMALL
                break

        # --- 3. Model selection & teardown ---
        self.training_data_ = training_data
        self.history_ = history
        model_selection_result = self.model_selection_rule.select(
            self.history_.records_to_dicts()
        )
        self.retained_iteration_ = int(model_selection_result.retained_iteration)
        retained_state = state.slice_prefix(
            self.retained_iteration_,
        )
        self._selected_kernel_coefficients = retained_state.kernel_coefficients()
        self._selected_centers = training_data.X_candidates[
            retained_state.selected_candidate_indices
        ].copy()
        self.stop_reason_ = stop_reason
        self.is_fitted_ = True
        self.n_selected_full_ = int(state.n_selected)
        self.n_selected_ = int(retained_state.selected_candidate_indices.shape[0])
        self.retained_state_ = retained_state
        self._fit_time_sec = time.perf_counter() - _t0
        self._print_fit_finished()
        return self

    def predict(self, X) -> np.ndarray:
        """Predict values at query points using the selected Newton basis."""
        self._require_fitted(method_name="predict")
        X_query = as_valid_point_matrix(X, name="X")
        _t0 = time.perf_counter()
        if self.n_selected_ == 0:
            result = np.zeros(X_query.shape[0], dtype=float)
        else:
            result = np.asarray(
                self.kernel.eval_prod(
                    X_query, self.selected_centers_,
                    np.asarray(self._selected_kernel_coefficients, dtype=float),
                ),
                dtype=float,
            ).ravel()
        self._predict_wall_time_sec = time.perf_counter() - _t0
        self._predict_n_points = X_query.shape[0]
        return result

    @property
    def selected_centers_(self) -> np.ndarray:
        """Return retained selected centers used by the fast prediction model."""
        self._require_fitted(method_name="selected_centers_")
        return np.asarray(self._selected_centers, dtype=float).copy()

    @property
    def selected_kernel_coefficients_(self) -> np.ndarray:
        """Return retained selected-kernel coefficients."""
        self._require_fitted(method_name="selected_kernel_coefficients_")
        return np.asarray(self._selected_kernel_coefficients, dtype=float).copy()

    @property
    def fit_time_sec(self) -> float:
        """Return wall time in seconds for the fit."""
        self._require_fitted(method_name="fit_time_sec")
        return float(self._fit_time_sec)

    def predict_summary(self) -> Dict[str, Any]:
        """Return a JSON-safe summary of the most recent ``predict()`` call.

        Returns
        -------
        predict_summary structure:

        ========================= =============================================
        Key                       Description
        ========================= =============================================
        n_points                  number of predicted points
        n_centers                 number of selected centers (model complexity)
        predict_wall_time_sec     predict() wall time
        ========================= =============================================

        Raises
        ------
        RuntimeError
            If ``predict()`` has not been called yet.
        """
        self._require_fitted(method_name="predict_summary")
        if self._predict_wall_time_sec is None:
            raise RuntimeError(
                "VKOGARebuilt.predict_summary() called before predict()."
            )
        return {
            "n_points": self._predict_n_points,
            "n_centers": int(self.n_selected_),
            "predict_wall_time_sec": self._predict_wall_time_sec,
        }

    def estimate_cond(self) -> float:
        """Compute the condition number of the retained kernel matrix K(C, C).

        Uses ``eigvalsh`` on the symmetrized kernel matrix of the selected
        centers.  Unrestricted — no size or SPD guard.  Use on small matrices
        only (eigvalsh is O(n³)).

        Returns
        -------
        float
            Condition number = λ_max / λ_min.
        """
        self._require_fitted(method_name="estimate_cond")
        centers = self.selected_centers_
        K = self.kernel.eval(centers, centers)
        K = 0.5 * (K + K.T)
        eigvals = np.linalg.eigvalsh(K)
        return float(eigvals[-1] / eigvals[0])

    def result_summary(self) -> dict:
        """Return a compact JSON-safe summary of the fitted solver result.

        Returns
        -------
        VKOGARebuilt result_summary structure:

        ===================== =================================================
        Key                   Description
        ===================== =================================================
        data                  training data info:
                                n_training_points, dim
        solver_type           ``"VKOGARebuilt"``
        config                solver configuration:
                                max_centers, power_tol, residual_tol,
                                kernel, point_selection, model_selection
        stop_reason           solver termination reason (string)
        fit_time_sec          wall time for the fit
        model                 retained model info:
                                n_selected_centers,
                                selected_candidate_indices,
                                kernel_coefficients (count, min, max,
                                mean, median, l2_norm, linf_norm, values),
                                quality
        training_history      list of per-iteration GreedyStepRecord dicts
        ===================== =================================================

        Each ``training_history`` record:

        ======================== =============================================
        Field                    Description
        ======================== =============================================
        iteration                1-based greedy step index
        n_selected               number of centers selected after this step
        selected_candidate_index training-data row index of the new center
        selection_rule_name      point-selection rule at this step
        selection_score          raw point-selection score
        selection_log_score      log-score for stable comparison
        selected_abs_residual    abs residual at the selected center
        selected_power           power function at the selected center
        normalizer               Newton basis normalizer
        coefficient              newly appended Newton coefficient
        residual_rmse            training residual RMSE after update
        residual_max             max abs training residual after update
        power_max                max power-function value after update
        condition_number         λ_max/λ_min (optional, see ``store_cond``)
        ======================== =============================================
        """
        self._require_fitted(method_name="result_summary")
        selected_indices = [
            int(index) for index in self.retained_state_.selected_candidate_indices
        ]
        coeffs = np.asarray(self._selected_kernel_coefficients, dtype=float)
        abs_coeffs = np.abs(coeffs)

        quality: Dict[str, Any] = {}
        if self.retained_iteration_ > 0:
            record = self.history_.records[self.retained_iteration_ - 1]
            quality["residual_rmse"] = _json_safe_float(record.residual_rmse)
            quality["residual_max"] = _json_safe_float(record.residual_max)
            quality["power_max"] = _json_safe_float(record.power_max)
            if record.condition_number is not None:
                quality["condition_number"] = _json_safe_float(
                    record.condition_number
                )

        return {
            "data": {
                "n_training_points": int(self.training_data_.n_training_points),
                "dim": int(self.training_data_.dim),
            },
            "solver_type": "VKOGARebuilt",
            "config": {
                "max_centers": int(self.max_centers),
                "power_tol": _json_safe_float(self.power_tol),
                "residual_tol": _json_safe_float(self.residual_tol),
                "kernel": self.kernel.to_dict(),
                "point_selection": self.point_selection_rule.to_dict(),
                "model_selection": self.model_selection_rule.to_dict(),
            },
            "stop_reason": str(self.stop_reason_),
            "fit_time_sec": self._fit_time_sec,
            "model": {
                "n_selected_centers": int(self.n_selected_),
                "selected_candidate_indices": selected_indices,
                "kernel_coefficients": {
                    "count": int(coeffs.size),
                    "min": _json_safe_float(np.min(coeffs)),
                    "max": _json_safe_float(np.max(coeffs)),
                    "mean": _json_safe_float(np.mean(coeffs)),
                    "median": _json_safe_float(np.median(coeffs)),
                    "l2_norm": _json_safe_float(np.linalg.norm(coeffs)),
                    "linf_norm": _json_safe_float(np.max(abs_coeffs)),
                    "values": coeffs.tolist(),
                },
                "quality": quality,
            },
            "training_history": self.history_.records_to_dicts(),
        }
