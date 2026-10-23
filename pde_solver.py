"""PDE greedy solver built on generalized Newton basis updates."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data import PDETrainingData
from .history import GreedyHistory, GreedyStepRecord
from .model_selection import BaseModelSelectionRule, BestScoreModelSelectionRule
from .pde_composer import OperatorComposer
from .point_selection import BetaGreedyPointSelectionRule, point_selection_rule_from_config
from .validation import as_valid_point_matrix, validate_index_range, validate_positive_int


PDE_STOP_MAX_CENTERS = "max_centers"
PDE_STOP_ALL_CANDIDATES = "all_candidates_selected"
PDE_STOP_POWER_TOO_SMALL = "power_too_small"
PDE_STOP_RESIDUAL_TOO_SMALL = "residual_too_small"


@dataclass
class PDEGreedyState:
    """State of a partial PDE greedy Newton-basis expansion."""

    data: PDETrainingData
    initial_power_candidates_sq: np.ndarray
    power_candidates_sq: np.ndarray
    initial_residual: np.ndarray
    residual: np.ndarray
    basis_values: np.ndarray
    selected_indices: np.ndarray
    selected_types: np.ndarray
    newton_coefficients: np.ndarray
    newton_normalizers: np.ndarray
    n_selected: int = 0

    @classmethod
    def create(
        cls,
        data: PDETrainingData,
        composer: OperatorComposer,
        max_centers: int,
    ) -> "PDEGreedyState":
        max_centers = validate_positive_int(max_centers, name="max_centers")
        initial_power_candidates_sq = np.asarray(
            composer.power_diagonal(data.functional_types, data.X_all),
            dtype=float,
        ).reshape(-1)
        if initial_power_candidates_sq.shape[0] != data.n_trial:
            raise ValueError(
                "composer.power_diagonal() must return one value per trial point; "
                f"got {initial_power_candidates_sq.shape[0]} and {data.n_trial}."
            )
        if not np.all(np.isfinite(initial_power_candidates_sq)):
            raise ValueError("initial PDE power diagonal must contain only finite values.")
        if np.any(initial_power_candidates_sq < 0.0):
            raise ValueError("initial PDE power diagonal must be non-negative.")

        initial_residual = data.rhs_all
        return cls(
            data=data,
            initial_power_candidates_sq=initial_power_candidates_sq,
            power_candidates_sq=initial_power_candidates_sq.copy(),
            initial_residual=initial_residual,
            residual=initial_residual.copy(),
            basis_values=np.empty((data.n_trial, max_centers), dtype=float, order="F"),
            selected_indices=np.empty((max_centers,), dtype=int),
            selected_types=np.empty((max_centers,), dtype=int),
            newton_coefficients=np.empty((max_centers,), dtype=float),
            newton_normalizers=np.empty((max_centers,), dtype=float),
            n_selected=0,
        )

    def update(
        self,
        idx: int,
        functional_type: int,
        normalizer: float,
        basis_values: np.ndarray,
    ) -> float:
        """Append one Newton basis column and update residual/power arrays."""
        k = int(self.n_selected)
        if k >= self.selected_indices.shape[0]:
            raise ValueError("Cannot update PDE state because it is already full.")

        basis_arr = np.asarray(basis_values, dtype=float).reshape(-1)
        if basis_arr.shape[0] != self.data.n_trial:
            raise ValueError(
                "basis_values length must match data.n_trial; "
                f"got {basis_arr.shape[0]} and {self.data.n_trial}."
            )

        coeff_new = float(self.residual[idx] / normalizer)
        self.residual -= coeff_new * basis_arr
        self.power_candidates_sq -= basis_arr**2
        np.maximum(self.power_candidates_sq, 0.0, out=self.power_candidates_sq)

        self.basis_values[:, k] = basis_arr
        self.selected_indices[k] = int(idx)
        self.selected_types[k] = int(functional_type)
        self.newton_coefficients[k] = coeff_new
        self.newton_normalizers[k] = float(normalizer)
        self.n_selected = k + 1
        return coeff_new

    def slice_prefix(self, n_selected: int) -> "PDEGreedyState":
        """Return a retained prefix state reconstructed from initial quantities."""
        n_selected = int(n_selected)
        if n_selected < 0 or n_selected > int(self.n_selected):
            raise ValueError(
                f"n_selected must be in [0, {self.n_selected}]; got {n_selected}."
            )

        basis_values = self.basis_values[:, :n_selected].copy()
        newton_coefficients = self.newton_coefficients[:n_selected].copy()
        residual = self.initial_residual - basis_values @ newton_coefficients
        power_candidates_sq = (
            self.initial_power_candidates_sq - np.sum(basis_values**2, axis=1)
        )
        power_candidates_sq = np.maximum(power_candidates_sq, 0.0)

        return PDEGreedyState(
            data=self.data,
            initial_power_candidates_sq=self.initial_power_candidates_sq,
            power_candidates_sq=power_candidates_sq,
            initial_residual=self.initial_residual,
            residual=residual,
            basis_values=basis_values,
            selected_indices=self.selected_indices[:n_selected].copy(),
            selected_types=self.selected_types[:n_selected].copy(),
            newton_coefficients=newton_coefficients,
            newton_normalizers=self.newton_normalizers[:n_selected].copy(),
            n_selected=n_selected,
        )

    def kernel_coefficients(self) -> np.ndarray:
        """Convert retained Newton coefficients to selected representer coefficients."""
        n_selected = int(self.n_selected)
        if n_selected == 0:
            return np.empty((0,), dtype=float)

        selected_basis = np.asarray(
            self.basis_values[self.selected_indices[:n_selected], :n_selected],
            dtype=float,
        )
        coefficients = np.asarray(
            self.newton_coefficients[:n_selected],
            dtype=float,
        ).reshape(-1)
        upper = selected_basis.T
        alpha = np.empty(n_selected, dtype=float)
        diag_tol = np.finfo(float).tiny
        for i in range(n_selected - 1, -1, -1):
            diag = float(upper[i, i])
            if not np.isfinite(diag) or abs(diag) <= diag_tol:
                raise ValueError(
                    "Cannot convert PDE Newton coefficients: "
                    f"near-zero triangular diagonal at index {i}: {diag!r}."
                )
            rhs = float(coefficients[i])
            if i + 1 < n_selected:
                rhs -= float(upper[i, i + 1:] @ alpha[i + 1:])
            alpha[i] = rhs / diag
        return alpha


@dataclass
class PDEGreedySolver:
    """Phase-1 PDE greedy solver over prebuilt collocation data."""

    composer: OperatorComposer
    max_centers: int
    point_selection_rule: BetaGreedyPointSelectionRule = field(
        default_factory=lambda: point_selection_rule_from_config("f_greedy")
    )
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
        self._selected_coefficients: np.ndarray | None = None
        self._selected_centers: np.ndarray | None = None
        self._selected_center_types: np.ndarray | None = None
        self._fit_time_sec: float | None = None

    def _require_fitted(self, method_name: str) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"PDEGreedySolver.{method_name}() called before fit().")

    def _compute_new_basis_values(
        self,
        data: PDETrainingData,
        state: PDEGreedyState,
        selected_index: int,
    ) -> tuple[int, int, float, np.ndarray]:
        idx = validate_index_range(
            selected_index,
            lo=0,
            hi=data.n_trial,
            name="selected_index",
        )
        functional_type = int(data.functional_types[idx])
        normalizer_sq = float(state.power_candidates_sq[idx])
        normalizer = float(np.sqrt(max(normalizer_sq, 0.0)))
        if normalizer <= self.power_tol:
            raise ValueError(
                "Cannot compute PDE Newton basis because selected functional power "
                f"is too small: sqrt(power_candidates_sq[{idx}])={normalizer:.6e} "
                f"<= power_tol={self.power_tol:.6e}."
            )

        X_all = data.X_all
        raw = self.composer.eval_candidate(
            functional_type,
            X_all,
            data.functional_types,
            X_all[idx:idx + 1],
        )
        basis_values_new = np.asarray(raw, dtype=float).reshape(-1)
        if basis_values_new.shape[0] != data.n_trial:
            raise ValueError(
                "composer.eval_candidate() must return one value per trial point; "
                f"got {basis_values_new.shape[0]} and {data.n_trial}."
            )

        k = int(state.n_selected)
        if k > 0:
            old_values_at_selected = state.basis_values[idx, :k]
            basis_values_new = basis_values_new - state.basis_values[:, :k] @ old_values_at_selected
        basis_values_new = basis_values_new / normalizer
        return idx, functional_type, normalizer, basis_values_new

    def fit(self, data: PDETrainingData) -> "PDEGreedySolver":
        """Fit a PDE greedy Newton-basis expansion."""
        if not isinstance(data, PDETrainingData):
            raise TypeError("data must be a PDETrainingData instance.")

        t0 = time.perf_counter()
        max_centers = min(self.max_centers, data.n_trial)
        state = PDEGreedyState.create(data, self.composer, max_centers=max_centers)
        history = GreedyHistory(records=[])
        stop_reason = PDE_STOP_MAX_CENTERS

        for _iteration in range(self.max_centers):
            if state.n_selected == data.n_trial:
                stop_reason = PDE_STOP_ALL_CANDIDATES
                break

            abs_residual = np.abs(state.residual)
            power_candidates = np.sqrt(np.maximum(state.power_candidates_sq, 0.0))
            selected_mask = np.zeros(data.n_trial, dtype=bool)
            selected_mask[state.selected_indices[:state.n_selected]] = True

            selection = self.point_selection_rule.select(
                abs_residual,
                power_candidates,
                selected_mask,
            )
            try:
                idx, functional_type, normalizer, basis_values = self._compute_new_basis_values(
                    data,
                    state,
                    selection.candidate_index,
                )
            except ValueError as exc:
                if "too small" not in str(exc):
                    raise
                stop_reason = PDE_STOP_POWER_TOO_SMALL
                break

            coeff_new = state.update(idx, functional_type, normalizer, basis_values)
            n_selected = int(state.n_selected)
            residual_rmse = float(np.sqrt(np.mean(state.residual**2)))
            residual_max = float(np.max(np.abs(state.residual)))
            power_max = float(np.max(np.sqrt(np.maximum(state.power_candidates_sq, 0.0))))

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
                condition_number=None,
            )
            history.append(record)
            if self.residual_tol > 0.0 and residual_rmse <= self.residual_tol:
                stop_reason = PDE_STOP_RESIDUAL_TOO_SMALL
                break

        self.training_data_ = data
        self.history_ = history
        self.n_selected_full_ = int(state.n_selected)
        if len(history) == 0:
            self.retained_iteration_ = 0
            retained_state = state.slice_prefix(0)
        else:
            model_selection_result = self.model_selection_rule.select(history.records_to_dicts())
            self.retained_iteration_ = int(model_selection_result.retained_iteration)
            retained_state = state.slice_prefix(self.retained_iteration_)

        self.retained_state_ = retained_state
        self._selected_coefficients = retained_state.kernel_coefficients()
        selected_indices = retained_state.selected_indices[:retained_state.n_selected]
        self._selected_centers = data.X_all[selected_indices].copy()
        self._selected_center_types = retained_state.selected_types[:retained_state.n_selected].copy()
        self.n_selected_ = int(retained_state.n_selected)
        self.stop_reason_ = stop_reason
        self.is_fitted_ = True
        self._fit_time_sec = time.perf_counter() - t0
        return self

    def predict(self, X) -> np.ndarray:
        """Predict solution values at query points."""
        self._require_fitted("predict")
        X_query = as_valid_point_matrix(X, name="X")
        if self.n_selected_ == 0:
            return np.zeros(X_query.shape[0], dtype=float)
        mixed = self.composer.eval_predict(
            self._selected_center_types,
            X_query,
            self._selected_centers,
        )
        return np.asarray(mixed @ self._selected_coefficients, dtype=float).reshape(-1)

    def predict_operator(self, X) -> np.ndarray:
        """Predict ``L[u]`` at query points."""
        self._require_fitted("predict_operator")
        X_query = as_valid_point_matrix(X, name="X")
        if self.n_selected_ == 0:
            return np.zeros(X_query.shape[0], dtype=float)
        mixed = self.composer.eval_operator_predict(
            self._selected_center_types,
            X_query,
            self._selected_centers,
        )
        return np.asarray(mixed @ self._selected_coefficients, dtype=float).reshape(-1)

    @property
    def selected_centers_(self) -> np.ndarray:
        self._require_fitted("selected_centers_")
        return np.asarray(self._selected_centers, dtype=float).copy()

    @property
    def selected_center_types_(self) -> np.ndarray:
        self._require_fitted("selected_center_types_")
        return np.asarray(self._selected_center_types, dtype=int).copy()

    @property
    def selected_coefficients_(self) -> np.ndarray:
        self._require_fitted("selected_coefficients_")
        return np.asarray(self._selected_coefficients, dtype=float).copy()

    @property
    def fit_time_sec(self) -> float:
        self._require_fitted("fit_time_sec")
        return float(self._fit_time_sec)

    def result_summary(self) -> dict[str, Any]:
        self._require_fitted("result_summary")
        coeffs = np.asarray(self._selected_coefficients, dtype=float)
        return {
            "solver_type": "PDEGreedySolver",
            "data": self.training_data_.to_dict(),
            "config": {
                "max_centers": int(self.max_centers),
                "power_tol": float(self.power_tol),
                "residual_tol": float(self.residual_tol),
                "point_selection": self.point_selection_rule.to_dict(),
                "model_selection": self.model_selection_rule.to_dict(),
            },
            "stop_reason": self.stop_reason_,
            "fit_time_sec": float(self._fit_time_sec or 0.0),
            "model": {
                "n_selected_centers": int(self.n_selected_),
                "selected_candidate_indices": [
                    int(index) for index in self.retained_state_.selected_indices[:self.n_selected_]
                ],
                "selected_types": [
                    int(t) for t in self.retained_state_.selected_types[:self.n_selected_]
                ],
                "coefficients": {
                    "count": int(coeffs.shape[0]),
                    "l2_norm": float(np.linalg.norm(coeffs)) if coeffs.size else 0.0,
                    "linf_norm": float(np.max(np.abs(coeffs))) if coeffs.size else 0.0,
                },
            },
            "training_history": self.history_.records_to_dicts(),
        }
