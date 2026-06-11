# vkoga2 Usage Note

This note records the current usage-level decisions for `vkoga2`. It is intentionally a working note rather than a final README.

## 1. Purpose

`VKOGARebuilt` is the rebuilt core solver for VKOGA-style greedy kernel approximation.

Current role:

- provide a clean NumPy-based VKOGA core;
- keep point selection, model selection, history, retained prefix state, and summaries explicit;
- serve as the backend for the future `greedy_kernel_rebuilt` approximator wrapper.

The core solver is independent of PUM, the experiment engine, and approximator wrappers.

## 2. Quick start

Recommended current default configuration:

```python
from vkoga2 import (
    VKOGARebuilt,
    Matern32Kernel,
    model_selection_rule_from_config,
    point_selection_rule_from_config,
)

model = VKOGARebuilt(
    kernel=Matern32Kernel(gamma=gamma),
    point_selection_rule=point_selection_rule_from_config("f_greedy"),
    model_selection_rule=model_selection_rule_from_config("keep_best_rmax"),
    max_centers=max_centers,
)
```

Default convention:

- The framework/config layer defaults to `point_selection_rule="f_greedy"` and
  `model_selection_rule="keep_best_rmax"`.
- The core solver example constructs both rule objects explicitly, so the two
  choices are visible at the same level.
- `VKOGARebuilt` still has an internal default for `model_selection_rule`
  (`keep_best_rmax`) for backward compatibility, but callers should usually
  pass both rules explicitly in user-facing examples.

## 3. Fit / predict API

Basic usage:

```python
model.fit(X, y)
y_pred = model.predict(X_query)
summary = model.predict_summary()
```

Semantics:

- `X, y` are the training points and target values.
- Selected centers are rows of `X`.
- Prediction uses the retained prefix model selected after the greedy path is
  built.
- `predict_summary()` returns a lightweight dict with `n_points`, `n_centers`,
  and `predict_wall_time_sec`.  Raises ``RuntimeError`` if called before
  ``predict()``.

## 4. Package overview map

The package has one public solver core and several small policy/data modules.
The intended reading order is:

```text
vkoga2/
|-- validation.py
|   shared input validators
|-- data.py
|   training-data packaging
|-- kernels.py
|   immutable kernel objects and kernel_from_config(...)
|-- distances.py
|   pairwise distance computations
|-- point_selection.py
|   beta-greedy center-selection rules
|-- model_selection.py
|   retained-prefix selection after the greedy path is built
|-- history.py
|   scalar per-step records
|-- solver.py
|   Newton-basis state, fit loop, retained model, prediction, summaries
`-- __init__.py
    public import surface
```

## 4.1 Input/output map

Core solver input:

```text
VKOGARebuilt(
    kernel,
    point_selection_rule,
    model_selection_rule,
    max_centers,
    store_cond=False,
    ...
)

fit(X, y)
```

Core solver internal training data:

```text
build_greedy_training_data(...)
    X_candidates      shape (n_training_points, dim)
    y_candidates      shape (n_training_points,)
```

Core solver fitted outputs:

```text
training_data_                  validated GreedyTrainingData
history_                        GreedyHistory with scalar records
                                (includes condition_number per iteration
                                 when store_cond=True and SPD and n≤2000)
retained_iteration_             1-based retained path index (= n_selected_)
retained_state_                  stripped retained-prefix state
n_selected_                     retained center count (= retained_iteration_)
n_selected_full_                full greedy-path center count
fit_time_sec                    elapsed wall time during fit()
selected_centers_               retained center coordinates (property)
selected_kernel_coefficients_   selected-kernel coefficients (property)
```

Public prediction/reporting outputs:

```text
predict(X_query)                 -> y_pred
predict_summary()                -> JSON-safe predict metadata
                                   (n_points, n_centers,
                                    predict_wall_time_sec)
result_summary()                 -> JSON-safe unified fitted-result summary
estimate_cond()                  -> condition number of K(C, C) (unrestricted)
selected_centers_                -> copy of retained centers (property)
selected_kernel_coefficients_    -> copy of retained coefficients (property)
```

## 4.2 Fit dependency flow

The fit path is:

```text
raw X, y
  |
  v
data.build_greedy_training_data
  |
  v
GreedySolverState.create(data, kernel, max_centers)
  |
  v
repeat until stop:
    residual/power vectors from state
      |
      v
    point_selection_rule.select(...)
      |
      v
    compute_new_basis_values(data, kernel, state, idx)
      |
      v
    state.update(idx, normalizer, basis_values)
      |
      v
    history.append(GreedyStepRecord)
  |
  v
model_selection_rule.select(history)
  |
  v
state.slice_prefix(retained_iteration)
  |
  v
retained_state.kernel_coefficients()
  |
  v
compact fitted solver:
    strip fit-time basis matrices via state.strip_basis()
    keep selected centers + selected-kernel coefficients
```

The important dependency direction is one-way:

```text
data / kernels / selection / model_selection / history
    feed into
solver
```

The core package does not import PUM, experiment-engine config, artifacts, or
usercase code.

## 4.3 Predict dependency flow

After fitting, prediction does not reconstruct the Newton basis.  It uses the
cached selected-kernel representation:

```text
X_query
  |
  v
as_valid_point_matrix
  |
  v
kernel.eval_prod(X_query, selected_centers, kernel_coefficients)
  |
  v
y_pred
```

This is why fitted states can strip the large basis matrices while preserving
prediction behavior.

## 5. Point selection

Point selection chooses the next center from the training points.

Implemented point-selection module:

```text
point_selection.py
```

Main objects:

- `PointSelectionResult`
- `BetaGreedyPointSelectionRule`
- `point_selection_rule_from_config(...)`

Supported named rules:

```text
p_greedy          beta = 0
f_times_p_greedy  beta = 0.5
f_greedy          beta = 1
f_over_p_greedy   beta = infinity
beta_greedy       user-specified beta
```

The unified finite-beta score is:

```text
score_beta(x) = |r(x)|^beta * P(x)^(1-beta)
```

The implementation uses `log_score` internally for stable comparison and stores readable actual `score` in the result.

## 6. Model selection and retained prefix model

The solver separates:

```text
point selection:
    chooses the next center

model selection:
    chooses the retained iteration from the greedy path
```

Current model-selection module:

```text
model_selection.py
```

Main objects:

- `ModelSelectionResult`
- `KeepLastModelSelectionRule`
- `BestScoreModelSelectionRule`
- `model_selection_rule_from_config(...)`

Core retained-prefix principle:

```text
fit generates the full greedy path
model_selection chooses retained_iteration = j
retained model = first j selected centers / Newton basis functions
```

The retained model is a prefix of the generated greedy path. It is not a refit and does not recompute coefficients with a different objective.

Important fitted attributes:

```text
history_:
    scalar records for all successful greedy iterations

retained_iteration_:
    1-based retained iteration, equal to retained prefix length

retained_state_:
    prefix state used for prediction and retained-model reporting

selected_centers_:
    copy of retained center coordinates (property)

selected_kernel_coefficients_:
    selected-kernel coefficients for prediction (property)
```

The full greedy path is represented by scalar history records and
`n_selected_full_`.  The fitted solver does not retain a separate full numerical
state after model selection.

## 7. History

History records scalar summaries for the greedy path.

Current history module:

```text
history.py
```

Main objects:

- `GreedyStepRecord`
- `GreedyHistory`

History stores scalar path indicators only, such as:

- selected candidate index;
- selection score and log-score;
- selected residual and power;
- Newton normalizer and coefficient;
- residual and power indicators after the update;
- condition number (optional, when ``store_cond=True`` and SPD and n≤2000).

Large arrays such as basis matrices, residual vectors, power vectors, and kernel matrices remain in solver state, not history.

## 8. Summary API

`VKOGARebuilt` provides one unified JSON-safe fitted-result summary.

### `predict_summary()`

Lightweight summary of the most recent ``predict()`` call.

```text
n_points                  number of predicted points
n_centers                 number of selected centers (model complexity)
predict_wall_time_sec     predict() wall time in seconds
```

Raised ``RuntimeError`` if called before ``predict()``.

### `result_summary()`

Unified summary of the fitted solver result. Top-level keys are ordered for
inspection:

```text
data:
    n_training_points          number of training points
    dim                        input dimension
solver_type                    "VKOGARebuilt"
config:
    max_centers                user-specified budget
    power_tol                  power convergence threshold
    residual_tol               residual convergence threshold
    kernel                     kernel.to_dict() metadata
    point_selection            point-selection rule + parameters
    model_selection            model-selection rule + parameters
stop_reason                    string (e.g. "max_centers", "residual_too_small")
fit_time_sec                   wall time in seconds for fit()
model:
    n_selected_centers         retained center count
    selected_candidate_indices list of training-data row indices
    kernel_coefficients:
        count, min, max, mean, median, l2_norm, linf_norm
        values                 actual coefficient array as a list
    quality:
        residual_rmse          sqrt(mean(residual²)) over training points
        residual_max           max(abs(residual)) over training points
        power_max              max(power function) over training points
        condition_number       λ_max/λ_min (only when available)
training_history               list of per-step GreedyStepRecord dicts

    Each `training_history` record:

    | Field | Description |
    |---|---|
    | `iteration` | 1-based greedy step index |
    | `n_selected` | number of centers selected after this step |
    | `selected_candidate_index` | training-data row index of the new center |
    | `selection_rule_name` | point-selection rule at this step |
    | `selection_score` | raw point-selection score |
    | `selection_log_score` | log-score for stable comparison |
    | `selected_abs_residual` | abs residual at the selected center |
    | `selected_power` | power function at the selected center |
    | `normalizer` | Newton basis normalizer |
    | `coefficient` | newly appended Newton coefficient |
    | `residual_rmse` | training residual RMSE after update |
    | `residual_max` | max abs training residual after update |
    | `power_max` | max power-function value after update |
    | `condition_number` | λ_max/λ_min (optional, when `store_cond=True`) |
```

`training_history` contains all greedy steps (the full path), not only the
retained prefix. The number of records equals `n_selected_full_`.

All numerical values are JSON-safe (non-finite → null). The summary does not
contain large arrays — basis matrices and residual vectors stay in the retained
state and can be stripped before pickle.

## 9. Verbose progress reporting

Constructor parameters:

```python
verbose: bool = False
report_every: int = 10
```

``store_cond`` controls whether the solver computes the kernel-matrix
condition number at each greedy step and stores it in the history record.
Only effective when the kernel is SPD and ``n_selected ≤ 2000``.

When `verbose=True`, the solver prints:

- start summary;
- periodic progress lines every `report_every` successful iterations;
- final summary.

This is intended for long-running local fits.

## 10. Framework boundary

The core `vkoga2` package is the subject of this note.  It does not
import PUM, experiment-engine config, artifacts, or usercase code.

The framework adapter lives outside the core package:


```text
pum_framework/function_approximation/approximators/greedy_kernel_rebuilt/
`-- vkoga_interpolant_rebuilt.py
```

Its role is to make the independent solver usable by the existing PUM and
experiment-engine pipeline.  It is intentionally a thin wrapper:

```text
VKOGAInterpolantRebuilt.fit(X, y)
    builds VKOGARebuilt(...)
    delegates solver.fit(X, y)
    exposes result_summary(), fit_summary(), model_summary(), selected centers,
    and history
```

The experiment-engine integration points are:

```text
pum_framework/function_approximation/experiment_engine/builders.py
    builds VKOGAInterpolantRebuilt from ModelConfig

pum_framework/function_approximation/experiment_engine/run_artifacts.py
    owns model naming, model identity, runtime metadata, and artifact layout
```

For a VKOGA-focused presentation, this framework layer is mainly integration
evidence: the standalone solver can be plugged into existing experiments without
making the solver depend on the framework.

## 11. Advanced performance knobs

Most users should leave these at their defaults. They are exposed for large
local fits, profiling, and reproducing solver optimization experiments.

Current effective optimization decisions:

- F-order (column-major) workspace is hardcoded — benchmarks showed it is
  consistently faster than C-order for the Newton-basis access pattern;
- preallocated workspace removes repeated matrix growth / copy overhead;
- shared candidate/residual arrays avoid duplicated fit-loop work;
- projection uses direct ``matrix @ vector`` via NumPy (no chunking).

Representative heavy profiling showed rebuilt fit is substantially faster than old VKOGA under current benchmark settings. See:

```text
developing_or_refactor/vkoga_rewrite/vkoga_rebuilt_fit_optimization_status.md
```

for the current optimization status note.
