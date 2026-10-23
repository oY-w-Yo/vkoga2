import json

import numpy as np
import pytest

from vkoga2.history import GreedyHistory, GreedyStepRecord


def _record(iteration=1, rmse=0.1):
    return GreedyStepRecord(
        iteration=iteration,
        n_selected=iteration,
        selected_candidate_index=iteration - 1,
        selection_rule_name="f_greedy",
        selection_score=1.0,
        selection_log_score=0.0,
        selected_abs_residual=1.0,
        selected_power=1.0,
        normalizer=1.0,
        coefficient=0.5,
        residual_rmse=rmse,
        residual_max=0.3,
        power_max=0.4,
    )


def test_greedy_step_record_to_dict_json_safe():
    record = _record()

    data = record.to_dict()

    json.dumps(data)
    assert data["iteration"] == 1
    assert data["residual_rmse"] == 0.1


def test_greedy_history_append_len_and_records_to_dicts():
    history = GreedyHistory(records=[])

    history.append(_record(iteration=1, rmse=0.3))
    history.append(_record(iteration=2, rmse=0.1))

    assert len(history) == 2
    rows = history.records_to_dicts()
    json.dumps(rows)
    assert rows[0]["iteration"] == 1
    assert rows[1]["iteration"] == 2


def test_greedy_history_to_dict():
    history = GreedyHistory(records=[_record(iteration=1)])

    data = history.to_dict()

    json.dumps(data)
    assert data["n_steps"] == 1
    assert data["records"][0]["n_selected"] == 1


def test_greedy_history_get_field_values():
    history = GreedyHistory(records=[_record(iteration=1, rmse=0.3), _record(iteration=2, rmse=0.1)])

    values = history.get_field_values("residual_rmse")

    np.testing.assert_allclose(values, np.array([0.3, 0.1]))


def test_greedy_history_unknown_field_raises():
    history = GreedyHistory(records=[_record()])

    with pytest.raises(ValueError, match="Unknown history field"):
        history.get_field_values("not_a_field")


def test_greedy_step_record_is_frozen():
    record = _record()

    with pytest.raises(Exception):
        record.iteration = 99
