import math

import pytest

from vkoga2.model_selection import (
    BestScoreModelSelectionRule,
    KeepLastModelSelectionRule,
    ModelSelectionResult,
    model_selection_rule_from_config,
)


def _records():
    return [
        {
            "iteration": 0,
            "n_selected": 1,
            "residual_rmse": 0.3,
            "residual_max": 0.5,
        },
        {
            "iteration": 1,
            "n_selected": 2,
            "residual_rmse": 0.1,
            "residual_max": 0.4,
        },
        {
            "iteration": 2,
            "n_selected": 3,
            "residual_rmse": 0.2,
            "residual_max": 0.2,
        },
    ]


def test_keep_last_returns_last_record():
    rule = KeepLastModelSelectionRule()

    result = rule.select(_records())

    assert isinstance(result, ModelSelectionResult)
    assert result.retained_iteration == 2
    assert result.retained_n_centers == 3
    assert result.score_name == "last_record"
    assert math.isnan(result.score_value)
    assert result.rule_name == "keep_last"


def test_best_score_selects_minimum_named_score():
    rule = BestScoreModelSelectionRule(score_name="residual_rmse")

    result = rule.select(_records())

    assert result.retained_iteration == 1
    assert result.retained_n_centers == 2
    assert result.score_name == "residual_rmse"
    assert result.score_value == 0.1
    assert result.rule_name == "best_score"


def test_best_score_tie_selects_earliest_best_record():
    records = [
        {"iteration": 0, "n_selected": 1, "residual_max": 0.2},
        {"iteration": 1, "n_selected": 2, "residual_max": 0.1},
        {"iteration": 2, "n_selected": 3, "residual_max": 0.1},
    ]

    result = BestScoreModelSelectionRule().select(records)

    assert result.retained_iteration == 1
    assert result.retained_n_centers == 2


def test_unknown_score_name_raises_clear_error():
    with pytest.raises(ValueError, match="missing score"):
        BestScoreModelSelectionRule(score_name="not_present").select(_records())


def test_empty_records_raise_clear_error():
    with pytest.raises(ValueError, match="at least one"):
        KeepLastModelSelectionRule().select([])
    with pytest.raises(ValueError, match="at least one"):
        BestScoreModelSelectionRule().select([])


def test_factory_supports_keep_last_and_best_score():
    assert isinstance(model_selection_rule_from_config("keep_last"), KeepLastModelSelectionRule)
    best = model_selection_rule_from_config("best_score", score_name="custom_score")
    assert isinstance(best, BestScoreModelSelectionRule)
    assert best.score_name == "custom_score"


def test_factory_supports_keep_best_rmse_alias():
    best = model_selection_rule_from_config("keep_best_rmse")

    assert isinstance(best, BestScoreModelSelectionRule)
    assert best.score_name == "residual_rmse"
    assert best.rule_name == "keep_best_rmse"
    assert best.select(_records()).rule_name == "keep_best_rmse"
    assert best.select(_records()).retained_iteration == 1


def test_factory_supports_keep_best_rmax_alias():
    best = model_selection_rule_from_config("keep_best_rmax")

    assert isinstance(best, BestScoreModelSelectionRule)
    assert best.score_name == "residual_max"
    assert best.rule_name == "keep_best_rmax"
    assert best.select(_records()).rule_name == "keep_best_rmax"
    assert best.select(_records()).retained_iteration == 2


def test_factory_unknown_rule_raises():
    with pytest.raises(ValueError, match="Unsupported model selection rule"):
        model_selection_rule_from_config("unknown")


def test_result_and_rules_are_frozen_dataclasses():
    result = KeepLastModelSelectionRule().select(_records())
    with pytest.raises(Exception):
        result.retained_iteration = 99

    keep = KeepLastModelSelectionRule()
    with pytest.raises(Exception):
        keep.rule_name = "other"

    best = BestScoreModelSelectionRule()
    with pytest.raises(Exception):
        best.score_name = "other"
