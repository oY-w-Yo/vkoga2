import numpy as np
import pytest

from vkoga2.point_selection import (
    BetaGreedyPointSelectionRule,
    PointSelectionResult,
    point_selection_rule_from_config,
)


def test_p_greedy_selects_largest_power_unselected_candidate():
    rule = point_selection_rule_from_config("p_greedy")
    result = rule.select(
        abs_residual=np.array([10.0, 1.0, 2.0]),
        power=np.array([1.0, 5.0, 3.0]),
        selected_mask=np.array([False, False, False]),
    )

    assert result.candidate_index == 1
    assert result.score == 5.0
    assert result.log_score == np.log(5.0)
    assert result.rule_name == "p_greedy"
    assert result.beta == 0.0


def test_f_greedy_selects_largest_abs_residual_unselected_candidate():
    rule = point_selection_rule_from_config("f_greedy")
    result = rule.select(
        abs_residual=np.array([10.0, 1.0, 2.0]),
        power=np.array([1.0, 100.0, 3.0]),
        selected_mask=np.array([False, False, False]),
    )

    assert result.candidate_index == 0
    assert result.score == 10.0
    assert result.log_score == np.log(10.0)
    assert result.rule_name == "f_greedy"
    assert result.beta == 1.0


def test_f_times_p_greedy_selects_largest_equivalent_residual_power_product():
    rule = point_selection_rule_from_config("f_times_p_greedy")
    abs_residual = np.array([4.0, 9.0, 5.0])
    power = np.array([9.0, 1.0, 4.0])
    result = rule.select(abs_residual, power, np.array([False, False, False]))

    # beta=0.5 uses sqrt(abs_residual * power), which has the same ordering
    # as abs_residual * power.
    assert result.candidate_index == 0
    assert np.isclose(result.score, np.sqrt(36.0))
    assert np.isclose(result.log_score, 0.5 * np.log(4.0) + 0.5 * np.log(9.0))
    assert result.rule_name == "f_times_p_greedy"
    assert result.beta == 0.5


def test_f_over_p_greedy_selects_largest_residual_over_power():
    rule = point_selection_rule_from_config("f_over_p_greedy")
    result = rule.select(
        abs_residual=np.array([4.0, 9.0, 5.0]),
        power=np.array([2.0, 10.0, 1.0]),
        selected_mask=np.array([False, False, False]),
    )

    assert result.candidate_index == 2
    assert result.score == 5.0
    assert result.log_score == np.log(5.0) - np.log(1.0)
    assert result.rule_name == "f_over_p_greedy"
    assert np.isinf(result.beta)


def test_custom_beta_greedy_uses_formula():
    beta = 0.25
    rule = point_selection_rule_from_config("beta_greedy", beta=beta)
    abs_residual = np.array([16.0, 9.0, 1.0])
    power = np.array([1.0, 4.0, 100.0])
    selected_mask = np.array([False, False, False])

    expected_scores = abs_residual**beta * power ** (1.0 - beta)
    result = rule.select(abs_residual, power, selected_mask)

    assert result.candidate_index == int(np.argmax(expected_scores))
    assert np.isclose(result.score, expected_scores[result.candidate_index])
    assert np.isclose(
        result.log_score,
        beta * np.log(abs_residual[result.candidate_index])
        + (1.0 - beta) * np.log(power[result.candidate_index]),
    )
    assert result.rule_name == "beta_greedy"
    assert result.beta == beta


def test_selected_mask_excludes_selected_highest_score_candidate():
    rule = point_selection_rule_from_config("f_greedy")
    result = rule.select(
        abs_residual=np.array([100.0, 10.0, 1.0]),
        power=np.array([1.0, 1.0, 1.0]),
        selected_mask=np.array([True, False, False]),
    )

    assert result.candidate_index == 1


def test_tie_selects_smallest_index():
    rule = point_selection_rule_from_config("p_greedy")
    result = rule.select(
        abs_residual=np.array([1.0, 2.0, 3.0]),
        power=np.array([5.0, 5.0, 5.0]),
        selected_mask=np.array([False, False, False]),
    )

    assert result.candidate_index == 0


def test_p_greedy_log_score_ignores_zero_residual_without_nan():
    rule = point_selection_rule_from_config("p_greedy")
    result = rule.select(
        abs_residual=np.array([0.0, 0.0, 0.0]),
        power=np.array([1.0, 5.0, 3.0]),
        selected_mask=np.array([False, False, False]),
    )

    assert result.candidate_index == 1
    assert result.score == 5.0
    assert result.log_score == np.log(5.0)


def test_f_greedy_log_score_ignores_zero_power_without_nan():
    rule = point_selection_rule_from_config("f_greedy")
    result = rule.select(
        abs_residual=np.array([1.0, 5.0, 3.0]),
        power=np.array([0.0, 0.0, 0.0]),
        selected_mask=np.array([False, False, False]),
    )

    assert result.candidate_index == 1
    assert result.score == 5.0
    assert result.log_score == np.log(5.0)


def test_zero_scores_are_usable_and_tie_to_smallest_available_index():
    rule = point_selection_rule_from_config("f_greedy")
    result = rule.select(
        abs_residual=np.array([0.0, 0.0, 0.0]),
        power=np.array([1.0, 2.0, 3.0]),
        selected_mask=np.array([True, False, False]),
    )

    assert result.candidate_index == 1
    assert result.score == 1e-300
    assert result.log_score == np.log(1e-300)


def test_extreme_scale_selection_uses_log_score_for_comparison():
    rule = point_selection_rule_from_config("beta_greedy", beta=2.0)
    abs_residual = np.array([1e150, 1e200])
    power = np.array([1.0, 1.0])
    selected_mask = np.array([False, False])

    result = rule.select(abs_residual, power, selected_mask)

    assert result.candidate_index == 1
    assert np.isinf(result.score)
    assert np.isfinite(result.log_score)
    assert np.isclose(result.log_score, 2.0 * np.log(1e200) - np.log(1.0))


def test_shape_mismatch_raises():
    rule = point_selection_rule_from_config("p_greedy")
    with pytest.raises(ValueError, match="same shape"):
        rule.select(
            abs_residual=np.array([1.0, 2.0]),
            power=np.array([1.0, 2.0, 3.0]),
            selected_mask=np.array([False, False]),
        )


def test_negative_power_raises():
    rule = point_selection_rule_from_config("p_greedy")
    with pytest.raises(ValueError, match="power must be non-negative"):
        rule.select(
            abs_residual=np.array([1.0, 2.0]),
            power=np.array([1.0, -1.0]),
            selected_mask=np.array([False, False]),
        )


@pytest.mark.parametrize(
    "abs_residual,power",
    [
        (np.array([np.nan, 1.0]), np.array([1.0, 2.0])),
        (np.array([1.0, 2.0]), np.array([np.inf, 2.0])),
    ],
)
def test_nan_or_inf_inputs_raise(abs_residual, power):
    rule = point_selection_rule_from_config("p_greedy")
    with pytest.raises(ValueError, match="finite"):
        rule.select(abs_residual, power, np.array([False, False]))


@pytest.mark.parametrize("beta", [-1.0, np.nan, -np.inf])
def test_invalid_beta_raises(beta):
    with pytest.raises(ValueError, match="beta"):
        BetaGreedyPointSelectionRule(beta=beta)


def test_selected_mask_must_be_bool():
    rule = point_selection_rule_from_config("p_greedy")
    with pytest.raises(ValueError, match="bool"):
        rule.select(
            abs_residual=np.array([1.0, 2.0]),
            power=np.array([1.0, 2.0]),
            selected_mask=np.array([0, 1]),
        )


def test_all_selected_raises():
    rule = point_selection_rule_from_config("p_greedy")
    with pytest.raises(ValueError, match="all candidates are already selected"):
        rule.select(
            abs_residual=np.array([1.0, 2.0]),
            power=np.array([1.0, 2.0]),
            selected_mask=np.array([True, True]),
        )


def test_factory_named_rules_return_correct_beta():
    assert point_selection_rule_from_config("p_greedy").beta == 0.0
    assert point_selection_rule_from_config("f_times_p_greedy").beta == 0.5
    assert point_selection_rule_from_config("f_greedy").beta == 1.0
    assert np.isinf(point_selection_rule_from_config("f_over_p_greedy").beta)
    assert point_selection_rule_from_config("beta_greedy", beta=0.25).beta == 0.25


def test_factory_beta_greedy_requires_beta():
    with pytest.raises(ValueError, match="requires beta"):
        point_selection_rule_from_config("beta_greedy")


def test_factory_unknown_rule_raises():
    with pytest.raises(ValueError, match="Unsupported selection rule"):
        point_selection_rule_from_config("fp_greedy")


def test_rule_and_result_are_frozen_dataclasses():
    rule = point_selection_rule_from_config("p_greedy")
    with pytest.raises(Exception):
        rule.beta = 1.0

    result = rule.select(
        abs_residual=np.array([1.0, 2.0]),
        power=np.array([1.0, 2.0]),
        selected_mask=np.array([False, False]),
    )
    assert isinstance(result, PointSelectionResult)
    assert hasattr(result, "score")
    assert hasattr(result, "log_score")
    with pytest.raises(Exception):
        result.candidate_index = 0
