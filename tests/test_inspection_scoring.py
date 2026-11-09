"""Hand-calculated adversarial fixtures, distinct from benchmark episodes."""

from dataclasses import replace
import math

import pytest

from envs.inspection.scoring import GradingMeasurement, final_static_quality, grade_dynamic_bins, localization_quality


def point(evidence_id="a", t=1.0, arrival=2.0, **changes):
    base = GradingMeasurement(evidence_id, t, arrival, (0.0, 0.0), "vehicle", True, True, True)
    return replace(base, **changes)


def grade(points, **options):
    return grade_dynamic_bins(points, lambda t: (0.0, 0.0), "vehicle", 0.0, 200.0, **options)


@pytest.mark.parametrize("error,expected", [(0, 1), (3, 1), (3.001, 2/3), (5, 2/3),
                                         (5.001, 1/3), (10, 1/3), (10.001, 0)])
def test_localization_boundaries(error, expected):
    assert localization_quality(error) == expected


def test_missing_bins_cannot_shrink_single_target_denominator():
    score, bins = grade([point()])
    assert len(bins) == 60
    assert score == pytest.approx(1 / 60)
    assert grade([])[0] == 0


def test_boundary_measurement_supports_one_bin_only():
    score, bins = grade([point(t=3, arrival=3)])
    assert score == pytest.approx(1 / 60)
    assert bins[0].evidence_id == "a"
    assert bins[1].evidence_id is None


@pytest.mark.parametrize("wrong", [{"claimed_class": "boat"}, {"identity_correct": False},
                                  {"quality_valid": False}, {"estimate_xy": (11.0, 0.0)}])
def test_latest_incorrect_does_not_fall_back_to_older_correct(wrong):
    newer = point("z", 2, 3, **wrong)
    score, bins = grade([point(), newer])
    assert score == 0
    assert bins[0].evidence_id == "z"


def test_latest_late_or_causally_invalid_is_removed_before_selection():
    late = point("z", 2, 9.001, claimed_class="boat")
    invalid = point("q", 2.5, 3, provenance_valid=False)
    score, bins = grade([point(), late, invalid])
    assert score == pytest.approx(1 / 60)
    assert bins[0].evidence_id == "a"
    assert grade([point(arrival=9)])[0] == pytest.approx(1/60)


def test_observed_minus_delivered_can_be_negative():
    points = [point(), point("late", 2, 10, claimed_class="boat")]
    assert grade(points, track="delivered")[0] == pytest.approx(1/60)
    assert grade(points, track="observed")[0] == 0


def test_error_uses_measurement_time_not_bin_endpoint():
    score, bins = grade_dynamic_bins([point(t=1, arrival=2, estimate_xy=(10, 0))],
                                     lambda t: (10*t, 0), "vehicle", 0, 200)
    assert score == pytest.approx(1/60)
    assert bins[0].quality == 1


def test_weight_update_inside_bin_applies_at_next_bin_start():
    score, bins = grade([point(), point("b", 4, 5)],
                        weight_at=lambda t: 3 if t >= 1.5 else 1)
    assert bins[0].weight == 1
    assert bins[1].weight == 3
    assert score == pytest.approx(4 / (1 + 59*3))


def test_tie_break_is_evidence_id_not_input_order_or_correctness():
    points = [point("z", 1, 2), point("a", 1, 2, claimed_class="boat")]
    assert grade(points)[0] == grade(points[::-1])[0] == 0
    assert grade(points)[1][0].evidence_id == "a"


def test_static_last_legal_version_can_reduce_score():
    points = [point(), point("z", 1, 4, claimed_class="boat")]
    assert final_static_quality(points, (0, 0), "vehicle", 3) == 1
    assert final_static_quality(points, (0, 0), "vehicle", 4) == 0
    assert final_static_quality([point(provenance_valid=False)], (0, 0), "vehicle", 4) == 0


def test_illegal_or_truncated_protocol_inputs_fail_instead_of_scoring_zero():
    with pytest.raises(ValueError, match="full observation window"):
        grade_dynamic_bins([], lambda t: (0, 0), "vehicle", 0, 180)
    with pytest.raises(ValueError, match="unique"):
        grade([point(), point()])
    with pytest.raises(ValueError):
        point(arrival=0)
    with pytest.raises(ValueError):
        localization_quality(math.nan)
    with pytest.raises(ValueError):
        grade([], weight_at=lambda t: 0)
