"""Independent expected counts for raw misses versus association suppression."""

from copy import deepcopy

import pytest

from eval.qualify_inspection_reference import recount_sensor_opportunities


def frame_fixture():
    return {
        "sensor_audit": [
            dict(evidence_id="target", t=.5, drone_id=0, entity_key="x", mode="survey"),
            dict(evidence_id="false", t=1., drone_id=1, entity_key=None, mode="survey")],
        "sensor_opportunities": [
            dict(t=.5, drone_id=0, entity_key="x", mode="survey", missed=False, emitted_evidence_id="target"),
            dict(t=1., drone_id=0, entity_key="x", mode="inspect", missed=True, emitted_evidence_id=None),
            dict(t=1., drone_id=0, entity_key="y", mode="inspect", missed=False, emitted_evidence_id=None)],
        "background_trials": [dict(t=.5, drone_id=0, triggered=True, emitted_evidence_id=None),
                              dict(t=1., drone_id=1, triggered=True, emitted_evidence_id="false"),
                              dict(t=1.5, drone_id=1, triggered=False, emitted_evidence_id=None)],
    }


def test_raw_misses_association_suppression_and_background_have_distinct_denominators():
    result = recount_sensor_opportunities(frame_fixture())
    assert result["sensor_modes"]["survey"]["raw_miss_fraction"] == 0.
    assert result["sensor_modes"]["inspect"] == dict(visible_opportunities=2, raw_detection_misses=1,
        emitted_measurements=0, association_suppressed=1, raw_miss_fraction=.5)
    assert result["background"] == dict(wide_frame_trials=3, raw_triggers=2,
        emitted_measurements=1, raw_trigger_fraction=2/3)
    assert result["emitted_evidence_reconciled"] == 2


@pytest.mark.parametrize("fault", ["missed_emission", "orphan", "duplicate", "wrong_frame"])
def test_structurally_valid_sensor_rows_cannot_hide_inconsistent_acquisition(fault):
    world = deepcopy(frame_fixture())
    if fault == "missed_emission":
        world["sensor_opportunities"][0]["missed"] = True
    elif fault == "orphan":
        world["background_trials"][1]["emitted_evidence_id"] = None
    elif fault == "duplicate":
        world["sensor_opportunities"].append(dict(world["sensor_opportunities"][0]))
    else:
        world["sensor_audit"][0]["t"] = 1.5
    with pytest.raises(ValueError):
        recount_sensor_opportunities(world)
