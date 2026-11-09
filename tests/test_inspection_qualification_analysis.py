"""Hand-made artifact fixtures for admission guards, not scientific evidence."""

import hashlib
import json
import pytest

from eval.analyze_inspection_qualification import analyze


def make_run(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path/"source.py"
    source.write_text("# controlled artifact fixture\n")
    run_dir = tmp_path/"run"; run_dir.mkdir()
    world = {"status": "completed", "clock_mode": "wallclock_1x", "simulated_seconds": 600,
             "max_clock_lag_seconds": .02, "clock_lag_p95_seconds": .01,
             "score": {"C": .9, "S": .8, "D": .1, "Q": .6, "H": 1, "E": 1, "Y": .6,
                       "SR": 0, "static_recall": 1},
             "delivered_observations": [{"evidence_id": "a", "covariance_xy": [1, 1]}],
             "sensor_audit": [{"evidence_id": "a", "entity_key": "t", "t": 1, "drone_id": 0,
                               "mode": "inspect", "truth_xy": [0, 0], "estimate_xy": [1, 1],
                               "predicted_class": "risk", "true_class": "risk"},
                              {"entity_key": None}]}
    world_path = run_dir/"world.json"; world_path.write_text(json.dumps(world))
    run = {"status": "completed", "sources_unchanged": True,
           "world_sha256": hashlib.sha256(world_path.read_bytes()).hexdigest(),
           "sources": {str(source): source.read_text()},
           "source_hashes": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
           "events": False, "coordinator_block_probe": {"displacement_m": 80, "actual_wait_seconds": 5.01,
                                                         "sim_advance_seconds": 5.0}}
    (run_dir/"run.json").write_text(json.dumps(run))
    contract = tmp_path/"contract.json"
    contract.write_text(json.dumps({"G3": {"reference_minimum_Y": .2, "reference_minimum_static_recall": .5,
                                          "maximum_clock_lag_seconds": .1, "maximum_wait_advance_error_seconds": .1,
                                          "normal_sensor_inspect_position_rmse_max_m": 3,
                                          "normal_sensor_confirmed_class_accuracy_min": .9}}))
    return run_dir, contract, source


def test_valid_fixture_reports_counts_rmse_and_separate_reference_scope(tmp_path):
    run, contract, _ = make_run(tmp_path)
    result = analyze([run], contract)
    assert result["reference_and_clock_evidence_pass"]
    assert result["sensor_point_thresholds_pass"]
    assert result["sensor_groups"]["inspect_normal"]["measurements"] == 1
    assert result["sensor_groups"]["inspect_normal"]["position_vector_rmse_m"] == 2**.5
    assert result["attempts"][0]["score"]["SR"] == 0
    assert result["limitations"]


@pytest.mark.parametrize("variant", ["source", "world_bytes"])
def test_changed_source_or_tampered_world_cannot_supply_qualification(tmp_path, variant):
    run, contract, source = make_run(tmp_path)
    if variant == "source":
        source.write_text("# changed\n")
    else:
        with (run/"world.json").open("a") as stream:
            stream.write("\n")
    result = analyze([run], contract)
    assert not result["attempts"][0]["admitted"]
    assert not result["reference_and_clock_evidence_pass"]


def test_operational_failure_slot_is_retained_and_blocks_admission(tmp_path):
    a, contract, _ = make_run(tmp_path/"a")
    b, _, _ = make_run(tmp_path/"b")
    path = b/"run.json"; data = json.loads(path.read_text()); data["status"] = "operational_failure"
    path.write_text(json.dumps(data))
    result = analyze([a, b], contract)
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["admitted"]
    assert not result["attempts"][1]["admitted"]
    assert not result["reference_and_clock_evidence_pass"]
    assert not result["sensor_point_thresholds_pass"]
    assert not result["sensor_point_thresholds_pass"]


def test_actual_clock_drift_blocks_clock_even_when_quality_is_high(tmp_path):
    run, contract, _ = make_run(tmp_path)
    world_path = run/"world.json"; world = json.loads(world_path.read_text())
    world["max_clock_lag_seconds"] = .2
    world_path.write_text(json.dumps(world))
    path = run/"run.json"; manifest = json.loads(path.read_text())
    manifest["world_sha256"] = hashlib.sha256(world_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    result = analyze([run], contract)
    assert result["attempts"][0]["reference_quality_pass"]
    assert not result["reference_and_clock_evidence_pass"]
def test_archive_maps_absolute_runner_into_new_root_without_writing_original(tmp_path):
    import hashlib
    from eval.inspection_artifacts import archive_source_bundle,verify_source_bundle
    original=tmp_path/"recorded"/"eval"/"run_inspection_reference.py"
    run={"sources":{str(original):"runner source","envs/inspection/a.py":"module source"}}
    run["source_hashes"]={p:hashlib.sha256(s.encode()).hexdigest() for p,s in run["sources"].items()}
    destination=tmp_path/"archive"
    archive_source_bundle(run,destination)
    assert not original.exists()
    assert (destination/"eval/run_inspection_reference.py").read_text()=="runner source"
    assert verify_source_bundle(run,destination)[str(original)]==destination/"eval/run_inspection_reference.py"

def test_trace_threshold_is_explicit_and_legacy_data_stays_under6m():
    import base64
    import hashlib
    import zlib
    import numpy as np
    import pytest
    from eval.inspection_tracking import COLUMNS,recount_tracking
    row=np.asarray([[1/240,*([7.,96.,94.,98.]*4)]],dtype='<f8')
    raw=row.tobytes()
    trace=dict(encoding='base64-zlib-little-endian-float64',columns=COLUMNS,rows=1,xy_tube_m=10.,
        raw_sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(zlib.compress(raw)).decode())
    record={'execution_tracking_trace':trace,'execution_tracking_240hz':{'physics_samples':1}}
    assert recount_tracking(record,expected_xy_tube_m=10.)['horizontal_tube_breach_steps']==0
    del trace['xy_tube_m']
    assert recount_tracking(record)['horizontal_tube_breach_steps']==1
    with pytest.raises(ValueError):recount_tracking(record,expected_xy_tube_m=10.)


def test_quality_clock_points_cannot_replace_complete_physical_reference_evidence(tmp_path):
    from eval.qualify_inspection_reference import qualify
    run,contract,_=make_run(tmp_path)
    settings=json.loads(contract.read_text())
    settings['executor_qualification']={'xy_tracking_tube_m':10.}
    contract.write_text(json.dumps(settings))
    result=qualify(run,contract)
    assert result['original_quality_clock_calibration']['reference_and_clock_evidence_pass']
    assert not result['passed'] and result['checks']['complete_saved_evidence'] is False

