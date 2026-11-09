"""One numeric telemetry row, no aircraft, model or real-time loop."""
import gc
import hashlib
import json
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[1]/'source'
sys.path.insert(0,str(SOURCE))
import timed_profile as module


def test_completed_metadata_keeps_cross_file_hashes_and_restores_gc(tmp_path):
    enabled=gc.isenabled();profile=module.TimedClockProfile(1)
    try:
        assert gc.isenabled() is False
        profile.anchor(profile.clock.monotonic());profile.initial_publication_end()
        profile.begin(0.);profile.requested_sleep(0.);profile.mark('wake');profile.rpc(0)
        profile.mark('rpc_end');profile.mark('step_end');profile.mark('progress_end')
        profile.lag(0.,False)
        for stage in ('stats_end','pub_end','log_end'):profile.mark(stage)
        profile.end(1/120)
    finally:profile.stop()
    assert gc.isenabled() is enabled and not profile.window.audit['errors']
    value=profile.save(tmp_path);clock=json.loads((tmp_path/'clock-profile.json').read_text())
    usage=json.loads((tmp_path/'clock-rusage-profile.json').read_text())
    assert clock['runtime_policy']==usage['runtime_policy']==value['runtime_policy']
    assert clock['gc_configuration_at_start']['enabled'] is False
    assert clock['recorder_changes_gc_configuration'] is True
    assert usage['base_clock_profile_sha256']==hashlib.sha256((tmp_path/'clock-profile.json').read_bytes()).hexdigest()
    assert value['rusage_profile']['sha256']==hashlib.sha256((tmp_path/'clock-rusage-profile.json').read_bytes()).hexdigest()


def test_constructor_error_removes_the_base_gc_callback(monkeypatch):
    def fail():raise RuntimeError('intentional native constructor failure')
    monkeypatch.setattr(module,'PhysicsRuntimeWindow',fail)
    before=list(gc.callbacks)
    try:module.TimedClockProfile(1)
    except RuntimeError as error:assert 'intentional' in str(error)
    else:raise AssertionError('constructor must reject failed native admission')
    assert list(gc.callbacks)==before


def test_post_start_metadata_error_rolls_back_the_entered_window(monkeypatch):
    from physics_runtime import PhysicsRuntimeWindow
    from test_runtime_window import Native,Collector
    native,collector=Native(),Collector();window=PhysicsRuntimeWindow(native=native,collector=collector)
    monkeypatch.setattr(module,'PhysicsRuntimeWindow',lambda:window)
    original=gc.get_count;before=list(gc.callbacks)
    def fail_after_start():
        if window.restore_needed and not window.closed:raise RuntimeError('post-start metadata fault')
        return original()
    monkeypatch.setattr(gc,'get_count',fail_after_start)
    try:module.TimedClockProfile(1)
    except RuntimeError as error:assert str(error)=='post-start metadata fault'
    else:raise AssertionError('metadata fault must reject construction')
    assert window.closed and not native.rt and collector.enabled
    assert list(gc.callbacks)==before and not window.audit['errors']
