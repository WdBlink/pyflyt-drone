"""Runtime lifecycle faults use a Fake mission; these are not flight results."""

import ast
import json
from pathlib import Path
import queue
from types import SimpleNamespace

import pytest

import envs.inspection.mission as missions
import eval.run_inspection_reference as runner


@pytest.mark.parametrize("fault", [None, "start", "loop", "restore"])
def test_worker_restores_runtime_and_preserves_failures(tmp_path, monkeypatch, fault):
    created, windows = [], []

    class Mission:
        def __init__(self, *args, **kwargs):
            self.t = 0.
            self.warmup = not created
            created.append(self)
            self.executor = SimpleNamespace(infrastructure_errors=[])
            self.flight = SimpleNamespace(world=SimpleNamespace(
                physics_steps=1, physics_hz=240, step_period=1/120))

        def step(self):
            if not self.warmup and fault == "loop":
                raise RuntimeError("injected control-loop failure")
            self.t += 1.

        def close(self):
            if not self.warmup:
                assert windows[0].closed

        def public_state(self, **kwargs):
            return {}

        def audit(self):
            return {"score": {"role": "software Fake"}}

        def business_audit(self):
            return {"fake_partial_audit": True}

    class Window:
        def __init__(self):
            self.closed = False
            self.audit = {"errors": []}
            windows.append(self)

        def start(self):
            assert len(created) == 2
            if fault == "start":
                raise RuntimeError("injected runtime-start failure")

        def close(self):
            self.closed = True
            if fault == "restore":
                self.audit["errors"] = ["injected restoration failure"]

    monkeypatch.setattr(missions, "InspectionMission", Mission)
    monkeypatch.setattr(runner, "PhysicsRuntimeWindow", Window)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    updates = queue.Queue()
    runner.worker(0, True, True, 1, queue.Queue(), updates, tmp_path)
    world = json.loads((tmp_path/"world.json").read_text())
    assert windows[0].closed
    assert world["physics_runtime"] == windows[0].audit
    assert world["status"] == ("completed" if fault is None else "operational_failure")
    if fault == "loop":
        assert world["partial_business_audit"] == {"fake_partial_audit": True}


def test_runtime_integration_keeps_physics_loop_unchanged():
    root = Path(__file__).resolve().parents[1]
    old = root/"paper/research/20261104-inspection/planning/physics-runtime-integration.v1/before/eval/run_inspection_reference.py"
    loop = lambda source: ast.dump(next(node for node in ast.walk(ast.parse(source))
                                      if isinstance(node, ast.While)))
    assert loop(old.read_text()) == loop((root/"eval/run_inspection_reference.py").read_text())


@pytest.mark.parametrize("fault", [None, "missing", "demoted", "gc", "boolean_tick", "restore"])
def test_new_source_receipts_require_the_runtime_audit(tmp_path, fault):
    from eval.inspection_episode_receipts import digest, validate_flight
    from tests.test_inspection_episode_receipts import inputs
    import subprocess
    plan, command, _ = inputs(tmp_path)
    plan["source_hashes"]["/fixture/eval/inspection_worker_runtime.py"] = "c"*64
    (tmp_path/"fixture-plan.json").write_text(json.dumps(plan))
    out = tmp_path/"flight"
    assert subprocess.run(command("ok")(out), check=False).returncode == 0
    world = json.loads((out/"world.json").read_text())
    audit = {"schema": "inspection-physics-runtime-window-v1", "errors": [],
        "mach_timebase": {"numer": 125, "denom": 3},
        "requested_ns": {"period": 8333333, "computation": 4000000, "constraint": 8000000},
        "requested_ticks": [200000, 96000, 192000, 1], "soft_realtime_not_a_guarantee": True,
        "before": {"values": [1], "default": False}, "restored": {"values": [1], "default": False},
        "active_start": {"values": [200000, 96000, 192000, 1], "default": False},
        "active_end": {"values": [200000, 96000, 192000, 1], "default": False},
        "rt_after_restore": {"default": True}, "gc_enabled_before": True,
        "gc_enabled_at_epoch": False, "gc_enabled_after": True, "window_wall_seconds": 600.01}
    if fault == "demoted":
        audit["active_end"]["default"] = True
    elif fault == "gc":
        audit["gc_enabled_at_epoch"] = True
    elif fault == "boolean_tick":
        audit["active_start"]["values"][-1] = True
    elif fault == "restore":
        audit["errors"] = ["restoration failed"]
    if fault != "missing":
        world["physics_runtime"] = audit
    (out/"world.json").write_text(json.dumps(world))
    run = json.loads((out/"run.json").read_text())
    run["world_sha256"] = digest(out/"world.json")
    (out/"run.json").write_text(json.dumps(run))
    value, evidence = validate_flight(out, plan, 0)
    assert (value == 0) if fault is None else value is None
    assert evidence["failure_reasons"] == ([] if fault is None else ["runtime_window_invalid"])
