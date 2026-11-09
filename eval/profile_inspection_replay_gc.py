"""Locate synchronous Python GC pauses in a saved development execution replay.

Diagnostic only: no model, new policy sample, real-time admission or speedup
claim. Run between live flights. Original inputs stay immutable. The saved
LLM manifest lacks the classic replayer's layout/seed keys; their exact derived
values are recorded, not presented as original worker fields.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile(directory, out):
    from eval.inspection_population import load_development_case
    from eval.inspection_scenarios import ScenarioMission
    from eval.replay_inspection_reference import replay

    directory, out = Path(directory), Path(out)
    if out.exists():
        raise ValueError("preserve previous diagnostics; choose a fresh output")
    original = {name: sha(directory/name) for name in ("run.json", "world.json", "private-case.json")}
    run = json.loads((directory/"run.json").read_text())
    case = load_development_case(json.loads((directory/"private-case.json").read_text()))
    if run["private_case_sha256"] != original["private-case.json"] or run["world_sha256"] != original["world.json"]:
        raise ValueError("original private/world byte binding differs")
    additions = {"layout": "private_frozen_dev_case", "seed": case.seed}
    if any(k in run and run[k] != v for k, v in additions.items()):
        raise ValueError("never override a conflicting original manifest field")
    derived = {**run, **additions}
    active = {"in_step": False, "step_start": None}
    starts, events = {}, []
    maximum = {"seconds": 0.0}
    steps = 0
    old_step = ScenarioMission.step

    def step(self, *args, **kwargs):
        nonlocal steps
        before, stamp = time.perf_counter(), self.t
        active.update(in_step=True, step_start=stamp)
        try:
            return old_step(self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter()-before
            if elapsed > maximum["seconds"]:
                maximum.update(seconds=elapsed, sim_time=stamp, started_at=before,
                               finished_at=before+elapsed)
            steps += 1
            active["in_step"] = False

    def collection(phase, info):
        gen, stamp = info["generation"], time.perf_counter()
        if phase == "start":
            starts[gen] = (stamp, active["in_step"], active["step_start"] if active["in_step"] else None)
        elif gen in starts:
            before, in_step, sim_time = starts.pop(gen)
            events.append({"generation": gen, "seconds": stamp-before, "started_at": before,
                           "finished_at": stamp, "in_step": in_step, "sim_time": sim_time,
                           "collected": info["collected"], "uncollectable": info["uncollectable"]})

    # The original saved JSON is a second copy of the mission ledger. Exclude
    # that pre-existing diagnostic heap from traversal; only newly allocated
    # replay objects remain collectable. This differs from the live worker and
    # is explicitly a profiling limitation, not a production configuration.
    gc.collect()
    gc.freeze()
    thresholds = gc.get_threshold()
    was_enabled = gc.isenabled()
    if not was_enabled:
        raise ValueError("baseline profiling requires default automatic GC enabled")
    ScenarioMission.step = step
    gc.callbacks.append(collection)
    try:
        with tempfile.TemporaryDirectory(prefix="inspection-gc-replay-") as temp:
            root = Path(temp)
            (root/"run.json").write_text(json.dumps(derived, indent=2)+"\n")
            for name in ("world.json", "private-case.json"):
                shutil.copyfile(directory/name, root/name)
            result = replay(root)
    finally:
        gc.callbacks.remove(collection)
        ScenarioMission.step = old_step
        gc.unfreeze()
    if any(sha(directory/name) != expected for name, expected in original.items()):
        raise ValueError("original diagnostic input changed")
    record = {"role": "instrumented non-realtime execution replay; not clock qualification or another policy sample",
              "original": {"directory": str(directory.resolve()), "files": original},
              "derived_run_additions": additions, "replay": result, "steps_timed": steps,
              "gc_thresholds": thresholds, "gc_enabled": was_enabled, "gc_events": events,
              "maximum_step": maximum, "profile_script_sha256": sha(__file__),
              "limits": "One profiled replay, frozen pre-existing input heap, added timing callbacks; cannot estimate live1x/model contention, establish past-pause attribution or choose a faster configuration."}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+"\n")
    if not result["passed"]:
        raise ValueError("instrumented replay differs; retain diagnostic but do not use its timing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    profile(args.run, args.out)
