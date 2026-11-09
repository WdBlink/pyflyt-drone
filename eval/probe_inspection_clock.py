"""Check that a blocked coordinator does not pause four-aircraft physics.

Development probe with a sleep surrogate, not measured model/API latency.
The worker owns PyBullet; the coordinator sees only a locked progress snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.qualify_inspection_control import fly


def run_world(shared, ready, output, duration):
    def publish(world):
        with shared.get_lock():
            shared[:] = [world.elapsed_time, *world.state(0)[3].tolist()]
        ready.set()

    try:
        # Trigger the actual aero kernels before defining the mission epoch.
        # Keep this separate flight as initialization evidence, then create a
        # fresh seeded world. No mission-time epoch adjustment or pause occurs.
        warmup = fly(1.0, 0, "orbit", 4)
        if not warmup["envelope_pass"]:
            raise RuntimeError("development warmup flight failed")
        result = fly(duration, 0, "orbit", 4, realtime=True, on_step=publish)
        result["initialization_warmup"] = warmup
    except Exception:
        result = {"envelope_pass": False, "operational_error": traceback.format_exc()}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def probe(duration, wait_seconds, output):
    context = mp.get_context("spawn")
    shared = context.Array("d", [0.0] * 4)
    ready = context.Event()
    process = context.Process(target=run_world, args=(shared, ready, output, duration))
    process.start()
    try:
        if not ready.wait(30):
            raise RuntimeError("physics worker did not become ready")
        with shared.get_lock():
            before = list(shared[:])
        wait_started = time.monotonic()
        time.sleep(wait_seconds)  # intentionally block the coordinator only
        actual_wait = time.monotonic() - wait_started
        with shared.get_lock():
            after = list(shared[:])
        process.join(timeout=duration + 15)
        if process.is_alive() or process.exitcode != 0:
            raise RuntimeError(f"physics worker failed or timed out: exitcode={process.exitcode}")
        world_result = json.loads(output.read_text())
        simulation_advance = after[0] - before[0]
        displacement = sum((a-b)**2 for a, b in zip(after[1:], before[1:]))**0.5
        return {
            "role": "development_wallclock_isolation_probe_not_model_latency_or_G3_qualification",
            "wait_type": "blocking_sleep_surrogate_no_model_call",
            "requested_wait_seconds": wait_seconds, "actual_wait_seconds": actual_wait,
            "before_sim_time_position": before, "after_sim_time_position": after,
            "sim_advance_during_wait": simulation_advance, "displacement_during_wait_m": displacement,
            "development_tolerance_seconds": 0.1,
            "passed": (world_result["envelope_pass"] and displacement > 1
                       and abs(simulation_advance - actual_wait) <= 0.1
                       and world_result["clock_lag_seconds"]["max"] <= 0.1),
            "worker_result": output.name,
            "worker_result_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--wait", type=float, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    worker_output = args.out.with_name(args.out.stem + ".world.json")
    if not all(math.isfinite(x) for x in (args.wait, args.duration)) or not (0 < args.wait < args.duration - 2) or args.out.exists() or worker_output.exists():
        parser.error("choose a valid wait/duration and new output paths")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = probe(args.duration, args.wait, worker_output)
    except Exception:
        result = {"role": "development_operational_failure", "passed": False,
                  "operational_error": traceback.format_exc()}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
