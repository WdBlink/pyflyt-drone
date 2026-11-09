"""Measure control signs on the unmodified PyFlyt fixedwing (development only)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import numpy as np
from PyFlyt.core import Aviary
from PyFlyt.core.drones import Fixedwing


def probe() -> dict:
    records = []
    for label, surface, amplitude in [
        ("neutral", 2, 0.0),
        ("left_aileron", 0, 0.1),
        ("horizontal_tail", 2, 0.1),
        ("vertical_tail", 3, 0.1),
    ]:
        world = Aviary(
            start_pos=np.array([[0.0, 0.0, 80.0]]),
            start_orn=np.zeros((1, 3)),
            drone_type="fixedwing",
            np_random=np.random.default_rng(0),
        )
        try:
            world.set_mode(-1)
            command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.4])
            command[surface] = amplitude
            initial = world.state(0).copy()
            for _ in range(24):  # 0.2 s at 120 control Hz / 240 physics Hz
                world.set_setpoint(0, command)
                world.step()
            records.append({
                "label": label,
                "command": command.tolist(),
                "initial_state": initial.tolist(),
                "final_state": world.state(0).tolist(),
                "simulation_seconds": world.elapsed_time,
                "physics_steps": world.physics_steps,
            })
        finally:
            world.disconnect()
    source = Path(inspect.getfile(Fixedwing))
    return {
        "role": "development_control_sign_probe_not_qualification",
        "versions": {p: importlib.metadata.version(p) for p in ("PyFlyt", "pybullet", "numpy")},
        "fixedwing_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "state_rows": ["body_angular_velocity_rad_s", "euler_rad", "body_velocity_m_s", "world_position_m"],
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; choose a new version")
    result = probe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for record in result["records"]:
        print(record["label"], "body rates:", record["final_state"][0])
