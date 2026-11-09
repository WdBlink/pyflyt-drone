"""Freeze the two valid callbacks before the independent slot030 unknown."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

project = Path.cwd()
run = project / "paper/research/20261104-inspection"
sdk = project / "logs/inspection-eight-condition-pilot.v4.continuation.v1"
out = run / "qualification/pilot-v4-slots-twentynine-thirty.v1"
assert not out.exists()
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
refs = {}


def bind(p):
    refs[str(p.relative_to(project))] = {"sha256": sha(p), "bytes": p.stat().st_size}


slots = read(sdk / "episode-order.json")["slots"][28:30]
assert [(s["order_index"], s["condition"], s["repeat_ordinal"]) for s in slots] == [(28, "B3-M2", 0), (29, "B2", 0)]
rows = []
for slot in slots:
    classic = slot["condition"] == "B2"
    root = project / "logs/inspection-eight-condition-dev-flights.v4"
    base = (root / "classic" if classic else root) / f"dev-L3-01-{slot['condition']}-0"
    attempt = base / "attempt-00"
    terminal = read(attempt / "terminal.json")
    assert terminal["status"] == "completed" and terminal["returncode"] == 0 and not (base / "attempt-01").exists()
    world_path = attempt / "flight/world.json"
    assert world_path.stat().st_size < 512 * 1024 * 1024
    world = read(world_path)
    validation = terminal["evidence"] if classic else terminal["physical_validation"]
    assert sha(world_path) == validation["world_sha256"]
    assert world["status"] == "completed" and world["simulated_seconds"] == 600 and world["physics_steps_executed"] == 144000
    assert world["clock_lag_over_0p1_seconds_ticks"] == 0 and world["max_clock_lag_seconds"] < .1 and world["physics_runtime"]["errors"] == []
    recount = read(Path("/tmp/inspection-v4-order29-classic-frozen-recount.v1.json")) if classic else terminal["evidence"]["frozen_score_recount"]
    assert recount["score_equal"] is True and recount["score"] == world["score"] and recount["world_sha256"] == sha(world_path)
    assert world["score"]["Y"] == terminal["Y"]
    callback_path = sdk / f"order-receipts/{slot['order_index']:06d}.json"
    callback = read(callback_path)
    assert all(callback[k] == slot[k] for k in ("order_index", "case_id", "condition", "repeat_ordinal"))
    raw = read(sdk / callback["raw"])
    assert raw["status"] == "completed" and raw["value"] == terminal["Y"] and raw["evidence"]["cache_reused"] is False
    ids = {r["response"]["report_id"] for r in world["reports"] if r["response"].get("status") == "accepted"}
    progress = json.loads((attempt / "flight/progress.jsonl").read_text().splitlines()[-1])
    assert progress["t"] == 600 and progress["reports"] == len(ids) and all(0 <= r["t"] <= 600 for r in world["reports"])
    for p in [base / "plan.json", attempt / "terminal.json", world_path, attempt / "flight/run.json", attempt / "flight/private-case.json", callback_path, sdk / callback["raw"]]:
        bind(p)
    for path, digest in terminal.get("settlement_file_hashes", {}).items():
        p = Path(path)
        assert sha(p) == digest
        bind(p)
    cleanup_path = sdk / "invocations/029.json" if classic else attempt / "process-tree.json"
    cleanup_value = read(cleanup_path)
    cleanup = cleanup_value["cleanup"] if classic else cleanup_value
    assert cleanup["status"] == "closed" and cleanup["remaining_owned"] == [] and cleanup["unclaimed_new_python_orphans"] == []
    bind(cleanup_path)
    rows.append({**slot, "attempts": 1, "score": {k: world["score"][k] for k in ("C", "S", "D", "Q", "H", "E", "Y", "SR")},
                 "clock": {k: world[k] for k in ("simulated_seconds", "physics_steps_executed", "max_clock_lag_seconds", "clock_lag_p95_seconds", "clock_lag_over_0p1_seconds_ticks")},
                 "report_tool_audit_entries_count": len(world["reports"]), "accepted_unique_report_ids_count": len(ids),
                 "usage": None if classic else terminal["evidence"]["model_meter"]["usage"]})
sdk_rows = [read(p) for kind in ("classic", "llm") for p in (sdk / kind / "raw").glob("*.json")]
counts = dict(Counter(r["status"] for r in sdk_rows))
assert counts == {"completed": 30, "failed": 1, "stopped": 77} and sum(r["attempted"] for r in sdk_rows) == 31
assert not list((run / "resource-requests").glob("dev-all8-pilot-v4-031*"))
out.mkdir()
captures = {}
for p, name in [(Path(__file__), "capture-source.py"), (Path("/tmp/inspection-v4-order29-classic-frozen-recount.v1.json"), "classic-order29-frozen-recount.json"),
                (Path("/tmp/inspection-v4-second-cleanup-native-status.v1.json"), "native-status.json"),
                (Path("/tmp/inspection-pilot-v4-continuation.v1.stdout"), "aborted-supervisor.stdout")]:
    q = out / name
    shutil.copyfile(p, q)
    captures[str(q.relative_to(run))] = sha(q)
for path, value in refs.items():
    assert sha(project / path) == value["sha256"]
result = {"role": "valid original callbacks29–30 only; following slot030 still unknown, no new barrier acknowledgement", "at": datetime.now(timezone.utc).isoformat(),
          "completed_callbacks": 30, "planned_callbacks": 108, "aborted_sdk_counts": counts, "attempted": 31, "slots": rows,
          "bound_files": refs, "immutable_captures": captures, "limits": "one completeL1 and partialL3; not formal/G4; ordinary clock summaries; original030unknown retained"}
(out / "binding.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"binding": str(out / "binding.json"), "sha256": sha(out / "binding.json"), "slots": rows}, allow_nan=False))
