"""Freeze one complete development root; no population inference or new effect."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

project = Path.cwd()
run = project / "paper/research/20261104-inspection"
sdk = project / "logs/inspection-eight-condition-pilot.v4.continuation.v2"
out = run / "qualification/pilot-v4-second-root.v1"
source = run / "recovery/pilot-v4-cleanup-030.v1/source/continue_pilot.py"
assert hashlib.sha256(source.read_bytes()).hexdigest() == "4c78d5b130f27f15f34483f7f88bf5a7c348cef2399d272faae3c63ba3502a8f"
spec = importlib.util.spec_from_file_location("_second_root_receipt_reader", source)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
read, sha = helper.read, helper.digest
assert not out.exists()
pause = json.loads((sdk / "checkpoint-pauses.jsonl").read_text().splitlines()[-1])
assert pause["order_index"] == 36 and (sdk / "checkpoint.request").is_file()
slots = read(sdk / "episode-order.json")["slots"][18:36]
assert len(slots) == 18 and all(s["case_id"] == "dev-L3-01" and s["level"] == 3 for s in slots)
assert Counter(s["condition"] for s in slots) == {"B0": 1, "B1": 1, "B2": 1, "B3-M1": 3, "B3-M2": 3, "B3-M3": 3, "B4-M1": 3, "B4-M2": 3}
reconciled = {24: helper.older().verify(helper.OLDER_ACCEPTANCE), 30: helper.verify(helper.BUNDLE / "acceptance.v1.json")}
refs, rows, observations = {}, [], []


def bind(path, expected=None):
    ref = helper.checked(path, expected)
    refs[ref["path"]] = {"sha256": ref["sha256"], "bytes": path.stat().st_size}


for ref in helper.live_frozen_sources():
    bind(project / ref["path"], ref["sha256"])
for path in [source, helper.OBSERVER, helper.OBSERVER_REVIEW, helper.OLDER_SOURCE, helper.OLDER_ACCEPTANCE, helper.BUNDLE / "acceptance.v1.json"]:
    bind(path)
classic_recounts = {"B0": Path("/tmp/inspection-v4-order35-classic-frozen-recount.v1.json"),
                    "B1": run / "qualification/pilot-v4-slots-twentysix-twentyeight.v1/classic-order26-frozen-recount.json",
                    "B2": run / "qualification/pilot-v4-slots-twentynine-thirty.v1/classic-order29-frozen-recount.json"}
for slot in slots:
    index, condition, repeat = slot["order_index"], slot["condition"], slot["repeat_ordinal"]
    classic = condition in classic_recounts
    flights = project / "logs/inspection-eight-condition-dev-flights.v4"
    base = (flights / "classic" if classic else flights) / f"dev-L3-01-{condition}-{repeat}"
    attempt = base / "attempt-00"
    plan, terminal = read(base / "plan.json"), read(attempt / "terminal.json")
    assert plan["case_id"] == slot["case_id"] and plan["condition"] == condition
    assert helper.canonical_digest(plan) == terminal["plan_sha256"] and not (base / "attempt-01").exists()
    callback_path = sdk / f"order-receipts/{index:06d}.json"
    callback = read(callback_path)
    assert all(callback[k] == slot[k] for k in ("order_index", "case_id", "condition", "repeat_ordinal"))
    raw_path = sdk / callback["raw"]
    raw = read(raw_path)
    assert raw["status"] == "completed" and raw["attempted"] is True
    assert (raw["case_id"], raw["condition"], raw["seed"]) == (slot["case_id"], condition, repeat)
    bind(attempt / "terminal.json", raw["evidence"]["episode_receipt_sha256"])
    if index in reconciled:
        value, evidence = reconciled[index]
        assert terminal["status"] == "unknown_process_cleanup" and terminal["Y"] is None
        assert value == raw["value"] and all(raw["evidence"][k] == v for k, v in evidence.items())
    else:
        assert terminal["status"] == "completed" and terminal["returncode"] == 0 and terminal["Y"] == raw["value"]
    world_path = attempt / "flight/world.json"
    assert world_path.stat().st_size < 512 * 1024 * 1024
    world = json.loads(world_path.read_bytes())
    validation = terminal["evidence"] if classic else terminal["physical_validation"]
    bind(world_path, validation["world_sha256"])
    assert world["status"] == "completed" and world["simulated_seconds"] == 600 and world["physics_steps_executed"] == 144000
    assert world["max_clock_lag_seconds"] < .1 and world["clock_lag_over_0p1_seconds_ticks"] == 0 and world["physics_runtime"]["errors"] == []
    recount = json.loads(classic_recounts[condition].read_bytes()) if classic else terminal["evidence"]["frozen_score_recount"]
    assert recount["score_equal"] is True and recount["world_sha256"] == sha(world_path) and recount["score"] == world["score"]
    assert raw["value"] == world["score"]["Y"]
    if classic and condition != "B0":
        bind(classic_recounts[condition])
    for path in [base / "plan.json", attempt / "flight/run.json", attempt / "flight/private-case.json", callback_path, raw_path]:
        bind(path)
    for name, digest in terminal.get("settlement_file_hashes", {}).items():
        bind(Path(name), digest)
    ids = {r["response"]["report_id"] for r in world["reports"] if r["response"].get("status") == "accepted"}
    final = json.loads((attempt / "flight/progress.jsonl").read_text().splitlines()[-1])
    assert final["t"] == 600 and final["reports"] == len(ids) and all(0 <= r["t"] <= 600 for r in world["reports"])
    if classic:
        invocation = sdk / f"invocations/{index:03d}.json"
        check = read(invocation)
        assert check["returncode"] == 0 and check["cleanup"]["status"] == "closed"
        bind(invocation)
    passive = []
    if index >= 31 and not classic:
        passive = raw["evidence"]["passive_close_observations"]
        assert len(passive) == 1
        tree = read(attempt / "process-tree.json")
        bind(attempt / "process-tree.json")
        for ref in passive:
            path = project / ref["path"]
            bind(path, ref["sha256"])
            observed = read(path)
            assert observed["original_result"] == tree and tree["status"] == "closed"
            assert observed["original_exception"] is None and observed["observation_errors"] == []
            assert observed["context"]["plan_sha256"] == terminal["plan_sha256"]
            observations.append(path)
    rows.append({**slot, "attempts": 1, "original_terminal_status": terminal["status"],
                 "accepted_cleanup_disposition": raw["evidence"].get("cleanup_disposition"),
                 "clock": {k: world[k] for k in ("simulated_seconds", "physics_steps_executed", "max_clock_lag_seconds", "clock_lag_p95_seconds", "clock_lag_over_0p1_seconds_ticks")},
                 "score": {k: world["score"][k] for k in ("C", "S", "D", "Q", "H", "E", "Y", "SR")},
                 "usage": None if classic else terminal["evidence"]["model_meter"]["usage"],
                 "report_tool_audit_entries_count": len(world["reports"]), "accepted_unique_report_ids_count": len(ids),
                 "passive_close_observations": passive, "world_bytes": world_path.stat().st_size})
    del world
assert len(observations) == 4
grouped = defaultdict(list)
for row in rows:
    grouped[row["condition"]].append(row)
means = {}
for condition, values in grouped.items():
    expected = 1 if condition in classic_recounts else 3
    assert len(values) == expected and sorted(r["repeat_ordinal"] for r in values) == list(range(expected))
    means[condition] = {"repeats": expected, "mean_score": {k: sum(r["score"][k] for r in values) / expected for k in ("C", "S", "D", "Q", "H", "E", "Y", "SR")}}
status_path = Path("/tmp/inspection-v4-second-root-native-status.v1.json")
status = json.loads(status_path.read_bytes())
budget = status["budget"]
assert status["stale"] == [] and budget["known_usage"] == {"calls": 1267, "tokens": 8154151, "cost": {}}
assert budget["reserved"] == {"calls": 0, "tokens": 0, "cost": {}} and budget["unknown_usage"] == [] and budget["unresolved_usage"] == []
out.mkdir()
captures = {}
for path, name in [(Path(__file__), "capture-source.py"), (classic_recounts["B0"], "classic-order35-frozen-recount.json"),
                   (status_path, "native-status.json"), (Path("/tmp/inspection-pilot-v4-continuation.v2.stdout"), "supervisor.at-second-root.stdout"),
                   (sdk / "checkpoint-pauses.jsonl", "checkpoint-pauses.at-second-root.jsonl")]:
    target = out / name
    shutil.copyfile(path, target)
    captures[str(target.relative_to(run))] = sha(target)
for path in observations:
    target = out / "passive-close" / path.name
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(path, target)
    captures[str(target.relative_to(run))] = sha(target)
for name, ref in refs.items():
    helper.checked(project / name, ref["sha256"])
result = {"role": "one complete L3 development root, all original repeats; not population/formal inference or final SDKouter",
          "at": datetime.now(timezone.utc).isoformat(), "case_id": "dev-L3-01", "completed_callbacks": 36, "complete_physical_root_blocks": 2,
          "planned_callbacks": 108, "provided_sdk_final_outer_recording_complete": False, "barrier": pause,
          "slots": rows, "condition_repeat_means": means, "bound_files": refs, "immutable_captures": captures,
          "native_budget": budget, "limits": "n1L3 plus priorn1L1;twooriginalunknowns retained with exactapplieddispositions;fourpassivesidecars not fullhot ancestry;cost/host/Codex/energyunmetered"}
(out / "binding.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"binding": str(out / "binding.json"), "sha256": sha(out / "binding.json"), "condition_repeat_means": means,
                  "last_two": rows[-2:], "native_known": budget["known_usage"]}, allow_nan=False))
