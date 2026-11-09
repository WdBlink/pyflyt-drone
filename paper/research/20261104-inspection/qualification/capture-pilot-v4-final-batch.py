"""Recount the final original v4 batch after the supervisor actually exits."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from admit_external import OUTPUT, check_disposition
from inspect_original import FLIGHTS, PROJECT, RUN, checked, context, inspect_attempt, previous, read
from launchd_provenance import match_sidecar

args = argparse.ArgumentParser(description=__doc__)
args.add_argument("first", type=int)
args.add_argument("next", type=int)
args.add_argument("status", type=Path)
args.add_argument("exit_receipt", type=Path)
a = args.parse_args()
assert 38 <= a.first < a.next == 108
exit_receipt = read(a.exit_receipt)
assert exit_receipt["session_id"] == 46816 and exit_receipt["exit_code"] == 0
completion = OUTPUT / "pilot-completion.json"
index = OUTPUT / "recording-completion-index.json"
assert read(completion)["actual_episode_denominator"] == 108
assert read(index)["episode_denominator"] == 108
assert not (OUTPUT / "checkpoint.request").exists()
out = RUN / f"qualification/pilot-v4-orders-{a.first:03d}-{a.next-1:03d}.v1"
assert not out.exists()
pause_file = OUTPUT / "checkpoint-pauses.jsonl"
pause = json.loads(pause_file.read_text().splitlines()[-1])
assert a.first <= pause["order_index"] < a.next
slots = read(OUTPUT / "episode-order.json")["slots"]
assert len(slots) == 108
def episode(slot):
    base = FLIGHTS / "classic" if slot["condition"] in ("B0","B1","B2") else FLIGHTS
    return base / f"{slot['case_id']}-{slot['condition']}-{slot['repeat_ordinal']}"
assert not list((RUN / "resource-requests").glob("dev-all8-pilot-v4-108*"))
assert len(list((OUTPUT / "order-receipts").glob("*.json"))) == a.next
owner = previous.owner()
assert owner["task_stop"] is None and owner["active_round"]
status = json.loads(a.status.read_bytes())
budget = status["budget"]
assert status["stale"] == [] and budget["unknown_usage"] == [] and budget["unresolved_usage"] == []
assert budget["reserved"] == {"calls":0,"tokens":0,"cost":{}}
refs, cases, recounted = {}, [], {}
def bind(path, expected=None):
    ref = checked(path, expected)
    refs[ref["path"]] = ref["sha256"]
    return ref
for ref in previous.live_frozen_sources():
    bind(PROJECT / ref["path"],ref["sha256"])
companion_path = OUTPUT / "recovery-adapter.json"
companion = read(companion_path)
bind(companion_path)
policy = companion["prospective_policy"]
for i in range(a.first, a.next):
    slot = slots[i]
    order = OUTPUT / f"order-receipts/{i:06d}.json"
    callback = read(order)
    assert all(callback[k] == slot[k] for k in ("case_id","condition","order_index","repeat_ordinal"))
    raw_path = OUTPUT / callback["raw"]
    raw = read(raw_path)
    assert raw["status"] == "completed" and raw["attempted"] is True
    assert (raw["case_id"], raw["condition"], raw["seed"]) == (slot["case_id"],slot["condition"],slot["repeat_ordinal"])
    e = raw["evidence"]
    assert e["cache_reused"] is False and e["actual_attempts"] == 1
    directory = episode(slot)
    attempt = directory / "attempt-00"
    assert Path(e["episode_receipt"]) == attempt / "terminal.json" and not (directory / "attempt-01").exists()
    for path in (directory / "plan.json", order,raw_path):
        bind(path)
    bind(attempt / "terminal.json",e["episode_receipt_sha256"])
    plan, terminal = read(directory / "plan.json"), read(attempt / "terminal.json")
    assert previous.canonical_digest(plan) == terminal["plan_sha256"] and terminal["returncode"] == 0
    world_path = attempt / "flight/world.json"
    world = json.loads(world_path.read_bytes())
    classical = slot["condition"] in ("B0","B1","B2")
    usage = None
    if classical:
        assert terminal["status"] == "completed" and terminal["Y"] == raw["value"]
        invocation = OUTPUT / f"invocations/{i:03d}.json"
        inv = read(invocation)
        assert inv["returncode"] == 0 and inv["cleanup"]["status"] == "closed"
        bind(invocation)
        world_ref = bind(world_path,terminal["evidence"]["world_sha256"])
        evaluator = RUN / "experiment/evaluator.v2.py"
        bind(evaluator,"de826d2488b5a8ad366ccec9ead8b81bebf6b794a533a38ac55338916848611b")
        output = subprocess.check_output([sys.executable,str(evaluator),"--world",str(world_path),
                    "--expected-world-sha256",world_ref["sha256"]],timeout=60)
        recount = json.loads(output)
        recounted[f"order{i:03d}-classic-recount.json"] = output
    else:
        reservation = f"dev-all8-pilot-v4-{i:03d}-a0"
        _, _, tree, y, bound = inspect_attempt(directory, 0, reservation, owner)
        assert y == raw["value"]
        for ref in bound:
            bind(PROJECT / ref["path"],ref["sha256"])
        side, = e["passive_close_observations"]
        side_path = PROJECT / side["path"]
        bind(side_path,side["sha256"])
        observed = read(side_path)
        assert observed["original_result"] == tree and observed["context"] == context(plan,reservation[:-3])
        assert observed["original_exception"] is None and observed["observation_errors"] == []
        if "cleanup_disposition" in e:
            check_disposition(directory,plan,reservation[:-3],y,e,policy)
            assert terminal["status"] == "unknown_process_cleanup" and terminal["Y"] is None
            ref = e["cleanup_disposition"]["proof"]
            bind(PROJECT / ref["path"],ref["sha256"])
            proof = read(PROJECT / ref["path"])
            q = proof["query"]
            bind(PROJECT / q["path"],q["sha256"])
            metadata = read(PROJECT / q["path"])
            log = metadata["events"]
            log_path = PROJECT / log["path"]
            bind(log_path,log["sha256"])
            attribution = match_sidecar(observed,tree,observed["context"],previous.OBSERVER_SHA,
                                        json.loads(log_path.read_bytes()),proof["attribution"]["gui_uid"])
            assert attribution == proof["attribution"] and metadata["boot"] == attribution["python_spawn"]["bootUUID"]
        else:
            assert tree["status"] == "closed" and terminal["status"] == "completed" and terminal["Y"] == y
        recount = terminal["evidence"]["frozen_score_recount"]
        usage = terminal["evidence"]["model_meter"]["usage"]
    assert recount["score_equal"] and recount["score"] == world["score"] and raw["value"] == world["score"]["Y"]
    assert world["simulated_seconds"] == 600 and world["physics_steps_executed"] == 144000
    assert world["max_clock_lag_seconds"] <= .1 and world["clock_lag_over_0p1_seconds_ticks"] == 0
    assert world["physics_runtime"]["errors"] == []
    progress = attempt / "flight/progress.jsonl"
    ids = {r["response"]["report_id"] for r in world["reports"] if r["response"].get("status") == "accepted"}
    last = json.loads(progress.read_text().splitlines()[-1])
    assert last["t"] == 600 and last["reports"] == len(ids) and all(0 <= r["t"] <= 600 for r in world["reports"])
    for path in (world_path,progress,attempt/"flight/run.json",attempt/"flight/private-case.json"):
        bind(path)
    cases.append({**slot, "terminal_status":terminal["status"], "disposition":e.get("cleanup_disposition"),
        "score":{key:world["score"][key] for key in ("C","S","D","Q","H","E","Y","SR")},
        "max_lag_seconds":world["max_clock_lag_seconds"],"p95_lag_seconds":world["clock_lag_p95_seconds"],
        "report_audit_entries":len(world["reports"]),"accepted_unique_ids":len(ids),"usage":usage,
        "passive_close_observations":e.get("passive_close_observations",[])})
for path in (completion,index,pause_file,Path(__file__).resolve()):
    bind(path)
assert all(previous.digest(PROJECT / path)==sha for path,sha in refs.items())
out.mkdir()
for source,name in ((a.status,"native-status.json"), (a.exit_receipt,"actual-supervisor-exit.json"),
        (pause_file,"checkpoint-pauses.jsonl"),
        (Path("/tmp/inspection-pilot-v4-continuation.v3.stdout"),"supervisor.stdout")):
    target=out/name
    shutil.copyfile(source,target)
    bind(target)
for name,data in recounted.items():
    (out/name).write_bytes(data)
    bind(out/name)
result={"role":"actual original v4 development batch; no new statistical replication or formal qualification",
        "previous_pause":pause,"exit_receipt":exit_receipt,"completed_callbacks":a.next,"denominator":108,"slots":cases,"native_budget":budget,"files":refs}
binding=out/"binding.json"
previous.durable_once(binding,result)
print(json.dumps({"binding":checked(binding),"slots":cases,"budget":budget["known_usage"]},allow_nan=False))
