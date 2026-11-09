"""Bind the completed v4 SDK matrix to the independently recounted development slots."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

from admit_external import OUTPUT
from inspect_original import PROJECT, RUN, checked, previous, read

Q = RUN / "qualification"
CAPTURES = ("first-root", "second-root", "third-root", "root-dev-L1-00", "root-dev-L3-00")
GROUPS = {"classic": (18, ("B0","B1","B2"), (0,)),
          "llm": (90, ("B3-M1","B3-M2","B3-M3","B4-M1","B4-M2"), (0,1,2)),
          "root-means": (48, ("B0","B1","B2","B3-M1","B3-M2","B3-M3","B4-M1","B4-M2"), (0,))}
refs = {}
def bind(path, expected=None):
    ref = checked(path, expected)
    assert refs.setdefault(ref["path"], ref["sha256"]) == ref["sha256"]
    return ref

bind(Path(__file__).resolve())
status = read(Q / f"pilot-v4-orders-{sys.argv[-1]}.v1/native-status.json")
assert status["stale"] == []
budget = status["budget"]
assert budget["reserved"] == {"calls": 0, "tokens": 0, "cost": {}}
assert budget["unknown_usage"] == budget["unresolved_usage"] == []
bind(Q / f"pilot-v4-orders-{sys.argv[-1]}.v1/native-status.json")
bind(Q / f"pilot-v4-orders-{sys.argv[-1]}.v1/actual-supervisor-exit.json")
owner = previous.owner()
assert owner["task_stop"] is None and owner["active_round"]
for ref in previous.live_frozen_sources():
    bind(PROJECT / ref["path"], ref["sha256"])
prefix_path = Q / "pilot-v4-cache38-prefix.v1/binding.json"
bind(prefix_path, "b6ba99cce2772efa13ef3932aea60effb33ce5c836f5c2ed286059f398c4fe05")
prefix = read(prefix_path)
assert prefix["completed_callbacks"] == 38 and prefix["new_physics_model_or_native_usage_operations"] == 0
old_denominators = {}
for suffix, completed in (("",24),(".continuation.v1",30),(".continuation.v2",37)):
    prior = PROJECT / f"logs/inspection-eight-condition-pilot.v4{suffix}"
    totals = Counter()
    for group, (denominator, _, _) in GROUPS.items():
        path = prior / group / "results.json"
        bind(path)
        result = read(path)
        assert result["execution_status"] == "failed"
        assert result["reported_denominator"] == len(result["runs"]) == denominator
        counts = Counter(row["status"] for row in result["runs"])
        if group == "root-means":
            assert counts == {"stopped":48}
        else:
            totals.update(counts)
    assert totals == {"completed":completed, "failed":1, "stopped":107-completed}
    old_denominators[prior.name] = dict(totals)
for name in ("source-hashes.json","source-files.json","dependency-hashes.json","input-hashes.json",
             "sampler-seeds.json","episode-order.json"):
    siblings = [PROJECT / f"logs/inspection-eight-condition-pilot.v4{suffix}" / name
                for suffix in ("",".continuation.v1",".continuation.v2",".continuation.v3")]
    pin = bind(siblings[0])["sha256"]
    for path in siblings[1:]:
        bind(path, pin)
assert not (OUTPUT / "checkpoint.request").exists()
schedule = read(OUTPUT / "episode-order.json")
schedule_hash = bind(OUTPUT / "episode-order.json")["sha256"]
slots = schedule["slots"]
assert schedule["seed"] == 2026110641 and len(slots) == 108
seeds = read(OUTPUT / "sampler-seeds.json")
assert seeds["master"] == 2026110642 and len(seeds["by_order_index"]) == 90
completion = read(OUTPUT / "pilot-completion.json")
index = read(OUTPUT / "recording-completion-index.json")
bind(OUTPUT / "pilot-completion.json")
bind(OUTPUT / "recording-completion-index.json")
conditions = {c: len(GROUPS[g][2]) for g in ("classic","llm") for c in GROUPS[g][1]}
assert completion["conditions"] == conditions and completion["independent_roots"] == 6
assert completion["actual_episode_denominator"] == index["episode_denominator"] == 108
assert completion["classic_episode_denominator"] == 18 and completion["llm_episode_denominator"] == 90
assert completion["outer_aggregate_rows"] == 48 and completion["primary_inference_ready"] is False
assert completion["classic_best_unchanged"] == "B1"
assert completion["original_classic_selection_sha256"] == "75f57580f0f5371d91a741fd4f9573e52192d019e9730335123a7ae67897a32e"
assert index["schedule_sha256"] == schedule_hash and set(index["results_sha256"]) == set(GROUPS)
cases = {s["case_id"] for s in slots}
assert len(cases) == 6
original = {}
for name in (*CAPTURES, *[f"orders-{part}" for part in sys.argv[1:]]):
    path = Q / f"pilot-v4-{name}.v1/binding.json"
    bind(path)
    block = read(path)
    for row in (block["rows"] if "rows" in block else block["slots"]):
        i = row["order_index"]
        assert i not in original
        original[i] = row
assert set(original) == set(range(108))
grid = {}
for group, (n, methods, reps) in GROUPS.items():
    result_path = OUTPUT / f"{group}/results.json"
    bind(result_path, index["results_sha256"][group])
    result = read(result_path)
    assert result["design"] == f"{group}/design.json"
    design_path = OUTPUT / result["design"]
    bind(design_path, result["design_sha256"])
    design = read(design_path)
    assert design["execution_schedule"] == {"path": "episode-order.json", "sha256": schedule_hash}
    assert set(design["cases"]) == cases and len(design["cases"]) == len(cases)
    assert design["conditions"] == list(methods) and design["seeds"] == list(reps)
    assert result["execution_status"] == "completed" and result["outcome"] == "inconclusive"
    assert result["reported_denominator"] == result["attempted_count"] == len(result["runs"]) == n
    runs = {}
    for row in result["runs"]:
        identity = row["case_id"], row["seed"], row["condition"]
        case, rep, condition = identity
        assert identity not in runs and row["raw"] == f"{group}/raw/{case}-{condition}-{rep}.json"
        raw = read(OUTPUT / row["raw"])
        pin = bind(OUTPUT / row["raw"])["sha256"]
        assert all(raw[k] == row[k] for k in ("case_id","seed","condition","status","value","attempted"))
        assert raw["status"] == "completed" and raw["attempted"] is True
        assert type(raw["value"]) in (int,float) and 0 <= raw["value"] <= 1
        assert raw["evidence"]["schedule_sha256"] == schedule_hash
        runs[identity] = row["raw"], pin, raw
    assert set(runs) == {(case, rep, cond) for case in cases for rep in reps for cond in methods}
    assert {str(p.relative_to(OUTPUT)) for p in (OUTPUT/group/"raw").glob("*.json")} == {r[0] for r in runs.values()}
    grid[group] = runs
seen = set()
known = dict(calls=0, tokens=0)
expected_reservations = set()
cache = Counter()
lag = []
for i, slot in enumerate(slots):
    case, rep, cond = slot["case_id"], slot["repeat_ordinal"], slot["condition"]
    ident = case, rep, cond
    assert slot["order_index"] == i and slot["root_block_index"] == i // 18 and ident not in seen
    assert slots[i-i%18]["case_id"] == case
    seen.add(ident)
    group = "classic" if cond in GROUPS["classic"][1] else "llm"
    path, pin, raw = grid[group][ident]
    p = OUTPUT / f"order-receipts/{i:06d}.json"
    order = read(p)
    bind(p)
    assert all(order[k] == v for k,v in slot.items())
    assert order["schedule_sha256"] == schedule_hash and order["raw"] == path
    start = OUTPUT / f"starts/{case}-{cond}-{rep}.json"
    bind(start)
    start_doc = read(start)
    assert start_doc["schedule_sha256"] == schedule_hash and all(start_doc[k] == slot[k]
           for k in ("case_id", "repeat_ordinal", "condition", "order_index"))
    r = original[i]
    assert all(r[k] == v for k,v in slot.items()) and raw["value"] == r["score"]["Y"]
    clock = r["clock"]["max_clock_lag_seconds"] if "clock" in r else r["max_lag_seconds"]
    assert 0 <= clock <= .1
    lag.append((clock, i))
    e = raw["evidence"]
    assert e["adapter_condition"] == cond and e["adapter_repeat_ordinal"] == rep and e["order_index"] == i
    assert e["actual_attempts"] == 1 and e["cache_reused"] is (i < 38)
    cache[e["cache_reused"]] += 1
    term = Path(e["episode_receipt"])
    bind(term, e["episode_receipt_sha256"])
    terminal = read(term)
    assert terminal["status"] == r.get("original_terminal_status", r.get("terminal_status"))
    assert previous.canonical_digest(read(term.parent.parent/"plan.json")) == terminal["plan_sha256"]
    bind(term.parent.parent/"plan.json")
    if terminal["status"] == "completed":
        assert terminal["Y"] == raw["value"] and e.get("cleanup_disposition") is None
    else:
        assert terminal["status"] == "unknown_process_cleanup" and terminal["Y"] is None
        assert terminal["retry_allowed"] is False and e["cleanup_disposition"]["original_status_retained"] == terminal["status"]
    if group == "llm":
        reservation = f"dev-all8-pilot-v4-{i:03d}-a0"
        expected_reservations.add(reservation)
        assert read(term.parent.parent/"plan.json")["sampler_seed"] == seeds["by_order_index"][str(i)]
        record = owner["usage_ledger"][reservation]
        meter = terminal["evidence"]["model_meter"]["usage"]
        assert record["status"] == "known" and record["reported_usage"] == r["usage"] == meter
        bind(RUN / "qualification/local-model-episodes" / reservation / "usage-receipt.json", record["receipt_sha256"])
        for key in known:
            known[key] += meter[key]
    else:
        # Historical classical summaries omit the LLM-only usage field.
        assert r.get("usage") is None and str(i) not in seeds["by_order_index"]
assert len(list((OUTPUT/"order-receipts").glob("*.json"))) == len(list((OUTPUT/"starts").glob("*.json"))) == 108
assert {name for name in owner["usage_ledger"] if name.startswith("dev-all8-pilot-v4-")} == expected_reservations
assert cache == {True: 38, False: 70}
for (case, _, cond), (_, _, raw) in grid["root-means"].items():
    group = "classic" if cond in GROUPS["classic"][1] else "llm"
    cells = [grid[group][case,rep,cond] for rep in range(conditions[cond])]
    e = raw["evidence"]
    assert e["inner_results"] == f"{group}/results.json" and e["inner_results_sha256"] == index["results_sha256"][group]
    assert e["not_an_additional_episode"] is True and e["actual_episode_count"] == len(cells)
    assert e["inner_attempts"] == [{"repeat_ordinal": rep, "path": path, "sha256": pin}
                                    for rep,(path,pin,_) in enumerate(cells)]
    assert raw["value"] == mean(cell["value"] for _,_,cell in cells)
roots = [{"case_id": block[0]["case_id"], "level": block[0]["level"],
          "Y": {method: grid["root-means"][block[0]["case_id"], 0, method][2]["value"] for method in conditions}}
         for block in (slots[i:i+18] for i in range(0, 108, 18))]
assert Counter(r["level"] for r in roots) == {1: 2, 2: 2, 3: 2}
macro = {method: mean(r["Y"][method] for r in roots) for method in conditions}
path = Q / "pilot-v4-complete.v1/binding.json"
path.parent.mkdir()
value = {"role":"completed development matrix; no formal result or G4 acceptance",
         "sdk_groups":{group:len(rows) for group,rows in grid.items()},
         "prior_aborted_sdk_inner_denominators_retained":old_denominators,
         "physical_episodes":108,"descriptive_root_blocks":roots,"development_macro_Y":macro,
         "max_lag_seconds":max(lag)[0],"max_lag_order":max(lag)[1],
         "strict_original_closed":sum(r.get("original_terminal_status", r.get("terminal_status")) == "completed" for r in original.values()),
         "original_cleanup_unknown_orders_retained":[i for i,r in original.items()
             if r.get("original_terminal_status", r.get("terminal_status")) == "unknown_process_cleanup"],
         "v4_known_local_usage":known,"native_budget":budget,"files":refs}
previous.durable_once(path, value)
print(json.dumps({"binding":checked(path),"summary":{k:v for k,v in value.items() if k != "files"}},allow_nan=False))
