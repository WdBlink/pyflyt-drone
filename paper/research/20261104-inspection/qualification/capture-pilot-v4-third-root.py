"""Join the 18 audited original L2 callbacks; no new measurements or inference."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean

import disposition037
from admit_external import OUTPUT
from inspect_original import BUNDLE, FLIGHTS, PROJECT, RUN, checked, previous, read

pause = json.loads((OUTPUT/"checkpoint-pauses.jsonl").read_text().splitlines()[-1])
assert pause["order_index"] == 54 and (OUTPUT/"checkpoint.request").is_file()
slots = read(OUTPUT/"episode-order.json")["slots"][36:54]
assert [s["order_index"] for s in slots] == list(range(36,54))
assert all(s["case_id"]=="dev-L2-00" and s["level"]==2 for s in slots)
counts = {"B0":1,"B1":1,"B2":1,"B3-M1":3,"B3-M2":3,"B3-M3":3,"B4-M1":3,"B4-M2":3}
assert Counter(s["condition"] for s in slots)==counts
refs = {}
def bind(path, sha=None):
    value = checked(path,sha)
    refs[value["path"]]=value["sha256"]
    return value
paths = [RUN/f"qualification/pilot-v4-orders-{i:03d}-{j:03d}.v1/binding.json"
         for i,j in ((38,40),(41,43),(44,46),(47,49),(50,52),(53,53))]
batches = [(read(path),bind(path)) for path in paths]
evidence = {s["order_index"]:(s,b["files"]) for b,_ in batches for s in b["slots"]}
assert set(evidence)==set(range(38,54))
facts = BUNDLE/"evidence.v1/binding.json"
bind(facts,disposition037.FACTS_SHA)
initial = read(facts)
pin = {r["path"]:r["sha256"] for r in initial["files"]}
history_y, history = disposition037.verify()
rows=[]
for slot in slots:
    i, condition, repeat = (slot[k] for k in ("order_index","condition","repeat_ordinal"))
    classic = condition in ("B0","B1","B2")
    base = FLIGHTS/"classic" if classic else FLIGHTS
    episode = base/f"dev-L2-00-{condition}-{repeat}"
    attempt = episode/"attempt-00"
    callback_path = OUTPUT/f"order-receipts/{i:06d}.json"
    c = read(callback_path)
    assert all(c[k]==slot[k] for k in ("order_index","case_id","condition","repeat_ordinal"))
    raw_path = OUTPUT/c["raw"]
    raw = read(raw_path)
    assert raw["status"]=="completed" and raw["attempted"] is True and raw["evidence"]["actual_attempts"]==1
    terminal_path = attempt/"terminal.json"
    terminal = read(terminal_path)
    assert terminal["returncode"]==0 and terminal["plan_sha256"]==previous.canonical_digest(read(episode/"plan.json"))
    assert not (episode/"attempt-01").exists()
    bind(terminal_path,raw["evidence"]["episode_receipt_sha256"])
    if i < 38:
        prior = initial["last_slots"][str(i)]
        score = prior["score_before_cleanup_gate"]
        bind(terminal_path,pin[str(terminal_path.relative_to(PROJECT))])
        assert (terminal["status"]==prior["terminal_status"] and raw["value"]==score["Y"]
                and raw["evidence"]["cache_reused"] is True)
        if i==37:
            assert history_y==raw["value"] and raw["evidence"]["cleanup_disposition"]==history["cleanup_disposition"]
        detail = {"score":{k:score[k] for k in ("C","S","D","Q","H","E","Y","SR")},
                  "max_lag_seconds":prior["max_lag_seconds"],"usage":prior["meter"],
                  "accepted_unique_ids":prior["accepted_unique_ids"],"report_audit_entries":prior["report_audit_entries"]}
        for name in ("flight/world.json","flight/run.json","flight/private-case.json"):
            path = attempt/name
            bind(path,pin[str(path.relative_to(PROJECT))])
    else:
        detail, pins = evidence[i]
        assert detail["score"]["Y"]==raw["value"] and detail["terminal_status"]==terminal["status"]
        assert raw["evidence"]["cache_reused"] is False and detail["disposition"]==raw["evidence"].get("cleanup_disposition")
        for path in (raw_path,callback_path,episode/"plan.json",terminal_path,attempt/"flight/world.json"):
            bind(path,pins[str(path.relative_to(PROJECT))])
    bind(raw_path)
    bind(callback_path)
    rows.append({**slot, **detail, "original_terminal_status":terminal["status"],
                 "disposition":raw["evidence"].get("cleanup_disposition")})
grouped=defaultdict(list)
for row in rows:
    grouped[row["condition"]].append(row)
means={}
for cond, values in grouped.items():
    assert sorted(r["repeat_ordinal"] for r in values)==list(range(counts[cond]))
    means[cond]={"repeat_count":len(values), "score":{k:mean(r["score"][k] for r in values)
                     for k in ("C","S","D","Q","H","E","Y","SR")}}
bind(Path(__file__).resolve())
assert all(previous.digest(PROJECT/path)==sha for path,sha in refs.items())
out=RUN/"qualification/pilot-v4-third-root.v1"
out.mkdir()
result={"role":"one complete L2 development root, all original repeats; no formal or population inference",
        "at":datetime.now(timezone.utc).isoformat(),"case_id":"dev-L2-00","completed_callbacks":54,
        "complete_root_blocks":3,"denominator":108,"pause":pause,"rows":rows,"within_root_means":means,
        "frozen_batches":[ref for _,ref in batches],"bound_files":refs,
        "sdk_final_outer_complete":False}
previous.durable_once(out/"binding.json",result)
print(json.dumps({"binding":checked(out/"binding.json"),"within_root_means":means},allow_nan=False))
