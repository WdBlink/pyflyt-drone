"""Join audited original development batches into one complete root."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
import sys

from admit_external import OUTPUT
from inspect_original import PROJECT, RUN, checked, previous, read

end = int(sys.argv[1])
assert end in (72,90)
start = end-18
pause = json.loads((OUTPUT/"checkpoint-pauses.jsonl").read_text().splitlines()[-1])
assert pause["order_index"]==end and (OUTPUT/"checkpoint.request").is_file()
slots = read(OUTPUT/"episode-order.json")["slots"][start:end]
case_id = slots[0]["case_id"]
assert len(slots)==18 and all(s["case_id"]==case_id for s in slots)
counts = {"B0":1,"B1":1,"B2":1,"B3-M1":3,"B3-M2":3,"B3-M3":3,"B4-M1":3,"B4-M2":3}
assert Counter(s["condition"] for s in slots)==counts
paths = [RUN/"qualification"/f"pilot-v4-orders-{part}.v1/binding.json" for part in sys.argv[2:]]
batches = [read(p) for p in paths]
evidence = [(row,batch["files"]) for batch in batches for row in batch["slots"]]
assert sorted(row["order_index"] for row,_ in evidence)==list(range(start,end))
refs = [checked(Path(__file__).resolve()), *[checked(path) for path in paths]]
rows = []
for row,pins in sorted(evidence,key=lambda pair:pair[0]["order_index"]):
    i=row["order_index"]
    slot=slots[i-start]
    assert all(row[k]==value for k,value in slot.items())
    order=OUTPUT/f"order-receipts/{i:06d}.json"
    raw_path=OUTPUT/read(order)["raw"]
    raw=read(raw_path)
    assert raw["status"]=="completed" and raw["attempted"] is True and raw["value"]==row["score"]["Y"]
    assert raw["evidence"]["actual_attempts"]==1 and raw["evidence"]["cache_reused"] is False
    terminal=Path(raw["evidence"]["episode_receipt"])
    assert read(terminal)["status"]==row["terminal_status"] and raw["evidence"].get("cleanup_disposition")==row["disposition"]
    for path in (order,raw_path,terminal,terminal.parent/"flight/world.json",terminal.parent.parent/"plan.json"):
        refs.append(checked(path,pins[str(path.relative_to(PROJECT))]))
    rows.append(row)
grouped=defaultdict(list)
for row in rows: grouped[row["condition"]].append(row)
means={}
for key,values in grouped.items():
    assert sorted(r["repeat_ordinal"] for r in values)==list(range(counts[key]))
    values_by_field={k:[r["score"][k] for r in values] for k in ("C","S","D","Q","H","E","Y","SR")}
    means[key]={"repeat_count":len(values),"score":{k:None if all(x is None for x in v) else mean(v)
                                                   for k,v in values_by_field.items()}}
for ref in refs: checked(PROJECT/ref["path"],ref["sha256"])
out=RUN/"qualification"/f"pilot-v4-root-{case_id}.v1"
out.mkdir()
value={"role":"one full development root, original repeats; no formal or population inference",
       "case_id":case_id,"completed_callbacks":end,"complete_root_blocks":end//18,
       "denominator":108,"pause":pause,"rows":rows,"within_root_means":means,"files":refs,
       "sdk_final_outer_complete":False}
previous.durable_once(out/"binding.json",value)
print(json.dumps({"binding":checked(out/"binding.json"),"means":means},allow_nan=False))
