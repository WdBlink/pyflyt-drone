"""Inventory closed development traces. Size samples are not future limits."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
LOGS = PROJECT / "logs"
PILOT = LOGS / "inspection-eight-condition-pilot.v4.continuation.v3"
Q = RUN / "qualification"
OUTPUT = Q / "formal-capacity-final.v1"
CLASSES = {"B0", "B1", "B2"}


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def files(root):
    assert root.is_dir() and not root.is_symlink()
    result = {}
    for path in root.rglob("*"):
        assert not path.is_symlink()
        if path.is_file():
            info = path.stat()
            assert info.st_nlink == 1
            result[path] = info.st_size
    return result


def sample(attempt, role, level, condition, reservation=None):
    paths = files(attempt)
    terminal = read(attempt/"terminal.json")
    assert (attempt/"flight/world.json") in paths
    wire = {name: paths.pop(attempt/name) for name in ("reserve-owner.json", "settle-owner.json")} if reservation else {}
    data = {"attempt":str(attempt.relative_to(PROJECT)), "role":role, "level":level, "condition":condition,
        "terminal_status":terminal["status"], "terminal_sha256":digest(attempt/"terminal.json"),
        "world_bytes":paths[attempt/"flight/world.json"], "episode_nonwire_bytes":sum(paths.values()),
        "native_stdout_bytes":wire, "files":len(paths)+len(wire),
        "longest_path_bytes":max(len(str(path).encode()) for path in paths)}
    if reservation:
        assert read(attempt/"intent.json")["reservation"] == reservation
        operation = files(LOGS/"inspection-model-resource-operations.v1"/reservation)
        receipt = files(Q/"local-model-episodes"/reservation)
        request = {p:p.stat().st_size for p in (RUN/"resource-requests").glob(reservation+"*.json")}
        assert len(request) == 2 and len(receipt) == 2
        assert not (set(paths)&set(operation) or set(paths)&set(receipt) or set(operation)&set(receipt))
        archive = next(p for p in receipt if p.name=="complete-attempt.zip")
        usage = next(p for p in receipt if p.name=="usage-receipt.json")
        data.update(reservation=reservation, model_originals_bytes=sum(operation.values()),
            archive_bytes=receipt[archive], usage_receipt_bytes=receipt[usage],
            resource_packet_bytes=sum(request.values()), files=data["files"]+len(operation)+len(receipt)+len(request),
            longest_path_bytes=max(data["longest_path_bytes"],max(len(str(path).encode())
                for path in (*operation,*receipt,*request))))
    return data


def main():
    pilot = Q/"pilot-v4-complete.v1/binding.json"
    assert digest(pilot) == "d51ef4f0552cd447c5c7450d8e389c345f7f12c947a1d037cbb8d22a99f6d622"
    schedule = read(PILOT/"episode-order.json")["slots"]
    assert len(schedule) == 108
    samples = []
    for slot in schedule:
        i, cond = slot["order_index"], slot["condition"]
        root = LOGS/"inspection-eight-condition-dev-flights.v4"
        if cond in CLASSES:
            root = root/"classic"
        directory = root/f"{slot['case_id']}-{cond}-{slot['repeat_ordinal']}"
        reservation = None if cond in CLASSES else f"dev-all8-pilot-v4-{i:03d}-a0"
        samples.append(sample(directory/"attempt-00","v4_original",slot["level"],cond,reservation))
    old = Q/"eight-condition-pilot-terminated.v3/binding.json"
    for row in read(old)["attempts"]:
        attempt = LOGS/"inspection-eight-condition-dev-flights.v3/dev-L3-00-B3-M2-1"/f"attempt-{row['attempt_ordinal']:02d}"
        assert digest(attempt/"terminal.json") == row["terminal_sha256"]
        samples.append(sample(attempt,"historical_clock_failure",3,"B3-M2",read(attempt/"intent.json")["reservation"]))
    clean = Q/"final-development-clean.v1/binding.json"
    assert read(clean)["clean_actual_attempts"] >= 1
    for attempt in sorted((LOGS/"inspection-final-development-clean.v1/devext-OOD-L3-00-B0-0").glob("attempt-*")):
        samples.append(sample(attempt,"installed_clean",3,"B0"))
    delay = Q/"final-development-delay.v1/result.json"
    assert read(delay)["actual_attempts"] >= 1
    for attempt in sorted((LOGS/"inspection-final-development-delay.v1").glob("attempt-*")):
        samples.append(sample(attempt,"installed_delay",3,"B4-M1",read(attempt/"intent.json")["reservation"]))
    store = shutil.disk_usage(PROJECT)
    assert len({path.stat().st_dev for path in (PROJECT,RUN,LOGS)}) == 1
    maxima = {}
    for group in ("classic","llm"):
        rows = [r for r in samples if ("reservation" in r)==(group=="llm")]
        maxima[group] = {k:max(row.get(k,0) for row in rows) for k in (
            "world_bytes","episode_nonwire_bytes","model_originals_bytes","archive_bytes","usage_receipt_bytes",
            "resource_packet_bytes","files","longest_path_bytes")}
    out = OUTPUT
    out.mkdir(exist_ok=True)
    value = {"role":"actual closed development inventory; no future bound or new experiment",
        "at":datetime.now(timezone.utc).isoformat(), "physical_sample_count":len(samples),
        "filesystem":{"total":store.total,"available":store.free,"device":PROJECT.stat().st_dev},
        "source_pins":{str(p.relative_to(PROJECT)):digest(p) for p in (
            pilot,old,clean,delay,Path(__file__).resolve())}, "observed_maxima":maxima,"samples":samples}
    with (out/"flight-storage.json").open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in value.items() if k not in ("samples","source_pins")},allow_nan=False))


if __name__ == "__main__":
    main()
