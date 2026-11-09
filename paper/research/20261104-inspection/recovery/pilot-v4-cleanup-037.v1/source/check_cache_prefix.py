"""Freeze the actual same-cohort prefix only while the next dispatch is paused."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

P = Path.cwd()
R = P / "paper/research/20261104-inspection"
O = P / "logs/inspection-eight-condition-pilot.v4"
V1 = O.with_name(O.name + ".continuation.v1")
V2 = O.with_name(O.name + ".continuation.v2")
V3 = O.with_name(O.name + ".continuation.v3")
FLIGHTS = P / "logs/inspection-eight-condition-dev-flights.v4"
OUT = R / "qualification/pilot-v4-cache38-prefix.v1"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = {}

    def bind(path, expected=None):
        assert path.is_relative_to(P) and path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
        actual = sha(path)
        assert expected is None or expected == actual, path
        files[str(path.relative_to(P))] = actual

    common = ("source-hashes.json", "source-files.json", "dependency-hashes.json", "input-hashes.json",
              "episode-order.json", "sampler-seeds.json")
    for name in common:
        frozen = (O / name).read_bytes()
        for root in (O, V1, V2, V3):
            assert (root / name).read_bytes() == frozen, (root, name)
            bind(root / name)
    for name, count in (("source-hashes.json",104), ("dependency-hashes.json",61), ("input-hashes.json",10)):
        actual = read(V3 / name)
        assert len(actual) == count and all(sha(Path(name)) == pin for name,pin in actual.items())
    planned = read(V3 / "episode-order.json")["slots"]
    assert len(planned) == 108 and [s["order_index"] for s in planned] == list(range(108))
    orig = [read(path) for group in ("classic", "llm") for path in (V2 / group / "raw").glob("*.json")]
    counts = Counter(row["status"] for row in orig)
    assert counts == {"completed":37,"failed":1,"stopped":70}
    exact = read(Path("/tmp/inspection-v4-037-accepted-verify.v1.json"))
    measured = []
    for index, slot in enumerate(planned[:38]):
        order = V3 / f"order-receipts/{index:06d}.json"
        record = read(order)
        assert all(record[k] == slot[k] for k in ("case_id", "condition", "order_index", "repeat_ordinal"))
        new_raw = V3 / record["raw"]
        old_raw = V2 / record["raw"]
        row, before = read(new_raw), read(old_raw)
        proof = row["evidence"]
        assert row["status"] == "completed" and row["attempted"] is True and proof["cache_reused"] is True
        assert (row["case_id"],row["condition"],row["seed"]) == (slot["case_id"],slot["condition"],slot["repeat_ordinal"])
        assert proof["actual_attempts"] == 1
        terminal = Path(proof["episode_receipt"])
        bind(terminal, proof["episode_receipt_sha256"])
        original = read(terminal)
        if index < 37:
            assert before["status"] == "completed" and before["value"] == row["value"], index
            assert {k:v for k,v in proof.items() if k!="cache_reused"} == {
                k:v for k,v in before["evidence"].items() if k!="cache_reused"}, index
        else:
            assert before["status"] == "failed" and before["attempted"] is True and exact["Y"] == row["value"]
            assert all(proof[k] == v for k,v in exact["evidence"].items())
        if index in (24,30,37):
            assert original["status"] == "unknown_process_cleanup" and original["Y"] is None and original["retry_allowed"] is False
            assert proof["cleanup_disposition"]["original_status_retained"] == "unknown_process_cleanup"
        else:
            assert original["status"] == "completed" and original["Y"] == row["value"]
        for side in proof.get("passive_close_observations", []):
            assert (P / side["path"]).parent == V2 / "cleanup-observations"
            bind(P / side["path"], side["sha256"])
        for path in (order,new_raw,old_raw):
            bind(path)
        measured.append(new_raw)
    pause = [json.loads(line) for line in (V3 / "checkpoint-pauses.jsonl").read_text().splitlines()]
    assert len(pause) == 1 and pause[0]["order_index"] == 38 and (V3 / "checkpoint.request").is_file()
    assert len(list((V3 / "order-receipts").glob("*.json"))) == 38
    assert planned[38]["case_id"] == "dev-L2-00" and planned[38]["condition"] == "B4-M2" and planned[38]["repeat_ordinal"] == 2
    assert not (FLIGHTS / "dev-L2-00-B4-M2-2").exists()
    assert len(list(FLIGHTS.rglob("attempt-*"))) == 38 and all(path.name=="attempt-00" for path in FLIGHTS.rglob("attempt-*"))
    assert not list((R / "resource-requests").glob("dev-all8-pilot-v4-038*"))
    assert not list((R / "qualification/local-model-episodes").glob("dev-all8-pilot-v4-038*"))
    assert not list((V3 / "cleanup-observations").glob("*")) and not (V3 / "external-cleanup-dispositions").exists()
    invocations = list((V3 / "invocations").glob("*.json"))
    assert len(invocations) == 6
    for path in invocations:
        inv = read(path)
        assert inv["returncode"] == 0 and inv["cleanup"]["status"] == "closed"
        assert inv["cleanup"]["remaining_owned"] == inv["cleanup"]["unclaimed_new_python_orphans"] == []
        bind(path)
    status_path = Path("/tmp/inspection-v4-cache38-native-status.v1.json")
    native = read(status_path)
    budget = native["budget"]
    assert native["stale"] == [] and budget["known_usage"] == {"calls":1314,"tokens":8480064,"cost":{}}
    assert budget["reserved"] == {"calls":0,"tokens":0,"cost":{}} and budget["unknown_usage"] == budget["unresolved_usage"] == []
    meta = read(V3 / "recovery-adapter.json")
    assert meta["physical_denominator"] == 108 and meta["cache_barrier_before_order"] == 38
    assert meta["reconciled"]["37"] == exact["evidence"]
    refs = meta["prospective_policy"]
    for ref in (refs["policy"],refs["review"],*refs["sources"],*meta["passive_observer"]):
        bind(P / ref["path"],ref["sha256"])
    bind(Path(__file__).resolve())
    capture = {"native-status.json": status_path,
               "accepted037-verify.json":Path("/tmp/inspection-v4-037-accepted-verify.v1.json"),
               "supervisor-at-barrier.stdout":Path("/tmp/inspection-pilot-v4-continuation.v3.stdout")}
    for name in (*common,"checkpoint.request","checkpoint-pauses.jsonl","recovery-adapter.json"):
        capture["sdk/" + name] = V3 / name
        bind(V3 / name)
    for raw in measured:
        capture["sdk/" + str(raw.relative_to(V3))] = raw
    OUT.mkdir()
    for name, source in capture.items():
        target = OUT / name
        target.parent.mkdir(exist_ok=True,parents=True)
        shutil.copyfile(source,target)
        bind(target)
    assert all(sha(P / path) == pin for path,pin in files.items())
    result = {"role":"actual 38 original v4 cached callbacks; no new physical/model effects or G4 authorization",
              "pause":pause[0],"old_status_counts":counts,"strict_cache":35,"scoped_disposition_cache":3,
              "completed_callbacks":38,"physical_denominator":108,"remaining_original_slots":70,
              "new_physics_model_or_native_usage_operations":0,"original_attempt_directories":38,
              "native_known_usage":budget["known_usage"],"prospective_policy":refs,"files":files}
    binding = OUT / "binding.json"
    with binding.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({"path":str(binding.relative_to(R)),"sha256":sha(binding),"next_order":38}))


if __name__ == "__main__":
    main()
