"""Assert and freeze the actual31-cache barrier before any new v4 effect."""
import hashlib
import json
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parents[3]
PROJECT = RUN.parents[2]
OLD = PROJECT / "logs/inspection-eight-condition-pilot.v4.continuation.v1"
NEW = PROJECT / "logs/inspection-eight-condition-pilot.v4.continuation.v2"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    refs = {}

    def bind(path):
        path = path.resolve()
        ref = {"path": str(path.relative_to(PROJECT)), "sha256": sha(path)}
        refs[ref["path"]] = ref
        return ref

    files = ["source-hashes.json", "source-files.json", "dependency-hashes.json",
             "input-hashes.json", "episode-order.json", "sampler-seeds.json"]
    for name in files:
        assert (OLD / name).read_bytes() == (NEW / name).read_bytes(), name
        original = PROJECT / "logs/inspection-eight-condition-pilot.v4" / name
        assert original.read_bytes() == (NEW / name).read_bytes(), name
        bind(original)
        bind(OLD / name)
        bind(NEW / name)
    for name, count in [("source-hashes.json", 104), ("dependency-hashes.json", 61),
                        ("input-hashes.json", 10)]:
        expected = read(NEW / name)
        assert len(expected) == count
        assert all(sha(Path(path)) == digest for path, digest in expected.items()), name
    slots = read(NEW / "episode-order.json")["slots"]
    assert len(slots) == 108 and [s["order_index"] for s in slots] == list(range(108))
    old_rows = [read(p) for kind in ("classic", "llm") for p in (OLD / kind / "raw").glob("*.json")]
    assert Counter(r["status"] for r in old_rows) == {"completed": 30, "failed": 1, "stopped": 77}
    assert next(r for r in old_rows if r["status"] == "failed")["condition"] == "B4-M1"
    paths = [p for kind in ("classic", "llm") for p in (NEW / kind / "raw").glob("*.json")]
    rows = sorted(((read(p), p) for p in paths), key=lambda pair: pair[0]["evidence"]["order_index"])
    assert len(rows) == 31
    for index, (row, path) in enumerate(rows):
        slot, evidence = slots[index], row["evidence"]
        assert evidence["order_index"] == index and row["status"] == "completed" and row["attempted"] is True
        assert evidence["cache_reused"] is True and evidence["actual_attempts"] == 1
        assert (row["case_id"], row["condition"], row["seed"]) == (
            slot["case_id"], slot["condition"], slot["repeat_ordinal"])
        assert sha(Path(evidence["episode_receipt"])) == evidence["episode_receipt_sha256"]
        bind(Path(evidence["episode_receipt"]))
        bind(path)
        old_path = OLD / path.relative_to(NEW)
        previous = read(old_path)
        bind(old_path)
        if index < 30:
            assert previous["status"] == "completed" and previous["value"] == row["value"]
            for key, value in previous["evidence"].items():
                if key != "cache_reused":
                    assert evidence[key] == value, (index, key)
        else:
            assert previous["status"] == "failed" and row["value"] == 0.3436623024638006
            disposition = evidence["cleanup_disposition"]
            assert disposition["original_status_retained"] == "unknown_process_cleanup"
            assert disposition["native_request_sha256"] == "366cf380ad7c5043f8316f9ddaaccfc202e3a210a542b4b912b9119f69df1fc4"
            assert read(Path(evidence["episode_receipt"]))["status"] == "unknown_process_cleanup"
    pause = [json.loads(line) for line in (NEW / "checkpoint-pauses.jsonl").read_text().splitlines()]
    assert len(pause) == 1 and pause[0]["order_index"] == 31 and (NEW / "checkpoint.request").is_file()
    assert slots[31]["case_id"] == "dev-L3-01" and slots[31]["condition"] == "B3-M1"
    attempts = list((PROJECT / "logs/inspection-eight-condition-dev-flights.v4").rglob("attempt-*"))
    assert len(attempts) == 31 and all(p.is_dir() and p.name == "attempt-00" for p in attempts)
    assert not list((RUN / "resource-requests").glob("dev-all8-pilot-v4-031*"))
    assert not list((RUN / "qualification/local-model-episodes").glob("dev-all8-pilot-v4-031*"))
    status_path = Path("/tmp/inspection-v4-cache31-native-status.v1.json")
    budget = read(status_path)["budget"]
    assert budget["known_usage"] == {"calls": 1153, "tokens": 7468127, "cost": {}}
    assert budget["reserved"] == {"calls": 0, "tokens": 0, "cost": {}}
    assert budget["unknown_usage"] == [] and budget["unresolved_usage"] == []
    companion = read(NEW / "recovery-adapter.json")
    assert companion["physical_denominator"] == 108
    assert companion["accepted030_evidence"] == read(Path("/tmp/inspection-v4-030-accepted-verify.v1.json"))["evidence"]
    for name in ("recovery-adapter.json", "checkpoint.request", "checkpoint-pauses.jsonl"):
        bind(NEW / name)
    old030 = read(OLD / "llm/raw/dev-L3-01-B4-M1-1.json")
    assert old030["status"] == "failed" and old030["attempted"] is True
    observer_dir = NEW / "cleanup-observations"
    assert not observer_dir.is_symlink() and not list(observer_dir.rglob("*"))
    for ref in [companion["adapter_source"], *companion["passive_observer"]]:
        assert sha(PROJECT / ref["path"]) == ref["sha256"]
        bind(PROJECT / ref["path"])
    assert companion["accepted024_evidence"]["cleanup_disposition"]["native_request_sha256"] == "afb193b959203b71d0eca6ffd9e7ea33a3127295806071d01f2b9e3913ce275d"
    invocations = list((NEW / "invocations").glob("*.json"))
    assert len(invocations) == 5
    for path in invocations:
        value = read(path)
        assert value["returncode"] == 0 and value["cleanup"]["status"] == "closed"
        assert value["cleanup"]["remaining_owned"] == [] and value["cleanup"]["unclaimed_new_python_orphans"] == []
        bind(path)
    bind(RUN / "recovery/pilot-v4-cleanup-024.v1/acceptance.v1.json")
    bind(Path(__file__))
    bind(RUN / "recovery/pilot-v4-cleanup-030.v1/acceptance.v1.json")
    out = RUN / "qualification/pilot-v4-cache31-prefix.v1"
    out.mkdir()
    copies = {"native-status.json": status_path,
              "accepted-verify.json": Path("/tmp/inspection-v4-030-accepted-verify.v1.json"),
              "stdout-at-barrier.jsonl": Path("/tmp/inspection-pilot-v4-continuation.v2.stdout")}
    for name in files + ["recovery-adapter.json", "checkpoint.request", "checkpoint-pauses.jsonl"]:
        copies["sdk/" + name] = NEW / name
    for _, path in rows:
        copies["sdk/" + str(path.relative_to(NEW))] = path
    for name, source in copies.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        bind(target)
    for ref in refs.values():
        assert sha(PROJECT / ref["path"]) == ref["sha256"], ref["path"]
    result = {"role": "actual same-cohort31-cache reconstruction, not new flights or G4",
              "pause": pause[0], "old_sdk_status_counts": dict(Counter(r["status"] for r in old_rows)),
              "continued_completed_callbacks": 31, "strict_cache_count": 29, "reconciled_cache_count": 2,
              "new_physics_or_model_effects": 0, "physical_denominator": 108, "remaining_original_slots": 77,
              "original_attempt_directories": 31, "budget": {k: budget[k] for k in (
                  "known_usage", "reserved", "unknown_usage", "unresolved_usage")},
              "limitations": "original unknown/abort retained; classic wrapper CLI cache reads are not new physical episodes; costs/Codex/host unmetered; fullpilot and G4 pending",
              "files": list(refs.values())}
    (out / "binding.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "verified31cache_before_order31", "binding": str(out / "binding.json"),
                      "sha256": sha(out / "binding.json"), "pause": pause[0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
