"""Reduce allocation for the 18 SHA-bound completed classic development worlds."""

from contextlib import ExitStack
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import shutil


PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
Q = RUN / "qualification"
SNAPSHOT = Q / "classic-six-root-cohort.v3"
CHECKPOINTS = Q / "formal-main-checkpoints.v1"
LOGS = PROJECT / "logs"
DEV = LOGS / "inspection-classic-dev-flights.v1"
SDK = LOGS / "inspection-classic-dev-recording.v3"
MAIN = LOGS / "inspection-main-primary.v1"
TOOL = LOGS / "inspection-g4-tools.v1/applesauce"
PACK = Q / "pack-closed-formal-phase.py"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def ref(path):
    return {"project_path": str(path.relative_to(PROJECT)), "sha256": digest(path)}


def paused():
    assert (MAIN / "checkpoint.request").is_file() and not (RUN / "datasets/holdout.v1").exists()
    pause = json.loads((MAIN / "checkpoint-pauses.jsonl").read_text().splitlines()[-1])
    assert (pause["order_index"], pause["case_id"], pause["condition"]) == (51, "main-OOD-L2-06", "B4-M2")
    assert not (RUN / "dispatch/main.v1/main-OOD-L2-06--B4-M2--2.json").exists()
    assert not (LOGS / "inspection-formal-episodes.v1/main/main-OOD-L2-06-B4-M2-2").exists()
    assert digest(CHECKPOINTS / "after-0051.json") == "c5cd479e2e8ee349ad3c29627360a893b05341ed65b11b863f4ecf60d976264b"


def frozen_sdk():
    binding = SNAPSHOT / "binding.json"
    assert digest(binding) == "785c8dae47a074aff6c93d6cf20ec6247d3fda4de9c5b5b915da03a8a3061ba6"
    values = read(binding)
    assert values["completed_slots"] == 18
    for name, expected in values["files"].items():
        assert Path(name).name == name
        assert digest(SNAPSHOT / name) == digest(SDK / name) == expected
    result = read(SDK / "results.json")
    assert result["execution_status"] == "completed" and result["attempted_count"] == 18
    assert len(result["runs"]) == 18 and all(row["status"] == "completed" for row in result["runs"])
    return binding, result


def close_out(stream, value):
    stream.write(json.dumps(value, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def main():
    paused()
    assert digest(PACK) == "48913b00749a3cb44021edde3ac77e44f8f0a0b146945c5a9c01c0e5fb11c04d"
    assert digest(TOOL) == "930a613509fa1db336f3810a59c7c1747d2241630ebac8629bef651896ff0a8b"
    assert len({p.stat().st_dev for p in (DEV, SDK, RUN, MAIN)}) == 1
    binding, result = frozen_sdk()
    slots = read(SDK / "schedule.json")["slots"]
    assert len(slots) == 18 and [s["order_index"] for s in slots] == list(range(18))
    result_rows = {row["id"]: row for row in result["runs"]}
    pack = runpy.run_path(str(PACK))["pack_file"]
    journal = CHECKPOINTS / "closed-classic-development-world-pack.v1.jsonl"
    out = CHECKPOINTS / "closed-classic-development-world-pack.v1.json"
    assert not journal.exists() and not out.exists()
    scope = []
    skipped_attempts = []
    with ExitStack() as stack:
        for slot in slots:
            case, condition, repeat = (slot[k] for k in ("case_id", "condition", "repeat_ordinal"))
            key = f"{case}-{condition}-{repeat}"
            assert condition in {"B0", "B1", "B2"} and repeat == 0 and case.startswith("dev-")
            result_row = result_rows[key]
            assert (result_row["case_id"], result_row["condition"], result_row["seed"]) == (case, condition, repeat)
            assert result_row["raw"] == f"raw/{key}.json"
            raw = SDK / result_row["raw"]
            payload = read(raw)
            ev = payload["evidence"]
            assert payload["status"] == "completed" and payload["value"] == result_row["value"]
            assert all(ev[k] == slot[k] for k in ("order_index", "case_id", "condition", "repeat_ordinal"))
            directory = DEV / key
            assert directory.is_dir() and not directory.is_symlink()
            lock = directory / ".lock"
            assert lock.is_file() and not lock.is_symlink()
            fcntl.flock(stack.enter_context(lock.open("rb")), fcntl.LOCK_EX | fcntl.LOCK_NB)
            terminal = Path(ev["episode_receipt"])
            assert terminal.parent.parent == directory and terminal.name == "terminal.json"
            assert terminal.parent.name in ("attempt-00", "attempt-01") and not terminal.is_symlink()
            attempts = sorted(directory.glob("attempt-*"))
            assert attempts[-1] == terminal.parent and len(attempts) == ev["actual_attempts"]
            assert digest(terminal) == ev["episode_receipt_sha256"]
            t = read(terminal)
            assert t["status"] == "completed" and t["Y"] == payload["value"]
            plan = read(directory / "plan.json")
            plan_sha = sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
            assert t["plan_sha256"] == ev["plan_sha256"] == plan_sha
            world = terminal.parent / "flight/world.json"
            expected = t["evidence"]["world_sha256"]
            assert not world.is_symlink() and world.stat().st_nlink == 1
            assert world.stat().st_dev == DEV.stat().st_dev
            assert digest(world) == t["record_hashes"]["flight/world.json"] == expected
            scope.append((slot["order_index"], raw, terminal, world, expected))
            for attempt in attempts[:-1]:
                prior = attempt / "terminal.json"
                assert read(prior)["status"] != "completed"
                skipped_attempts.extend(ref(path) for path in (prior, attempt / "flight/world.json"))
        assert len(scope) == 18 and len(skipped_attempts) == 2
        before = shutil.disk_usage(DEV).free
        assert before > 2 * max(x[3].stat().st_size for x in scope)
        frozen_records = [(ref(raw), ref(terminal)) for _, raw, terminal, _, _ in scope]
        saved = 0
        compressed = 0
        with journal.open("x") as stream:
            for (i, raw, terminal, world, expected), (raw_ref, terminal_ref) in zip(scope, frozen_records, strict=True):
                packed = pack(world, expected, TOOL, DEV)
                saved += packed["before_allocated"] - packed["after_allocated"]
                compressed += packed["status"] == "compressed"
                close_out(stream, {"order_index": i, "raw": raw_ref, "terminal": terminal_ref,
                                   "world": ref(world), **packed})
        paused()
        frozen_sdk()
        assert all(digest(world) == expected for _, _, _, world, expected in scope)
        assert all(ref(raw) == old_raw and ref(terminal) == old_terminal
                   for (_, raw, terminal, _, _), (old_raw, old_terminal)
                   in zip(scope, frozen_records, strict=True))
        assert all(digest(PROJECT / old["project_path"]) == old["sha256"] for old in skipped_attempts)
        after = shutil.disk_usage(DEV).free
        value = {"role": "closed classical development APFS allocation only; all original logical bytes preserved",
                 "frozen_development_binding": ref(binding), "packer": ref(PACK), "tool": ref(TOOL),
                 "attempt_journal": ref(journal), "scope_count": len(scope),
                 "compressed_world_count": compressed, "logical_sha_verified": len(scope),
                 "original_failed_attempt_untouched": skipped_attempts,
                 "allocated_bytes_reduction": saved, "same_volume_free_before": before,
                 "same_volume_free_after": after, "observed_free_bytes_change": after - before,
                 "formal_effect": "none; order51 paused and untouched"}
        with out.open("x") as stream:
            close_out(stream, value)
        print(json.dumps({"artifact": ref(out), "saved_allocated_bytes": saved, "same_volume_free_after": after,
                          "compressed_world_count": compressed, "verified_count": len(scope)}))


if __name__ == "__main__":
    main()
