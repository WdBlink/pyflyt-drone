"""Preserve sealed v4 development world bytes while reducing APFS allocation."""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
Q = RUN / "qualification"
LOGS = PROJECT / "logs"
SDK = LOGS / "inspection-eight-condition-pilot.v4.continuation.v3"
DEV = LOGS / "inspection-eight-condition-dev-flights.v4"
FOLDER = Q / "formal-main-checkpoints.v1"
TOOL = LOGS / "inspection-g4-tools.v1/applesauce"
PACK = Q / "pack-closed-formal-phase.py"
sys.path.insert(0, str(PROJECT))

from eval.inspection_episode_receipts import durable_once
from eval.inspection_formal_admission import file_sha


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    return {"project_path": str(path.relative_to(PROJECT)), "sha256": file_sha(path)}


def closed_sdk(binding):
    index = SDK / "recording-completion-index.json"
    order = SDK / "episode-order.json"
    for path in (index, order):
        assert file_sha(path) == binding["files"][str(path.relative_to(PROJECT))]
    body = read(index)
    assert body["episode_denominator"] == 108 and body["schedule_sha256"] == file_sha(order)
    for group, count in (("classic", 18), ("llm", 90), ("root-means", 48)):
        result = SDK / group / "results.json"
        assert file_sha(result) == body["results_sha256"][group] == binding["files"][str(result.relative_to(PROJECT))]
        data = read(result)
        assert data["execution_status"] == "completed" and data["reported_denominator"] == data["attempted_count"] == count
        assert len(data["runs"]) == count and all(row["status"] == "completed" for row in data["runs"])
    return order, index


def main():
    assert file_sha(PACK) == "48913b00749a3cb44021edde3ac77e44f8f0a0b146945c5a9c01c0e5fb11c04d"
    assert file_sha(TOOL) == "930a613509fa1db336f3810a59c7c1747d2241630ebac8629bef651896ff0a8b"
    assert not (RUN / "datasets/holdout.v1").exists()
    main_logs = LOGS / "inspection-main-primary.v1"
    assert (main_logs / "checkpoint.request").is_file()
    assert json.loads((main_logs / "checkpoint-pauses.jsonl").read_text().splitlines()[-1])["order_index"] == 32
    assert (main_logs / "order-receipts/000031.json").is_file() and not (main_logs / "order-receipts/000032.json").exists()
    binding_path = Q / "pilot-v4-complete.v1/binding.json"
    assert file_sha(binding_path) == "d51ef4f0552cd447c5c7450d8e389c345f7f12c947a1d037cbb8d22a99f6d622"
    binding = read(binding_path)
    order, index = closed_sdk(binding)
    assert binding["physical_episodes"] == 108 and binding["strict_original_closed"] == 105
    assert binding["original_cleanup_unknown_orders_retained"] == [24, 30, 37]
    inventory = Q / "formal-capacity-final.v1/flight-storage.json"
    expected_inventory = read(Q / "formal-capacity-final.v1/capacity-with-lossless-pack.v1.json")["files"][
        str(inventory.relative_to(PROJECT))]
    assert file_sha(inventory) == expected_inventory
    samples = [s for s in read(inventory)["samples"] if s["role"] == "v4_original"]
    assert len(samples) == 108
    slots = read(order)["slots"]
    scope = []
    pack = runpy.run_path(str(PACK))["pack_file"]
    journal = FOLDER / "closed-development-world-pack.v1.jsonl"
    out = FOLDER / "closed-development-world-pack.v1.json"
    assert not journal.exists() and not out.exists()
    with ExitStack() as stack:
        for i, (sample, slot) in enumerate(zip(samples, slots, strict=True)):
            assert slot["order_index"] == i and sample["condition"] == slot["condition"]
            if i in (24, 30, 37):
                assert sample["terminal_status"] != "completed"
                continue
            assert sample["terminal_status"] == "completed"
            base = DEV / ("classic" if slot["condition"] in ("B0","B1","B2") else "")
            directory = base / f"{slot['case_id']}-{slot['condition']}-{slot['repeat_ordinal']}"
            attempt = directory / "attempt-00"
            assert attempt.relative_to(PROJECT).as_posix() == sample["attempt"]
            lock = directory / ".lock"
            assert lock.is_file() and not lock.is_symlink()
            fcntl.flock(stack.enter_context(lock.open("rb")), fcntl.LOCK_EX | fcntl.LOCK_NB)
            terminal = attempt / "terminal.json"
            assert file_sha(terminal) == sample["terminal_sha256"] == binding["files"][str(terminal.relative_to(PROJECT))]
            data = read(terminal)
            world = attempt / "flight/world.json"
            expected = data.get("physical_validation", data.get("evidence"))["world_sha256"]
            assert data["status"] == "completed" and world.is_relative_to(DEV)
            assert world.stat().st_dev == DEV.stat().st_dev and file_sha(world) == expected
            scope.append((i, terminal, world, expected))
        assert len(scope) == 105
        before = shutil.disk_usage(DEV).free
        assert before > 2 * max(world.stat().st_size for _,_,world,_ in scope)
        allocated = 0
        changed = 0
        with journal.open("x") as stream:
            for i,terminal,world,expected in scope:
                result = pack(world, expected, TOOL, DEV)
                allocated += result["before_allocated"] - result["after_allocated"]
                changed += result["status"] == "compressed"
                stream.write(json.dumps({"order":i,"terminal":ref(terminal),"world":ref(world),**result})+"\n")
                stream.flush()
                os.fsync(stream.fileno())
        for _,terminal,world,expected in scope:
            assert file_sha(world) == expected and file_sha(terminal) == binding["files"][str(terminal.relative_to(PROJECT))]
        closed_sdk(binding)
        after = shutil.disk_usage(DEV).free
        value = {"role":"post-completion development physical allocation only; all 105 original world SHA verified",
                 "scope":"closed pilot v4 original strict receipts; three historical unknown-cleanup attempts excluded",
                 "sdk_binding":ref(binding_path),"sdk_order":ref(order),"sdk_index":ref(index),
                 "scope_count":len(scope),"compressed_count":changed,"logical_shas_verified":len(scope),
                 "allocated_block_savings_bytes":allocated,"free_before_bytes":before,"free_after_bytes":after,
                 "measured_free_change_bytes":after-before,"journal":ref(journal),"tool":ref(TOOL),
                 "formal_checkpoint_still_present":(main_logs/"checkpoint.request").is_file()}
        assert value["formal_checkpoint_still_present"]
        durable_once(out,value)
    print(json.dumps({"artifact":{"path":str(out.relative_to(RUN)),"sha256":file_sha(out)},**value}))


if __name__ == "__main__":
    main()
