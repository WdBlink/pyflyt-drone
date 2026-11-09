"""Verify and, when requested, release one exact paused main callback."""

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
RUNTIME = RUN / "runtime/autoresearch-document-snapshots-v2"
CHECK = RUN / "qualification/formal-main-checkpoints.v1"
APPROVAL = CHECK / "after-0072-three-id-roots-rolling-capacity-admission.v1.json"
MAIN = PROJECT / "logs/inspection-main-primary.v1"
EPISODES = PROJECT / "logs/inspection-formal-episodes.v1/main"
JOURNAL = PROJECT / "logs/inspection-main-space-main-rolling.v1"
MARKER = MAIN / "checkpoint.request"
SCOPE = ("main-ID-L3-15", "main-ID-L2-08", "main-ID-L1-03")
FINDING = "inspection-main-three-id-roots-rolling-capacity-v1"
CLASSIC = {"B0", "B1", "B2"}
IDENTITY = ("order_index", "case_id", "condition", "repeat_ordinal")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def match(row, slot, keys=IDENTITY):
    return all(row[k] == slot[k] for k in keys)


def checked(ref, path):
    assert ref["project_path"] == str(path.relative_to(PROJECT))
    assert path.resolve() == path and path.is_file() and path.stat().st_nlink == 1
    with path.open("rb") as stream:
        raw = stream.read(ref["size_bytes"] + 1)
    assert len(raw) == ref["size_bytes"] and sha256(raw).hexdigest() == ref["sha256"]
    return raw


def file_identity(path):
    info = path.stat(follow_symlinks=False)
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def owner_identity():
    return tuple(file_identity(RUN / ".paper" / name) for name in ("control.json", "checkpoint.json"))


def latest_pause():
    return json.loads((MAIN / "checkpoint-pauses.jsonl").read_text().splitlines()[-1])


def locate(slot):
    case, cond, repeat = (slot[k] for k in IDENTITY[1:])
    return RUN / "dispatch/main.v1" / f"{case}--{cond}--{repeat}.json", EPISODES / f"{case}-{cond}-{repeat}"


def df_bytes():
    lines = subprocess.check_output(["df", "-Pk", str(MAIN)], text=True).splitlines()
    assert len(lines) == 2, "df did not return one mounted volume"
    fields = lines[1].split()
    return (fields[0], MAIN.stat().st_dev), int(fields[3]) * 1024


def inspect_owner(admission, index, remaining_llm, snapshot_sha):
    sys.path.insert(0, str(PROJECT))
    from eval.inspection_formal_admission import Artifact, native_owner_reader, require_native_binding

    home = RUN / ".paper"
    envelope = admission["native"]
    for name, limit in envelope["owner_pair_bytes"].items():
        assert (home / f"{name}.json").stat().st_size <= limit, "owner file outgrew projection"
    documents = list((home / "round-documents").glob("*.json"))
    assert documents and max(p.stat().st_size for p in documents) <= envelope["document_bytes"]
    control = native_owner_reader(RUNTIME)(RUN)
    rounds, active = control["rounds"], control["active_round"]
    assert control["pending_round"] is None and control["stale_count"] == 0 and not control["task_stop"]
    assert active["stage"] == "experiment" and active["id"] == f"round-{len(rounds) + 1}"
    assert 0 <= (datetime.now(timezone.utc) - datetime.fromisoformat(active["started_at"])).total_seconds() < 1800, "record native progress at this safe boundary"
    assert all(row["status"] == "known" for row in control["usage_ledger"].values()), "unknown or held usage"
    bound = rounds[admission["native_admission_round"] - 1]
    pin = Artifact(str(APPROVAL.relative_to(RUN)), digest(APPROVAL))
    assert require_native_binding(control, FINDING, [pin]) == bound["request"]["request_sha256"]
    recent = rounds[-1]["request"]["evidence"]
    current = str((CHECK / f"after-{index:04d}.json").relative_to(RUN))
    previous = CHECK / f"after-{index - 1:04d}.json"
    assert recent.get(current) == snapshot_sha or (
        index % 2 == 1 and recent.get(str(previous.relative_to(RUN))) == digest(previous)
    ), "current or immediately preceding safe boundary must be recorded in native state"
    remaining_rounds = (1296 - index + 1) // 2
    counts = Counter()
    wire = envelope["wire"]
    trace_dir = RUN.with_name(RUN.name + "-cli")
    cutoff = envelope["baseline_last_cli_trace"]
    assert (trace_dir / cutoff).is_file()
    for path in trace_dir.glob("*.json"):
        if path.name <= cutoff:
            continue
        row = read(path)
        cmd = row["command"]
        assert cmd in wire and row["exit_code"] == 0, "new native command or failure needs review"
        assert path.stat().st_size <= wire[cmd]["trace"], "actual native trace exceeds projection"
        stdout_bytes = len((json.dumps(row["response"], indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
        assert stdout_bytes <= wire[cmd]["stdout"], "actual native stdout exceeds projection"
        counts[cmd] += 1
    for cmd, ceiling in envelope["command_counts"].items():
        further = 2 * remaining_llm if cmd in ("budget-reserve", "budget-settle") else remaining_rounds
        assert counts[cmd] + further <= ceiling, "remaining main work no longer fits native command shape"
    for event in control["history"][envelope["baseline_history_count"]:]:
        if "operation" not in event:
            assert event["event"] == "session_started"
            continue
        obj = event["result_ref"]
        path = RUN / obj["path"]
        assert path.parent == home / "operation-results" and digest(path) == obj["sha256"]
        result = read(path)["result"]
        cmd = {"reserved": "budget-reserve", "known": "budget-settle", "round_recorded": "apply-round"}[result["status"]]
        assert path.stat().st_size <= wire[cmd]["result_object"], "stored native result exceeds projection"
    size = runpy.run_path(str(RUN / "qualification/project-g4-owner-capacity.py"))["size"]
    since = rounds[admission["native_admission_round"] - 1:]
    assert len(since) + remaining_rounds <= envelope["command_counts"]["apply-round"]
    for row in since:
        assert size(row) <= envelope["regular_round_bytes"], "round record exceeds projected regular shape"
        request = row["request"]
        rid = int(request["round"]["id"].removeprefix("round-"))
        request_path = Path(f"/tmp/inspection-round{rid}-request.v1.json")
        judgment = RUN / "control/round-decisions" / (request["request_sha256"] + "-judgment.json")
        assert read(request_path) == request and read(judgment) == row["decision"]
        assert request_path.stat().st_size + judgment.stat().st_size <= envelope["regular_request_decision_bytes"]
    return control["usage_ledger"], counts


def inspect_previous(snapshot, slot, ledger, admission):
    prev = snapshot["evidence_rows"][-1]
    assert match(prev, slot) and prev["physical_seconds"] == 600 and prev["steps"] == 144000
    assert 0 <= prev["max_clock_lag_seconds"] <= .1
    _, directory = locate(slot)
    llm = slot["condition"] not in CLASSIC
    key = "llm" if llm else "classic"
    measure = RUN / "qualification/measure-g4-storage.py"
    assert digest(measure) == "31ce460eaea8a278de4f9cbea19930fd5dae7ac46dd3e910ad21b96df8d65043"
    sample = runpy.run_path(str(measure))["sample"]
    attempts = sorted(directory.glob("attempt-*"))
    assert len(attempts) == len(prev["attempts"]) and 1 <= len(attempts) <= 2
    sdk_receipt = MAIN / "order-receipts" / f"{slot['order_index']:06d}.json"
    receipt = json.loads(checked(prev["sdk_order_receipt"], sdk_receipt))
    raw_path = MAIN / key / "raw" / (directory.name + ".json")
    assert receipt["raw"] == str(raw_path.relative_to(MAIN)) and match(receipt, slot)
    raw = json.loads(checked(prev["sdk_raw"], raw_path))
    checked(prev["world"], attempts[-1] / "flight/world.json")
    assert raw["status"] == "completed" and raw["evidence"]["score_recount_exact"] is True
    result = []
    for path, ref in zip(attempts, prev["attempts"]):
        terminal = path / "terminal.json"
        checked(ref, terminal)
        reservation = read(path / "intent.json").get("reservation")
        assert bool(reservation) == llm and (not llm or ledger[reservation]["status"] == "known")
        row = sample(path, "rolling_main_closed", slot["level"], slot["condition"], reservation)
        assert all(row[field] <= cap for field, cap in admission["component_allowance"][key].items()), "attempt exceeded component allowance"
        if llm:
            accounted = ledger[reservation]
            usage = RUN / "qualification/local-model-episodes" / reservation / "usage-receipt.json"
            assert accounted["receipt"] == str(usage.relative_to(RUN)) and digest(usage) == accounted["receipt_sha256"]
            for label in ("reserve", "settle"):
                assert row["native_stdout_bytes"][f"{label}-owner.json"] <= admission["native"]["wire"][f"budget-{label}"]["stdout"]
        result.append(row)
    assert result[-1]["terminal_status"] == "completed"
    return result


def verify(index, expected):
    assert 72 <= index < 126, "scope ends before order126"
    approval_bytes = APPROVAL.read_bytes()
    assert sha256(approval_bytes).hexdigest() == expected, "capacity decision identity changed"
    admission = json.loads(approval_bytes)
    assert admission["scope"] == {"phase": "main", "first": 72, "end_exclusive": 126, "cases": list(SCOPE)}
    assert admission["native_admission_round"] == 134
    assert digest(Path(__file__)) == admission["guard_source"]["sha256"]
    assert digest(RUN / "qualification/project-g4-owner-capacity.py") == "8ef11e2821541098c76a5441e74d7362fc949035fecdada6fc96144086819ffb"
    order = RUN / "execution/main.v1/episode-order.json"
    assert digest(order) == "8bdcfcfa60d564092feb002ed5262a7a0bb60719a49afc4bea158a4a6e69c6e7"
    slots = read(order)["slots"]
    assert [row["order_index"] for row in slots] == list(range(1296))
    assert [row["case_id"] for row in slots[72:126]] == [case for case in SCOPE for _ in range(18)]
    assert all(row["domain"] == "ID" for row in slots[72:126])
    slot = slots[index]
    packet, attempt = locate(slot)
    assert MARKER.is_file() and not MARKER.is_symlink()
    marker = file_identity(MARKER)
    assert len({p.stat().st_dev for p in (PROJECT, RUN, MAIN, EPISODES, JOURNAL, Path("/tmp"))}) == 1
    pause = latest_pause()
    assert match(pause, slot, IDENTITY[:3]) and pause["role"] == "between episodes before reservation/dispatch"
    owner = owner_identity()

    def still_paused():
        assert file_identity(MARKER) == marker and latest_pause() == pause
        assert not packet.exists() and not attempt.exists() and not (RUN / "datasets/holdout.v1").exists()
        assert owner_identity() == owner and MAIN.stat().st_dev == marker[0]

    still_paused()
    receipts = MAIN / "order-receipts"
    assert {p.name for p in receipts.glob("*.json")} == {f"{i:06d}.json" for i in range(index)}
    assert match(read(receipts / f"{index - 1:06d}.json"), slots[index - 1])
    snapshot_path = CHECK / f"after-{index:04d}.json"
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot, snapshot_sha = json.loads(snapshot_bytes), sha256(snapshot_bytes).hexdigest()
    assert snapshot["pause"] == pause and snapshot["completed_callbacks"] == index and snapshot["callback_denominator"] == 1296
    assert snapshot["next_callback_untouched"] is True and snapshot["holdout_open"] is False
    if index > 72:
        prior = read(JOURNAL / f"{index - 1:06d}.json")
        assert match(prior, slots[index - 1]) and prior["native_capacity_decision_sha256"] == expected
        assert prior["completed_barrier_sha256"] == digest(CHECK / f"after-{index - 1:04d}.json")
        assert min(prior["initial_free_bytes"], prior["free_bytes"]) >= prior["floor_bytes"]
        assert prior["initial_free_bytes"] - prior["free_bytes"] <= 16 * 1024**2
    classic = sum(s["condition"] in CLASSIC for s in slots[index:])
    llm = 1296 - index - classic
    raw = admission["raw_attempt_bytes"]
    assert raw == {"classic": 29546687, "llm": 44488936} and admission["temporary_bytes"] == 88977872
    floor = admission["main_only_other_bytes"] + 2 * (classic * raw["classic"] + llm * raw["llm"]) + admission["temporary_bytes"]
    ledger, counts = inspect_owner(admission, index, llm, snapshot_sha)
    attempts = inspect_previous(snapshot, slots[index - 1], ledger, admission)
    first_device, first = df_bytes()
    time.sleep(2)
    device, second = df_bytes()
    assert first_device == device and device[1] == marker[0]
    assert min(first, second) >= floor and first - second <= 16 * 1024**2, (first, second, floor)
    still_paused()
    return packet, still_paused, {
        "role": "verified pre-dispatch capacity; not a flight result",
        "at": datetime.now(timezone.utc).isoformat(), **{k: slot[k] for k in IDENTITY},
        "remaining_main_slots": {"classic": classic, "llm": llm},
        "completed_barrier_sha256": snapshot_sha, "native_capacity_decision_sha256": expected,
        "initial_free_bytes": first, "free_bytes": second, "floor_bytes": floor,
        "volume": device[0], "device": device[1],
        "closed_slot_attempt_sizes": attempts, "new_native_command_counts": dict(counts),
    }


def main():
    if sys.flags.optimize:
        raise SystemExit("guard requires Python assertions")
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    assert all(os.environ.get(key) == "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"))
    with (JOURNAL / ".guard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        packet, still_paused, observation = verify(args.index, args.approval_sha256)
        if not args.release:
            print(json.dumps(observation), flush=True)
            return
        with (JOURNAL / f"{args.index:06d}.json").open("x") as stream:
            stream.write(json.dumps(observation) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        def interrupt(sig, _frame):
            raise SystemExit(128 + sig)

        for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
            signal.signal(sig, interrupt)
        still_paused()
        try:
            MARKER.unlink()
            print(f"released order {args.index}, free {observation['free_bytes']} B", flush=True)
            deadline = time.monotonic() + 180
            while not packet.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError("actual dispatch not observed; inspect paused state before continuing")
                time.sleep(.2)
        finally:
            try:
                MARKER.touch(exist_ok=False)
            except FileExistsError:
                assert MARKER.is_file() and not MARKER.is_symlink()
        print(f"re-armed after actual dispatch of order {args.index}", flush=True)


if __name__ == "__main__":
    main()
