"""Verify and, when requested, release one exact paused main callback."""

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
RUNTIME = RUN / "runtime/autoresearch-document-snapshots-v2"
CHECK = RUN / "qualification/formal-main-checkpoints.v1"
MAIN = PROJECT / "logs/inspection-main-primary.v1"
EPISODES = PROJECT / "logs/inspection-formal-episodes.v1/main"
JOURNAL = PROJECT / "logs/inspection-main-space-main-rolling.v1"
MARKER = MAIN / "checkpoint.request"
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


def inspect_owner(admission, index, remaining_llm, snapshot_sha, finding, refs):
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
    bound_index = admission["native_admission_round"] - 1
    bound = rounds[bound_index]
    previous_ref = admission["previous_admission"]
    predecessor_round = read(RUN / previous_ref["path"])["native_admission_round"]
    assert type(predecessor_round) is int and 0 < predecessor_round <= bound_index
    predecessor = rounds[predecessor_round - 1]
    previous_finding = predecessor["decision"]["finding"]["id"]
    assert predecessor["request"]["evidence"].get(previous_ref["path"]) == previous_ref["sha256"]
    assert bound["decision"]["finding"].get("revises") == previous_finding
    pins = [Artifact(ref["path"], ref["sha256"]) for ref in refs]
    assert require_native_binding(control, finding, pins) == bound["request"]["request_sha256"]
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
    since = rounds[envelope["baseline_recorded_rounds"]:]
    assert len(since) + remaining_rounds <= envelope["command_counts"]["apply-round"]
    for row in since:
        assert size(row) <= envelope["regular_round_bytes"], "round record exceeds projected regular shape"
        request = row["request"]
        key = request["request_sha256"]
        request_path = RUN / "control/round-decisions" / (key + "-request.json")
        judgment = RUN / "control/round-decisions" / (key + "-judgment.json")
        assert read(request_path) == request and read(judgment) == row["decision"]
        assert request_path.stat().st_size + judgment.stat().st_size <= envelope["durable_request_decision_bytes"]
    return control["usage_ledger"], counts, datetime.fromisoformat(active["started_at"])


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


def check_ref(ref, exact=None):
    assert type(ref) is dict and set(ref) == {"path", "sha256"}
    path = RUN / ref["path"]
    assert re.fullmatch(r"[a-f0-9]{64}", ref["sha256"])
    assert path.resolve() == path and path.is_relative_to(RUN) and path.is_file()
    assert path.stat().st_nlink == 1 and (exact is None or path == exact)
    fingerprint = file_identity(path)
    assert digest(path) == ref["sha256"] and file_identity(path) == fingerprint
    return path, fingerprint


def pinned_json(ref, path):
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == ref["sha256"]
    return json.loads(raw)


def verify(index, expected, approval_name):
    identity = re.fullmatch(r"after-([0-9]{4})-main-phase-rolling-capacity-admission\.v([1-9][0-9]*)\.json", approval_name)
    assert identity, "an exact main phase capacity approval basename is required"
    first = int(identity[1])
    finding = f"inspection-main-phase-rolling-capacity-after-{first:04d}-v{identity[2]}"
    approval = CHECK / approval_name
    pin = {"path": str(approval.relative_to(RUN)), "sha256": expected}
    check_ref(pin, approval)
    admission = pinned_json(pin, approval)
    assert admission["schema"] == "inspection-main-phase-rolling-capacity-decision-v1"
    scope = admission["scope"]
    assert set(scope) == {"phase", "first", "end_exclusive", "cases"}
    end = scope["end_exclusive"]
    assert type(end) is int and type(scope["first"]) is int and scope["first"] == first
    assert scope["phase"] == "main" and 0 < first <= index < end <= 1296, "outside approved main scope"
    envelope = admission["native"]
    assert type(admission["native_admission_round"]) is int
    assert 1 <= envelope["baseline_recorded_rounds"] < admission["native_admission_round"]
    refs = [pin, admission["guard_source"], admission["projection"],
            admission["independent_review"], admission["completed_barrier"], admission["previous_admission"]]
    checked_refs = [check_ref(ref) for ref in refs]
    paths = [path for path, _ in checked_refs]
    admission_fingerprints = [fingerprint for _, fingerprint in checked_refs]
    assert len(set(paths)) == len(paths)
    assert paths[1] == Path(__file__).resolve() and paths[4] == CHECK / f"after-{first:04d}.json"
    assert paths[5].parent == CHECK
    projection = pinned_json(refs[2], paths[2])
    review = pinned_json(refs[3], paths[3])
    measured_fields = ("scope", "completed_barrier", "previous_admission", "previous_slot_release",
                       "component_allowance", "raw_attempt_bytes", "initial_remaining_slots", "initial_raw_bytes",
                       "temporary_bytes", "native", "main_only_other_bytes", "initial_floor_bytes")
    assert projection["live_volume_fits"] is True
    assert all(admission[key] == projection[key] for key in measured_fields), "approval differs from measured sizing"
    assert review["schema"] == "inspection-main-phase-capacity-review-v1"
    assert review["decision"] == "accept_main_phase_capacity_and_one_slot_guard"
    assert review["unresolved_blocking_findings"] == [] and review["reviewer"].startswith("/root/")
    assert review["guard_source"] == refs[1] and review["projection"] == refs[2] and review["scope"] == scope
    assert digest(RUN / "qualification/project-g4-owner-capacity.py") == "8ef11e2821541098c76a5441e74d7362fc949035fecdada6fc96144086819ffb"
    order = RUN / "execution/main.v1/episode-order.json"
    assert digest(order) == "8bdcfcfa60d564092feb002ed5262a7a0bb60719a49afc4bea158a4a6e69c6e7"
    slots = read(order)["slots"]
    assert [row["order_index"] for row in slots] == list(range(1296))
    assert scope["cases"] == list(dict.fromkeys(row["case_id"] for row in slots[first:end]))
    assert all(row["domain"] in {"ID", "OOD"} and row["case_id"].startswith("main-") for row in slots[first:end])
    slot = slots[index]
    packet, attempt = locate(slot)
    assert MARKER.is_file() and not MARKER.is_symlink()
    marker = file_identity(MARKER)
    assert len({p.stat().st_dev for p in (PROJECT, RUN, MAIN, EPISODES, JOURNAL, Path("/tmp"))}) == 1
    pause = latest_pause()
    assert match(pause, slot, IDENTITY[:3]) and pause["role"] == "between episodes before reservation/dispatch"
    owner = owner_identity()
    owner_started_at = None

    def still_paused():
        assert file_identity(MARKER) == marker and latest_pause() == pause
        assert not packet.exists() and not attempt.exists() and not (RUN / "datasets/holdout.v1").exists()
        assert owner_identity() == owner and MAIN.stat().st_dev == marker[0]
        assert [file_identity(p) for p in paths] == admission_fingerprints
        if owner_started_at is not None:
            assert 0 <= (datetime.now(timezone.utc) - owner_started_at).total_seconds() < 1800

    still_paused()
    receipts = MAIN / "order-receipts"
    assert {p.name for p in receipts.glob("*.json")} == {f"{i:06d}.json" for i in range(index)}
    assert match(read(receipts / f"{index - 1:06d}.json"), slots[index - 1])
    snapshot_path = CHECK / f"after-{index:04d}.json"
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot, snapshot_sha = json.loads(snapshot_bytes), sha256(snapshot_bytes).hexdigest()
    assert snapshot["pause"] == pause and snapshot["completed_callbacks"] == index and snapshot["callback_denominator"] == 1296
    assert snapshot["next_callback_untouched"] is True and snapshot["holdout_open"] is False
    prior_path = JOURNAL / f"{index - 1:06d}.json"
    prior_decision = expected
    if index == first:
        checked(admission["previous_slot_release"], prior_path)
        prior_decision = admission["previous_admission"]["sha256"]
    prior = read(prior_path)
    assert match(prior, slots[index - 1]) and prior["native_capacity_decision_sha256"] == prior_decision
    assert prior["completed_barrier_sha256"] == digest(CHECK / f"after-{index - 1:04d}.json")
    assert min(prior["initial_free_bytes"], prior["free_bytes"]) >= prior["floor_bytes"]
    assert prior["initial_free_bytes"] - prior["free_bytes"] <= 16 * 1024**2
    classic = sum(s["condition"] in CLASSIC for s in slots[index:])
    llm = 1296 - index - classic
    raw = admission["raw_attempt_bytes"]
    caps = admission["component_allowance"]
    assert set(raw) == set(caps) == {"classic", "llm"}
    assert set(caps["classic"]) == {"episode_nonwire_bytes"}
    assert set(caps["llm"]) == {"episode_nonwire_bytes", "model_originals_bytes", "archive_bytes",
                               "usage_receipt_bytes", "resource_packet_bytes"}
    assert all(type(v) is int and v > 0 for fields in caps.values() for v in fields.values())
    assert raw == {group: sum(fields.values()) for group, fields in caps.items()}
    assert admission["temporary_bytes"] == 2 * max(raw.values())
    initial_classic = sum(s["condition"] in CLASSIC for s in slots[first:])
    initial_llm = 1296 - first - initial_classic
    initial_raw = 2 * (initial_classic * raw["classic"] + initial_llm * raw["llm"])
    assert admission["initial_remaining_slots"] == {"classic": initial_classic, "llm": initial_llm}
    assert admission["initial_raw_bytes"] == initial_raw
    assert admission["main_only_other_bytes"] >= 11 * 1024**3
    assert admission["initial_floor_bytes"] == initial_raw + admission["main_only_other_bytes"] + admission["temporary_bytes"]
    floor = admission["main_only_other_bytes"] + 2 * (classic * raw["classic"] + llm * raw["llm"]) + admission["temporary_bytes"]
    ledger, counts, owner_started_at = inspect_owner(admission, index, llm, snapshot_sha, finding, refs)
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
    parser.add_argument("--approval-name", required=True)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    assert all(os.environ.get(key) == "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"))
    with (JOURNAL / ".guard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        packet, still_paused, observation = verify(args.index, args.approval_sha256, args.approval_name)
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
