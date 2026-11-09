"""Rebase the accepted main storage shape at a real scope boundary."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys
import time

PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
Q = RUN / "qualification"
CHECK = Q / "formal-main-checkpoints.v1"
MAIN = PROJECT / "logs/inspection-main-primary.v1"
JOURNAL = PROJECT / "logs/inspection-main-space-main-rolling.v1"
Q147 = CHECK / "after-0091-classic-buffered-rolling-capacity-admission.v1.json"
Q147_SHA = "9ddf3886942d3c84492758941b11da98b1f2e1db43f19c1a3088e0075c772e8d"
Q147_FINDING = "inspection-main-classic-buffered-rolling-capacity-v1"
BASE = Q / "guard-formal-main-rolling.py"
BASE_SHA = "2143908dffccf08be68542465874567beed52aac6af35eb92ae345f7df932feb"
ORDER = RUN / "execution/main.v1/episode-order.json"
ORDER_SHA = "8bdcfcfa60d564092feb002ed5262a7a0bb60719a49afc4bea158a4a6e69c6e7"
CLASSIC = {"B0", "B1", "B2"}
KEYS = ("order_index", "case_id", "condition", "repeat_ordinal")


def read(path):
    return json.loads(path.read_bytes())


def hash_file(path):
    return sha256(path.read_bytes()).hexdigest()


def project_ref(path):
    return {"project_path": str(path.relative_to(PROJECT)), "sha256": hash_file(path),
            "size_bytes": path.stat().st_size}


def run_ref(path):
    return {"path": str(path.relative_to(RUN)), "sha256": hash_file(path)}


def check_identity(value, slot):
    assert all(value[key] == slot[key] for key in KEYS)


def inspect_boundary(index, barrier_sha, status_sha):
    assert index == 126, "this rebase is scoped to the actual end of round 147 authority"
    assert hash_file(BASE) == BASE_SHA and hash_file(Q147) == Q147_SHA and hash_file(ORDER) == ORDER_SHA
    guard = runpy.run_path(str(BASE))
    status_path = JOURNAL / f"native-status-at-{index:04d}.json"
    status = read(status_path)
    assert hash_file(status_path) == status_sha and status["stale"] == []
    budget = status["budget"]
    assert budget["unknown_usage"] == budget["unresolved_usage"] == [] and budget["reserved"]["calls"] == 0
    old = read(Q147)
    assert old["scope"]["end_exclusive"] == index
    slots = read(ORDER)["slots"]
    assert [row["order_index"] for row in slots] == list(range(1296))
    marker = MAIN / "checkpoint.request"
    pause = guard["latest_pause"]()
    assert marker.is_file() and pause["order_index"] == index and not (RUN / "datasets/holdout.v1").exists()
    snapshot_path = CHECK / f"after-{index:04d}.json"
    assert hash_file(snapshot_path) == barrier_sha
    snapshot = read(snapshot_path)
    assert snapshot["pause"] == pause and snapshot["completed_callbacks"] == index
    assert snapshot["next_callback_untouched"] is True and snapshot["holdout_open"] is False
    packet, attempt = guard["locate"](slots[index])
    assert not packet.exists() and not attempt.exists()
    receipts = MAIN / "order-receipts"
    assert {p.name for p in receipts.glob("*.json")} == {f"{j:06d}.json" for j in range(index)}
    for slot in slots[:index]:
        check_identity(read(receipts / f"{slot['order_index']:06d}.json"), slot)
    for path in (PROJECT, RUN, MAIN, JOURNAL, Path("/tmp"), guard["EPISODES"]):
        assert path.stat().st_dev == marker.stat().st_dev
    return slots, snapshot, snapshot_path, guard, old, status_path, budget["known_usage"]

def measure(index, barrier_sha, status_sha):
    slots, snapshot, snap_path, guard, old, status_path, usage = inspect_boundary(index, barrier_sha, status_sha)
    mark = guard["file_identity"](MAIN / "checkpoint.request")
    owner = guard["owner_identity"]()
    check_owner = guard["inspect_owner"]
    check_owner.__globals__.update(APPROVAL=Q147, FINDING=Q147_FINDING)
    remaining = Counter("classic" if row["condition"] in CLASSIC else "llm" for row in slots[index:])
    assert dict(remaining) == {"classic": 195, "llm": 975}
    ledger, native_counts = check_owner(old, index, remaining["llm"], barrier_sha)
    assert all(row["status"] == "known" for row in ledger.values())
    actual = {key: sum(row["reported_usage"].get(key, 0) for row in ledger.values()) for key in ("calls", "tokens")}
    assert usage == {**actual, "cost": {}}
    recent = []
    journal_refs = []
    for closed in range(old["scope"]["first"], index):
        path = JOURNAL / f"{closed:06d}.json"
        row = read(path)
        check_identity(row, slots[closed])
        assert row["native_capacity_decision_sha256"] == Q147_SHA
        assert row["completed_barrier_sha256"] == hash_file(CHECK / f"after-{closed:04d}.json")
        assert min(row["initial_free_bytes"], row["free_bytes"]) >= row["floor_bytes"]
        assert row["initial_free_bytes"] - row["free_bytes"] <= 16 * 1024**2
        recent.extend(row["closed_slot_attempt_sizes"])
        journal_refs.append(project_ref(path))
    probe = {**old, "component_allowance": {
        g: {k: 2**40 for k in fields} for g, fields in old["component_allowance"].items()}}
    recent += guard["inspect_previous"](snapshot, slots[index - 1], ledger, probe)
    assert sum(row["terminal_status"] == "completed" for row in recent) == index - old["scope"]["first"] + 1
    samples = {group: [row for row in recent if ("reservation" in row) == (group == "llm")]
               for group in old["component_allowance"]}
    maxima = {g: {k: max(row[k] for row in samples[g]) for k in fields}
              for g, fields in old["component_allowance"].items()}
    caps = {g: {k: max(old_value, (112 * max(maxima[g][k], old["observed_component_maxima"][g][k]) + 99) // 100)
                for k, old_value in fields.items()} for g, fields in old["component_allowance"].items()}
    raw = {group: sum(fields.values()) for group, fields in caps.items()}
    total = 2 * sum(raw[group] * count for group, count in remaining.items())
    temporary = 2 * max(raw.values())
    from eval.inspection_formal_admission import native_owner_reader

    control = native_owner_reader(guard["RUNTIME"])(RUN)
    native = {**old["native"], "durable_request_decision_bytes": 32 * 1024}
    observed_durable = []
    for row in control["rounds"][native["baseline_recorded_rounds"]:]:
        request = row["request"]
        stem = RUN / "control/round-decisions" / request["request_sha256"]
        req, decision = Path(str(stem) + "-request.json"), Path(str(stem) + "-judgment.json")
        assert read(req) == request and read(decision) == row["decision"]
        observed_durable.append(req.stat().st_size + decision.stat().st_size)
    assert observed_durable and max(observed_durable) <= native["durable_request_decision_bytes"]
    extra_durable = native["durable_request_decision_bytes"] * native["command_counts"]["apply-round"]
    other = old["main_only_other_bytes"] + extra_durable
    floor = total + temporary + other
    volume1, free1 = guard["df_bytes"]()
    time.sleep(2)
    volume2, free2 = guard["df_bytes"]()
    assert volume1 == volume2 and volume1[1] == mark[0] and free1 - free2 <= 16 * 1024**2
    assert mark == guard["file_identity"](MAIN / "checkpoint.request") and owner == guard["owner_identity"]()
    assert guard["latest_pause"]() == snapshot["pause"]
    packet, attempt = guard["locate"](slots[index])
    assert not packet.exists() and not attempt.exists() and not (RUN / "datasets/holdout.v1").exists()
    return dict(
        role="actual read-only capacity rebase at the order-126 pause; no dispatch authority",
        scope={"phase":"main","first":index,"end_exclusive":len(slots),
               "cases":list(dict.fromkeys(s["case_id"] for s in slots[index:]))},
        untouched_next_slot=slots[index], completed_barrier=run_ref(snap_path),
        native_status=project_ref(status_path), known_usage=usage,
        previous_admission=run_ref(Q147), order=run_ref(ORDER),
        previous_slot_release=journal_refs[-1], observed_slot_guard_refs=journal_refs,
        observed_attempts=len(recent), recent_component_maxima=maxima,
        raw_buffer={"observed_margin_percent":12,"never_reduces_prior_component_caps":True},
        component_allowance=caps, raw_attempt_bytes=raw,
        initial_remaining_slots=dict(remaining), initial_raw_bytes=total, temporary_bytes=temporary,
        native=native, observed_native_counts=dict(native_counts), main_only_other_bytes=other,
        prior_main_only_other_bytes=old["main_only_other_bytes"], extra_durable_round_file_bytes=extra_durable,
        observed_max_durable_request_decision_bytes=max(observed_durable),
        initial_floor_bytes=floor, volume=volume1[0], device=volume1[1], observed_free_bytes=[free1,free2],
        margin_bytes=min(free1,free2)-floor, live_volume_fits=min(free1,free2)>=floor,
        limitations=["The native envelope and fixed headroom are inherited from the verified full-main sizing shape.",
                     "The observed component sizes and margin are workload allowances checked before every next dispatch.",
                     "No new experiment, storage packing, holdout opening, scientific analysis or write stage."],
    )

def main():
    if sys.flags.optimize:
        raise SystemExit("projection requires Python assertions")
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int)
    parser.add_argument("--barrier-sha256", required=True)
    parser.add_argument("--status-sha256", required=True)
    args = parser.parse_args()
    report = measure(args.index, args.barrier_sha256, args.status_sha256)
    report["source"] = run_ref(Path(__file__).resolve())
    path = CHECK / f"after-{args.index:04d}-main-phase-capacity-sizing.v1.json"
    with path.open("x") as output:
        json.dump(report, output, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({"artifact":run_ref(path), "source":report["source"],
                      "initial_floor_bytes":report["initial_floor_bytes"], "observed_free_bytes":report["observed_free_bytes"],
                      "margin_bytes":report["margin_bytes"], "live_volume_fits":report["live_volume_fits"]}))


if __name__ == "__main__":
    main()
