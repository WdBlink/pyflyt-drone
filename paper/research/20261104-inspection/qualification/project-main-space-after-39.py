"""Read-only projection for the observed order-39 main barrier."""

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil


PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
Q = RUN / "qualification/formal-main-checkpoints.v1"
OUT = PROJECT / "logs/inspection-main-primary.v1"
ATTEMPTS = PROJECT / "logs/inspection-formal-episodes.v1/main"
COUNT = 39
CLASSIC = {"B0", "B1", "B2"}


def bound_json(path, expected_sha):
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == expected_sha
    return json.loads(raw), {"path": str(path.relative_to(RUN)), "sha256": expected_sha}


def read_json(path):
    return json.loads(path.read_bytes())


def project_capacity():
    order, order_ref = bound_json(
        RUN / "execution/main.v1/episode-order.json",
        "8bdcfcfa60d564092feb002ed5262a7a0bb60719a49afc4bea158a4a6e69c6e7",
    )
    bound, bound_ref = bound_json(
        Q / "after-0032-capacity-rebase.v1.json",
        "9fa14e6dd87497df91fbd12ee2b720355debc3603a9ae585e1bd6af9a9aeb53e",
    )
    export, export_ref = bound_json(
        Q / "after-0033-export-capacity-acceptance.v1.json",
        "2d90b212c977e6e5e5d00b521ada3fb2ddc4c10f78a5c6114fb111e042ead6ed",
    )
    checkpoint, checkpoint_ref = bound_json(
        Q / "after-0039.json",
        "f91ad503010f96048559146601dcc4e363f207047c55187e02efbd4af83fc716",
    )
    sizes, sizes_ref = bound_json(
        Q / "after-0039-observed-size-window.v1.json",
        "2ddbf64939c7b5b99a484dd487264e3c42e053b177b5c9ca78446cc368ef33ff",
    )
    assert checkpoint["completed_callbacks"] == COUNT and checkpoint["callback_denominator"] == 1296
    assert checkpoint["next_callback_untouched"] is True and checkpoint["holdout_open"] is False
    assert sizes["slot_range_zero_based_end_exclusive"] == [32, COUNT]
    assert sizes["observed_attempts"] == COUNT - 32
    assert all(not values for values in sizes["observed_positive_excess_bytes"].values())
    assert sizes["floor_unchanged_bytes"] == export["remaining_required_bytes"]
    assert (OUT / "checkpoint.request").is_file() and not (RUN / "datasets/holdout.v1").exists()
    assert len({path.stat().st_dev for path in (PROJECT, RUN, Q, OUT, ATTEMPTS)}) == 1

    slots = order["slots"]
    assert len(slots) == 1296 and [slot["order_index"] for slot in slots] == list(range(len(slots)))
    fields = ("order_index", "case_id", "condition", "repeat_ordinal")
    pause = json.loads((OUT / "checkpoint-pauses.jsonl").read_text().splitlines()[-1])
    next_slot = {key: slots[COUNT][key] for key in fields}
    assert pause == checkpoint["pause"] and all(pause[k] == next_slot[k] for k in fields[:-1])
    case, condition, repeat = (next_slot[k] for k in fields[1:])
    assert (case, condition, repeat) == ("main-OOD-L2-06", "B1", 0)
    assert not (RUN / "dispatch/main.v1" / f"{case}--{condition}--{repeat}.json").exists()
    assert not (ATTEMPTS / f"{case}-{condition}-{repeat}").exists()
    receipts = OUT / "order-receipts"
    assert {p.name for p in receipts.glob("*.json")} == {f"{i:06d}.json" for i in range(COUNT)}
    finished = Counter()
    for slot in slots[:COUNT]:
        receipt = read_json(receipts / f"{slot['order_index']:06d}.json")
        assert all(receipt[key] == slot[key] for key in fields)
        finished["classic" if slot["condition"] in CLASSIC else "llm"] += 1
    assert finished == {"classic": 7, "llm": 32}
    totals = Counter("classic" if slot["condition"] in CLASSIC else "llm" for slot in slots)
    assert totals == {"classic": 216, "llm": 1080}
    left = totals - finished
    assert sum(left.values()) == len(slots) - COUNT

    cap = bound["attempt_cap_unchanged"]
    assert cap == 2 and bound["filesystem_headroom_bytes"] == 10 * 1024**3
    other = bound["other_reserved_bytes"]
    assert other >= bound["filesystem_headroom_bytes"]
    raw, packed = bound["raw_per_attempt"], bound["packed_per_attempt"]
    tracks = dict(bound["remaining_tracks"])
    tracks["main"] = [left["classic"], left["llm"]]
    retained = 0
    phase = {}
    for name in ("main", "holdout", "clean", "ablation", "latency"):
        classic, llm = tracks[name]
        phase[name] = retained + cap * (classic * raw["classic"] + llm * raw["llm"]) + other
        retained += cap * (classic * packed["classic"] + llm * packed["llm"])
    main_packed_once = left["classic"] * packed["classic"] + left["llm"] * packed["llm"]
    main_packed_cap = cap * main_packed_once
    revised_global = max(
        retained + other + export["revised_export_allowance_bytes"], *phase.values()
    )
    remaining_after_packed_main = max(
        retained - main_packed_cap + other + export["revised_export_allowance_bytes"],
        *(peak - main_packed_cap for name, peak in phase.items() if name != "main"),
    )
    pack_temporary = cap * max(raw.values())
    proposed_main_floor = phase["main"] + pack_temporary
    free = shutil.disk_usage(PROJECT).free
    return {
        "schema": "inspection-main-phase-space-projection-v1",
        "role": "current observation for independent review, no dispatch authority",
        "at": datetime.now(timezone.utc).isoformat(),
        "inputs": [order_ref, bound_ref, export_ref, checkpoint_ref, sizes_ref],
        "next_unstarted_slot": next_slot,
        "completed_slots": dict(finished),
        "remaining_tracks_classic_llm": tracks,
        "raw_per_attempt_bytes": raw,
        "packed_per_attempt_bytes": packed,
        "attempt_cap": cap,
        "unchanged_other_reserve_bytes": other,
        "included_filesystem_headroom_bytes": bound["filesystem_headroom_bytes"],
        "cap2_phase_peaks_bytes": phase,
        "remaining_cap2_all_phase_export_peak_bytes": revised_global,
        "existing_admission_floor_bytes": export["remaining_required_bytes"],
        "same_volume_free_bytes": free,
        "main_only_margin_bytes": free - phase["main"],
        "additional_pack_temporary_reserve_bytes": pack_temporary,
        "proposed_main_only_floor_bytes": proposed_main_floor,
        "proposed_main_only_margin_bytes": free - proposed_main_floor,
        "remaining_all_phase_margin_bytes": free - revised_global,
        "existing_floor_margin_bytes": free - export["remaining_required_bytes"],
        "next_classic_slot_cap2_raw_bytes": cap * raw["classic"],
        "after_main_exact_pack_remaining_cap2_floor_bytes": remaining_after_packed_main,
        "illustrative_main_packed_once_bytes": main_packed_once,
        "illustrative_main_packed_cap2_bytes": main_packed_cap,
        "unopened_holdout": True,
        "future_cap2_readmission_required": True,
    }


if __name__ == "__main__":
    print(json.dumps(project_capacity(), indent=2, sort_keys=True))
