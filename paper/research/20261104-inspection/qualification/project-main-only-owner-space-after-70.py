"""Read the real owner and re-encode conservative main-only workload shapes."""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import shutil
import sys
import time


PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
Q = RUN / "qualification"
CHECK = Q / "formal-main-checkpoints.v1"
MAIN = PROJECT / "logs/inspection-main-primary.v1"
SOURCE = Q / "project-g4-owner-capacity.py"
PREVIOUS = Q / "formal-capacity-final.v1/native-storage-projection.json"
CLASSIC = {"B0", "B1", "B2"}
COUNT = 70


def read(path):
    return json.loads(path.read_bytes())


def main(status_path):
    t0 = time.monotonic()
    impl = runpy.run_path(str(SOURCE))
    file_hash = impl["file_hash"]
    assert file_hash(SOURCE) == "8ef11e2821541098c76a5441e74d7362fc949035fecdada6fc96144086819ffb"
    assert file_hash(PREVIOUS) == "6cee4dd0dd3978facdc9b1f3bce7c7c411c33c09abdca8710c56b2aca588fb0f"
    assert file_hash(status_path) == "d1c00bb0672f43752da2decdea34f32387d17c6d9dbe7c9f31bd95867710096f"
    assert file_hash(CHECK / "after-0070.json") == "2e4ee66fffa15429fc98b57e7699191ac0856d0ec9dedfabd97e32b56239e632"
    order = RUN / "execution/main.v1/episode-order.json"
    assert file_hash(order) == "8bdcfcfa60d564092feb002ed5262a7a0bb60719a49afc4bea158a4a6e69c6e7"
    assert (MAIN / "checkpoint.request").is_file() and not (RUN / "datasets/holdout.v1").exists()
    assert not (RUN / "dispatch/main.v1/main-ID-L2-13--B4-M2--2.json").exists()
    assert not (PROJECT / "logs/inspection-formal-episodes.v1/main/main-ID-L2-13-B4-M2-2").exists()
    assert MAIN.stat().st_dev == RUN.stat().st_dev == PROJECT.stat().st_dev
    checkpoint = read(CHECK / "after-0070.json")
    pause = json.loads((MAIN / "checkpoint-pauses.jsonl").read_text().splitlines()[-1])
    assert checkpoint["pause"] == pause and pause["order_index"] == checkpoint["completed_callbacks"] == COUNT
    assert checkpoint["next_callback_untouched"] is True and checkpoint["holdout_open"] is False
    receipts = MAIN / "order-receipts"
    assert {p.name for p in receipts.glob("*.json")} == {f"{i:06d}.json" for i in range(COUNT)}
    slots = read(order)["slots"]
    assert len(slots) == 1296 and [s["order_index"] for s in slots] == list(range(len(slots)))
    done = Counter()
    for slot in slots[:COUNT]:
        r = read(receipts / f"{slot['order_index']:06d}.json")
        assert all(r[k] == slot[k] for k in ("order_index","case_id","condition","repeat_ordinal"))
        done["classic" if slot["condition"] in CLASSIC else "llm"] += 1
    assert done == {"classic": 12, "llm": 58}
    remaining = (216 - done["classic"], 1080 - done["llm"])
    assert remaining == (204, 1022)

    ctl = impl["Controller"](RUN, impl["RUNTIME"])
    def pins():
        return {name: {"sha256": file_hash(ctl.home / name), "bytes": (ctl.home / name).stat().st_size}
                for name in ("control.json", "checkpoint.json")}
    with ctl.lock():
        base, before = ctl.load(), pins()
    status = read(status_path)
    known = impl["summary"](base)
    assert status["stale"] == [] and status["budget"]["known_usage"] == known["known_usage"]
    assert known["unknown_usage"] == known["unresolved_usage"] == [] and known["reserved"]["calls"] == 0
    assert base["active_round"]["id"] == "round-132" and base["pending_round"] is None
    objects = impl["closure"](RUN, base)
    current_docs = impl["documents"](RUN)
    original_doc_count = len(current_docs)
    object_bytes = sum((RUN / path).stat().st_size for path in objects)
    for slot in slots[COUNT:]:
        # In addition to the original shape's dispatch, auxiliary and round paths.
        for label in ("per-slot-completed-receipt", "per-slot-check-and-storage-evidence"):
            path = "qualification/synthetic-main-only-storage-allowance/" + label + f"-{slot['order_index']:04d}.json"
            assert path not in current_docs
            current_docs[path] = {"sha256": impl["HASH"], "lines": 1200}
    old_shape = read(PREVIOUS)["scenarios"]["cap2_stress_shape"]
    actual_size = impl["size"]
    current_templates = [row for row in base["rounds"]
                         if "documents_ref" in row["request"]["round"]]
    assert current_templates and all(row in current_templates for row in base["rounds"][-12:])
    historical_peak = max(current_templates, key=actual_size)
    regular_peak = deepcopy(max(current_templates[-20:], key=actual_size))
    if actual_size(regular_peak) < 12800:
        regular_peak["decision"]["synthetic_storage_padding"] = "x" * (12800-actual_size(regular_peak))
    assert 12800 <= actual_size(regular_peak) < 13000
    exceptional_rounds = 64
    exception_buffer_base = deepcopy(base)
    exception_buffer_base["rounds"].extend([historical_peak] * exceptional_rounds)
    shape = impl["shape"]
    shape.__globals__["TRACKS"] = {"main": remaining}
    source_flight = Q / "formal-capacity-final.v1/flight-storage.json"
    maxima = read(source_flight)["observed_maxima"]
    profile = read(PREVIOUS)["profile_caps"]
    assert profile == {"calls":32,"tokens":180000}
    conservative_raw = {"classic":29546687,"llm":44488936}
    attempts = 2 * remaining[1]
    main_raw = 2 * (remaining[0] * conservative_raw["classic"] + remaining[1] * conservative_raw["llm"])
    headroom = 10 * 1024**3
    temp = 2 * max(conservative_raw.values())
    scenarios = {}
    for label, rounds, template, owner_base, exceptions in (
        ("historical_maximum_repeated_1400",1400,historical_peak,base,0),
        ("recent_regular_plus_exception_reserve_1400",1400,regular_peak,exception_buffer_base,exceptional_rounds),
        ("recent_regular_plus_exception_reserve_2000",2000,regular_peak,exception_buffer_base,exceptional_rounds),
    ):
        candidate = shape(owner_base,current_docs,template,status,maxima,profile,2,rounds,object_bytes)
        assert candidate["attempts"] == {"classic":2*remaining[0],"llm":attempts}
        within_owner = max(candidate["owner_pair_bytes"].values()) < candidate["single_owner_file_limit"]
        within_document = candidate["document_object_bytes"] < candidate["ordinary_json_limit"]
        old_wire = old_shape["native_peak_encoded_by_command"]
        by_command = {key:{k:max(v,old_wire[key][k]) for k,v in fields.items()}
                      for key,fields in candidate["native_peak_encoded_by_command"].items()}
        counts = candidate["native_command_counts_in_shape"]
        assert counts["budget-reserve"] == counts["budget-settle"] == attempts
        counters = {
            "episode_native_stdout": attempts * (by_command["budget-reserve"]["stdout"] + by_command["budget-settle"]["stdout"]),
            "additional_archive_native_stdout": attempts * by_command["budget-reserve"]["stdout"],
            "native_cli_traces": sum(count*by_command[k]["trace"] for k,count in counts.items()),
            "unique_operation_results": sum(count*by_command[k].get("result_object",0) for k,count in counts.items()),
            "round_document_objects": rounds * candidate["document_object_bytes"],
            "request_and_decision_files": (max(candidate["storage"]["round_request_and_judgment_files"],rounds*12586)
                + exceptions*(actual_size(historical_peak["request"])+actual_size(historical_peak["decision"]))),
            "miscellaneous_and_analysis": old_shape["storage"]["miscellaneous_analysis_allowance"],
            "state_and_checkpoint": sum(max(size,old_shape["owner_pair_bytes"][k]) for k,size in candidate["owner_pair_bytes"].items()),
            "filesystem_headroom": headroom,
        }
        assert counters["miscellaneous_and_analysis"] == 1024**3
        candidate_other = sum(counters.values())
        scenarios[label] = {"round_shape_count":rounds,"projection":candidate,
                            "historical_large_rounds_in_extra_owner_reserve":exceptions,
                            "regular_round_bytes_in_scenario":actual_size(template),
                            "owner_state_within_limit":within_owner,"document_within_limit":within_document,
                            "conservative_wire_maxima":by_command,"main_other_counters":counters,
                            "main_other_bytes":candidate_other,
                            "main_raw_plus_temporary_and_other_bytes":main_raw+temp+candidate_other}
    assert pins() == before
    free = shutil.disk_usage(PROJECT).free
    for scenario in scenarios.values():
        scenario["fresh_margin_bytes"] = free-scenario["main_raw_plus_temporary_and_other_bytes"]
    return {"role":"read-only actual native baseline, in-memory sizing; no dispatch authority",
            "at":datetime.now(timezone.utc).isoformat(),"original_source_sha256":file_hash(SOURCE),
            "prior_all_phase_shape_sha256":file_hash(PREVIOUS),"closed_barrier_sha256":file_hash(CHECK/"after-0070.json"),
            "owner_baseline_unchanged":before,"owner_semantic_sha256":impl["digest"](base),
            "actual_known_usage":known["known_usage"],"original_document_count":original_doc_count,
            "additional_per_slot_document_paths":len(current_docs)-original_doc_count,
            "max_historical_current_schema_round_encoded_bytes":actual_size(historical_peak),
            "max_recent_regular_round_encoded_bytes":max(map(actual_size,current_templates[-20:])),
            "remaining_slots":{"classic":remaining[0],"llm":remaining[1]},
            "raw_main_attempt_allowances":conservative_raw,"full_cap2_raw_main_bytes":main_raw,
            "unchanged_pack_temporary_bytes":temp,"unchanged_filesystem_headroom_bytes":headroom,
            "same_volume_free_bytes":free,"scenarios":scenarios,
            "limits":["the projections bound named synthetic shapes, not arbitrary future payloads",
                      "for the mixed shape the total document count includes exceptional records, and their maximum owner/request bytes are added again as an extra reserve",
                      "no native mutations, real exports, phase changes, or holdout access",
                      "rounds are planning scenarios; real growth and phase boundaries must be rechecked"],
            "elapsed_seconds":time.monotonic()-t0}


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])),indent=2,sort_keys=True,allow_nan=False))
