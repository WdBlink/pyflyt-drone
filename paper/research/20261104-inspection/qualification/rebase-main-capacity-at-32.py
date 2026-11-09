"""Recompute the remaining qualified capacity shape at the actual 32-slot barrier."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
Q = RUN / "qualification"
C = Q / "formal-capacity-final.v1"
OUT = Q / "formal-main-checkpoints.v1"
LOGS = PROJECT / "logs"
sys.path.insert(0, str(PROJECT))

from eval.inspection_episode_receipts import durable_once
from eval.inspection_formal_admission import file_sha


def read(path):
    return json.loads(path.read_bytes())


def ref(path):
    return {"path": str(path.relative_to(RUN)), "sha256": file_sha(path)}


def main():
    main_logs = LOGS / "inspection-main-primary.v1"
    assert (main_logs / "checkpoint.request").is_file()
    assert json.loads((main_logs / "checkpoint-pauses.jsonl").read_text().splitlines()[-1])["order_index"] == 32
    assert {x.name for x in (main_logs/"order-receipts").glob("*.json")} == {f"{i:06d}.json" for i in range(32)}
    assert not (RUN / "datasets/holdout.v1").exists()
    assert not (RUN / "dispatch/main.v1/main-OOD-L3-06--B3-M2--2.json").exists()
    for d in (Q, LOGS, PROJECT, RUN):
        assert d.stat().st_dev == PROJECT.stat().st_dev

    frozen = C / "capacity-with-lossless-pack.v1.json"
    native = C / "native-storage-projection.json"
    flight = C / "flight-storage.json"
    measure = Q / "measure-g4-storage.py"
    before_subject = C / "apfs-temporary-compression.v1.json"
    pack_path = OUT / "closed-development-world-pack.v1.json"
    scratch_path = OUT / "after-0032-scratch-cleanup.v1.json"
    closed = OUT / "after-0032.json"
    order = RUN / "execution/main.v1/episode-order.json"
    assert file_sha(frozen) == "6a56020990722aaa6edd3c56fd252533804183a956aa49ef65d10c5bc3488db3"
    assert file_sha(closed) == "85ff6da515a0b44a493f3f665c60c51193d51d85d06a761cd5af66e20a49627d"
    assert file_sha(order) == "8bdcfcfa60d564092feb002ed5262a7a0bb60719a49afc4bea158a4a6e69c6e7"
    assert file_sha(pack_path) == "9c5e6a020ef2de922cc9d141fe3206b9cbfabd7cffd727c6728e59c1af43802f"
    assert file_sha(scratch_path) == "2b1a3c9c9f7700e200f03aa3d55ca34dd856c99b831cdda02be9b495b99f5dca"
    capacity, model, flight_data = read(frozen), read(native), read(flight)
    for path in (native,flight):
        assert file_sha(path) == capacity["files"][str(path.relative_to(PROJECT))]
    assert file_sha(measure) == flight_data["source_pins"][str(measure.relative_to(PROJECT))]
    assert read(pack_path)["scope_count"] == read(pack_path)["logical_shas_verified"] == 105
    assert read(closed)["completed_callbacks"] == 32
    samples, done = [], Counter()
    slot_receipts = []
    sample = runpy.run_path(str(measure))["sample"]
    for i, s in enumerate(read(order)["slots"][:32]):
        assert s["order_index"] == i
        cond = s["condition"]
        group = "classic" if cond in ("B0","B1","B2") else "llm"
        done[group] += 1
        d = LOGS / "inspection-formal-episodes.v1/main" / f"{s['case_id']}-{cond}-{s['repeat_ordinal']}"
        receipt = main_logs / "order-receipts" / f"{i:06d}.json"
        row = read(receipt)
        assert all(row[k] == s[k] for k in ("case_id","condition","repeat_ordinal","order_index"))
        slot_receipts.append({"project_path":str(receipt.relative_to(PROJECT)),"sha256":file_sha(receipt)})
        attempts = sorted(d.glob("attempt-*"))
        assert len(attempts) in (1,2) and read(attempts[-1]/"terminal.json")["status"] == "completed"
        for a in attempts:
            res = read(a/"intent.json").get("reservation")
            assert bool(res) == (group == "llm")
            samples.append(sample(a,"closed_formal",s["level"],cond,res))
    assert done == {"classic":4,"llm":28} and len(samples) == 33
    old = flight_data["observed_maxima"]
    observed = {}
    for group in old:
        subset = [x for x in samples if ("reservation" in x) == (group == "llm")]
        observed[group] = {k:max(x.get(k,0) for x in subset) for k in old[group]}
        observed[group]["native_stdout_max"] = max((v for x in subset for v in x["native_stdout_bytes"].values()), default=0)
    keys = {"classic":("episode_nonwire_bytes",),"llm":("episode_nonwire_bytes","model_originals_bytes","archive_bytes",
        "usage_receipt_bytes","resource_packet_bytes")}
    expansions = {g:{k:max(0,observed[g][k]-old[g][k]) for k in fields} for g,fields in keys.items()}
    # No new credit for compression of enlarged components. Charge their full
    # observed growth to both the raw and the packed remaining shape.
    raw = {g:capacity["per_attempt_base_before"][g]+sum(expansions[g].values()) for g in keys}
    packed = {g:capacity["per_attempt_base_after"][g]+sum(expansions[g].values()) for g in keys}
    s = model["scenarios"]["cap2_stress_shape"]["storage"]
    other = s["persistent_delta"]-s["observed_component_maxima_times_attempts"]+s["filesystem_headroom"]
    assert s["filesystem_headroom"] == 10*1024**3
    tracks = {phase:list(count) for phase,count in model["tracks"].items()}
    assert list(tracks) == ["main","holdout","clean","ablation","latency"]
    tracks["main"] = [tracks["main"][0]-done["classic"],tracks["main"][1]-done["llm"]]
    stages, retained = {}, 0
    for phase,(classic,llm) in tracks.items():
        stages[phase] = retained + 2*(classic*raw["classic"]+llm*raw["llm"]) + other
        retained += 2*(classic*packed["classic"]+llm*packed["llm"])
    final = retained+other+s["export_directory_plus_zip_allowance"]
    required = max(final,*stages.values())
    assert required == 156017544308 and final == required
    free = shutil.disk_usage(PROJECT).free
    value = {"role":"actual capacity rebase with frozen cap2 shape and no completed-slot future duplication",
        "at":datetime.now(timezone.utc).isoformat(), "frozen":ref(frozen),"native_shape":ref(native),
        "completed_checkpoint":ref(closed),"same_volume_free_bytes":free,
        "admission_ready_for_safe_review":free>=required,
        "remaining_required_bytes":required,"margin_bytes":free-required,
        "raw_per_attempt":raw,"packed_per_attempt":packed,"old_sample_maxima":old,
        "actual_33_attempt_maxima":observed,"added_component_bytes":expansions,"known_completed_slots":dict(done),
        "remaining_tracks":tracks,"phase_peaks_additional_bytes":stages,"export_peak_additional_bytes":final,
        "export_allowance_bytes":s["export_directory_plus_zip_allowance"],"other_reserved_bytes":other,
        "filesystem_headroom_bytes":s["filesystem_headroom"],"attempt_cap_unchanged":2,
        "only_closed_slots_subtracted":True,"cleanup":ref(scratch_path),"development_world_pack":ref(pack_path),
        "formal_recording_bytes_changed":False,"slot_receipts":slot_receipts,"formal_attempt_samples":samples}
    path = OUT / "after-0032-capacity-rebase.v1.json"
    durable_once(path,value)
    print(json.dumps({"artifact":ref(path),**{k:value[k] for k in ("same_volume_free_bytes","remaining_required_bytes",
        "margin_bytes","phase_peaks_additional_bytes","added_component_bytes","admission_ready_for_safe_review")}}))


if __name__ == "__main__":
    main()
