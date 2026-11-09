"""Combine actual storage probes with the post-phase compressed storage policy."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT/"paper/research/20261104-inspection"
Q = RUN/"qualification/formal-capacity-final.v1"


def read(name):
    return json.loads((Q/name).read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    samples=read("flight-storage.json")
    model=read("native-storage-projection.json")
    copies=read("apfs-temporary-compression.v1.json")
    witness=read("apfs-pack-software-witness.v2.json")
    tool=PROJECT/"logs/inspection-g4-tools.v1/applesauce"
    pack=RUN/"qualification/pack-closed-formal-phase.py"
    assert digest(tool)==copies["tool_sha256"]==witness["tool_sha256"]
    assert digest(pack)==witness["source_sha256"] and len(samples["samples"])==copies["samples"]==112
    # The deployed wrapper skips files under 16 KiB. Add all such observed
    # source blocks to the compressed-copy envelope, with no savings credit.
    allowances={"classic":0,"llm":0}
    for row in samples["samples"]:
        bases=[PROJECT/row["attempt"]]
        if "reservation" in row:
            bases.append(PROJECT/"logs/inspection-model-resource-operations.v1"/row["reservation"])
        allowance=sum(p.stat().st_blocks*512 for base in bases for p in base.rglob("*")
                      if p.is_file() and not p.is_symlink() and p.stat().st_size<16384)
        group="llm" if len(bases)==2 else "classic"
        allowances[group]=max(allowances[group],allowance)
    old=samples["observed_maxima"]
    packed=copies["per_class_allocated_maxima"]
    unchanged_model=sum(old["llm"][k] for k in ("archive_bytes","usage_receipt_bytes","resource_packet_bytes"))
    before={"classic":old["classic"]["episode_nonwire_bytes"],
            "llm":old["llm"]["episode_nonwire_bytes"]+old["llm"]["model_originals_bytes"]+unchanged_model}
    after={"classic":packed["classic"]["episode_nonwire"]+allowances["classic"],
           "llm":packed["llm"]["episode_nonwire"]+packed["llm"]["model_originals"]+unchanged_model+allowances["llm"]}
    free=shutil.disk_usage(PROJECT).free
    results={}
    for label,scenario in model["scenarios"].items():
        n=scenario["attempts"]
        cap=n["llm"]//1728
        storage=scenario["storage"]
        other=storage["persistent_delta"]-storage["observed_component_maxima_times_attempts"]+storage["filesystem_headroom"]
        retained=0
        stages={}
        for phase,(classic,llm) in model["tracks"].items():
            stages[phase]=retained+cap*(classic*before["classic"]+llm*before["llm"])+other
            retained+=cap*(classic*after["classic"]+llm*after["llm"])
        final=retained+other+storage["export_directory_plus_zip_allowance"]
        peak=max(final,*stages.values())
        results[label]={"phase_peak_before_pack":stages,"compressed_export_peak":final,
                        "required_sample_shape_bytes":peak,"available_margin_bytes":free-peak}
    value={"role":"same-device observed compression plus explicit storage scenarios, no future capacity guarantee",
        "at":datetime.now(timezone.utc).isoformat(),"free_bytes":free,"scenarios":results,
        "per_attempt_base_before":before,"per_attempt_base_after":after,"skipped_small_file_allowances":allowances,
        "tool_release":"https://github.com/Dr-Emann/applesauce/releases/tag/applesauce-cli-v0.5.30",
        "tool_arm64_archive_sha256":"a9056a441e266360718dff3803f26bf9ae3e693bba11eb38030b9b4772a0f80d",
        "tool":{"path":str(tool.relative_to(PROJECT)),"sha256":digest(tool)},
        "rules":["One phase at a time. Apply APFS packing only after full SDK completion and matrix lock acquisition.",
                 "Verify current available disk before each phase and after packing; an adverse actual change blocks dispatch.",
                 "No change to logical traces, score, retry cap, scientific N, owner storage or actual physical/model inputs.",
                 "Measured maxima and export/headroom allowances apply only to this stated shape; no guarantee under all future payloads."],
        "files":{str(path.relative_to(PROJECT)):digest(path) for path in (
            Q/"flight-storage.json",Q/"native-storage-projection.json",Q/"apfs-temporary-compression.v1.json",
            Q/"apfs-pack-software-witness.v2.json",pack,Path(__file__).resolve())}}
    with (Q/"capacity-with-lossless-pack.v1.json").open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in value.items() if k!="files"}))


if __name__=="__main__":
    main()
