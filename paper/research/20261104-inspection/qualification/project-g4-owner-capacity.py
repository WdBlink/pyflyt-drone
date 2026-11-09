"""Use the native encoders on named, in-memory formal workload shapes."""
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[4]
RUN = PROJECT / "paper/research/20261104-inspection"
RUNTIME = RUN / "runtime/autoresearch-document-snapshots-v2"
OUT = RUN / "qualification/formal-capacity-final.v1"
sys.path.insert(0, str(RUNTIME))

from lib.budget import summary
from lib.round_documents import FORMAT, MAX_BYTES, RESULT_FORMAT, closure, encoded
from lib.rounds import checkpoint, documents
from lib.runtime import Controller, OWNER_STATE_MAX_BYTES, digest, file_hash

TRACKS = {"main":(216,1080), "holdout":(36,180), "clean":(36,180), "ablation":(0,108), "latency":(0,180)}
STAMP = "2026-11-08T00:00:00.000000+00:00"
HASH = "a"*64


def size(value):
    return len(encoded(value))+1


def transport(name, value, stored):
    result = {"stdout":len(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False).encode())+1,
              "trace":len(json.dumps({"command":name,"exit_code":0,"response":value},
                                      ensure_ascii=False,allow_nan=False).encode())+1}
    if stored:
        result["result_object"] = size({"schema":RESULT_FORMAT,"result":value})
    return result


def ref(path):
    return {"path":path,"sha256":HASH}


def synthetic_resource(track, i, attempt, profile):
    phase = "holdout" if track=="holdout" else "main"
    case = phase+"-OOD-L3-00"
    condition = "B4-M1"
    arm = {"clean":"clean","ablation":"no-progressive-confirmation-guidance","latency":"delay-20"}.get(track)
    folder = "secondary-"+track if arm else phase
    packet = ref(f"dispatch/{folder}.v1/{case}--{condition}--"+(f"{arm}--" if arm else "")+"2.json")
    binding = {"packet":packet,"phase":phase,"execution":ref(f"execution/{folder}.v1/manifest.json"),
               "order_index":1295,"case_id":case,"condition":condition,"repeat_ordinal":2,"sampler_seed":2**32-1}
    if arm:
        binding.update(track=track,arm=arm,parent_order_index=1295)
    purpose = {"kind":"inspection_local_generation","phase":phase,"model_registry_sha256":HASH,
               "model_receipt_sha256":HASH,"condition":condition,"case_id":case,"repeat_ordinal":2,
               "episode_plan_sha256":HASH,"attempt_ordinal":attempt,"formal_dispatch":packet,
               "formal_dispatch_binding":binding}
    key = f"formal-{track}-{'f'*12}-{i:04d}-a{attempt}"
    receipt = ref(f"qualification/local-model-episodes/{key}/usage-receipt.json")
    reserved = {"reserved":profile,"purpose":json.dumps(purpose,sort_keys=True),"status":"reserved","created_at":STAMP}
    settled = {**reserved,"status":"known","reported_usage":profile,"receipt":receipt["path"],"unreserved":False,
               "receipt_sha256":HASH,"receipts":[receipt],"settled_at":STAMP}
    return key,reserved,settled


def shape(base, current_docs, peak_round, status, maxima, profile, cap, rounds, owner_objects_bytes):
    """All additions live only in memory; no seed, authorization or owner write."""
    projected = copy.deepcopy(base)
    doc = dict(current_docs)
    new_res = []
    obj = {"format":RESULT_FORMAT,"path":".paper/operation-results/"+HASH+".json","sha256":HASH}

    def operation(key):
        row = {"request_sha256":HASH,"result_ref":obj}
        assert key not in projected["operations"]
        projected["operations"][key] = row
        projected["history"].append({"operation":key,"at":STAMP,**row})

    def entry(path, lines):
        assert path not in doc
        doc[path] = {"sha256":HASH,"lines":lines}

    for track, (classic,llm) in TRACKS.items():
        for i in range(llm):
            for attempt in range(cap):
                key,reserved,settled = synthetic_resource(track,i,attempt,profile)
                assert key not in projected["usage_ledger"]
                projected["usage_ledger"][key] = settled
                new_res.append((key,reserved,settled))
                operation(key)
                operation("settle-"+key)
                for kind in ("reserve","settle"):
                    entry(f"resource-requests/{key}.{kind}.json",6)
                entry(settled["receipt"],1800)
        for i in range(classic+llm):
            entry(f"dispatch/shape-{track}.v1/holdout-OOD-L3-00--B4-M1--no-progressive-confirmation-guidance--{i:04d}.json",40)
    for i in range(1000):
        entry(f"qualification/formal-auxiliary-shape/production-manifest-and-independent-evidence-{i:04d}.json",1200)
    for i in range(rounds):
        operation(f"capacity-shape-formal-round-{i:04d}")
        projected["history"].append({"event":"session_started","id":f"capacity-shape-formal-session-{i:04d}","at":STAMP})
        projected["rounds"].append(peak_round)
        entry(f"qualification/formal-progress/round-{i:04d}-verified-progress-evidence.json",1200)
    for i in range(200):
        operation(f"capacity-shape-formal-other-operation-{i:04d}")
    doc_bytes = size({"schema":FORMAT,"documents":doc})
    projected["active_round"] = peak_round["request"]["round"]
    projected["pending_round"] = peak_round["request"]
    key,reserved,settled = max(new_res,key=lambda item:len(item[0]))
    final_budget = summary(projected)
    projected["usage_ledger"][key] = reserved
    reserve_budget = summary(projected)
    projected["usage_ledger"][key] = settled
    assert final_budget["unknown_usage"]==[] and final_budget["known_usage"]["cost"]=={}
    pair = {"control":size(projected),"checkpoint":size({"state":projected,"sha256":HASH})}
    peak_wire = {
        "budget-reserve":transport("budget-reserve",{"status":"reserved","reservation":key,"budget":reserve_budget},True),
        "budget-settle":transport("budget-settle",{"status":"known","reservation":key,"budget":final_budget},True),
        "status":transport("status",{**status,"budget":final_budget,"checkpoint":checkpoint(projected)},False),
        "prepare-round":transport("prepare-round",peak_round["request"],False),
        "begin-round":transport("begin-round",{"status":"round_started","round":projected["active_round"],
                                                "budget":final_budget,"checkpoint":checkpoint(projected)},False),
        "apply-round":transport("apply-round",{"status":"round_recorded","round":len(projected["rounds"]),
            "stale_count":0,"budget":final_budget,"checkpoint":checkpoint(projected),"attention_required":False,
            "total_findings":len(projected["rounds"])},True)}
    n_classic = cap*sum(row[0] for row in TRACKS.values())
    n_llm = len(new_res)
    classic = maxima["classic"]["episode_nonwire_bytes"]
    llm = sum(maxima["llm"][k] for k in ("episode_nonwire_bytes","model_originals_bytes","archive_bytes",
                                        "usage_receipt_bytes","resource_packet_bytes"))
    retained_attempts = n_classic*classic+n_llm*llm
    reserve, settle = (peak_wire[key] for key in ("budget-reserve","budget-settle"))
    native_stdout = n_llm*(reserve["stdout"]+settle["stdout"])
    archive_wire_allowance = n_llm*reserve["stdout"]
    commands = Counter({"budget-reserve":n_llm,"budget-settle":n_llm,
                        "status":rounds+30,"prepare-round":rounds,"begin-round":rounds,"apply-round":rounds})
    traces = sum(count*peak_wire[key]["trace"] for key,count in commands.items())
    results = sum(count*peak_wire[key].get("result_object",0) for key,count in commands.items())
    docs = rounds*doc_bytes
    miscellaneous_and_analysis = 1024**3
    round_files = rounds*(size(peak_round["request"])+size(peak_round["decision"]))
    closure_growth = results+docs
    persistent = (retained_attempts+native_stdout+archive_wire_allowance+traces+closure_growth+round_files
                  +miscellaneous_and_analysis+sum(pair.values()))
    # Finalizer copies accepted assets, usage receipts and owner objects,
    # and retains the copy plus ZIP. Raw flights remain where recorded.
    export_assets = (owner_objects_bytes+closure_growth+pair["control"]
                     +n_llm*maxima["llm"]["usage_receipt_bytes"]+miscellaneous_and_analysis)
    export_peak = 2*export_assets
    headroom = 10*1024**3
    return {"attempts":{"classic":n_classic,"llm":n_llm},"round_template_count":rounds,
        "owner_pair_bytes":pair,"single_owner_file_limit":OWNER_STATE_MAX_BYTES,
        "document_object_bytes":doc_bytes,"document_paths":len(doc),"ordinary_json_limit":MAX_BYTES,
        "max_document_ref_path_bytes":max(len(path.encode()) for path in doc),
        "native_peak_encoded_by_command":peak_wire,"native_command_counts_in_shape":dict(commands),
        "storage":{"observed_component_maxima_times_attempts":retained_attempts,
                   "separate_episode_native_stdout":native_stdout,"additional_zip_reserve_wire_allowance":archive_wire_allowance,
                   "native_cli_traces":traces,"unique_result_objects_no_dedup_credit":results,
                   "round_document_objects_no_dedup_credit":docs,"round_request_and_judgment_files":round_files,
                   "miscellaneous_analysis_allowance":miscellaneous_and_analysis,"persistent_delta":persistent,
                   "export_directory_plus_zip_allowance":export_peak,"filesystem_headroom":headroom,
                   "projected_needed":persistent+export_peak+headroom}}


def main(status_path):
    start = time.monotonic()
    ctl = Controller(RUN,RUNTIME)
    with ctl.lock():
        base = ctl.load()
        pins = {name:{"sha256":file_hash(ctl.home/name),"bytes":(ctl.home/name).stat().st_size}
                for name in ("control.json","checkpoint.json")}
    load_seconds = time.monotonic()-start
    status = json.loads(status_path.read_bytes())
    known = summary(base)
    assert status["stale"]==[] and known["known_usage"]==status["budget"]["known_usage"]
    assert known["unknown_usage"]==known["unresolved_usage"]==[] and not known["reserved"]["calls"]
    assert base["active_round"] and base["pending_round"] is None and base["task_stop"] is None
    objects = closure(RUN,base)
    current_docs = documents(RUN)
    object_bytes = sum((RUN/path).stat().st_size for path in objects)
    sample = OUT/"flight-storage.json"
    maxima = json.loads(sample.read_bytes())["observed_maxima"]
    registry = json.loads((RUN/"planning/local-model-registry.v1.json").read_bytes())
    profile = {name:max(value["generation_profile"][field] for value in registry["models"].values())
               for name,field in (("calls","max_calls"),("tokens","max_total_tokens"))}
    peak = max(base["rounds"][-12:],key=size)
    assert "documents_ref" in peak["request"]["round"]
    scenarios = {label:shape(base,current_docs,peak,status,maxima,profile,cap,rounds,object_bytes)
                 for label,cap,rounds in (("nominal",1,750),("cap2_stress_shape",2,2000))}
    free = shutil.disk_usage(PROJECT).free
    for value in scenarios.values():
        value["sample_shape_disk_margin"] = free-value["storage"]["projected_needed"]
        value["owner_pair_margin"] = OWNER_STATE_MAX_BYTES-max(value["owner_pair_bytes"].values())
        value["document_object_margin"] = MAX_BYTES-value["document_object_bytes"]
    source = Path(__file__).resolve()
    output = {"role":"actual owner baseline and scoped in-memory workload projections; not formal execution",
        "at":datetime.now(timezone.utc).isoformat(),"baseline":pins,"base_state_semantic_sha256":digest(base),
        "baseline_native_load_seconds":load_seconds,"baseline_document_paths":len(current_docs),
        "baseline_referenced_owner_objects":len(objects),"baseline_referenced_owner_object_bytes":object_bytes,
        "prior_normal_round_shape_bytes":size(peak),"tracks":TRACKS,"profile_caps":profile,
        "filesystem_available_bytes":free,"scenarios":scenarios,
        "source_sha256":file_hash(source),"flight_inventory_sha256":file_hash(sample),
        "status_sha256":file_hash(status_path),
        "limits":["Observed per-attempt maxima and projected shapes are not future mathematical bounds.",
                  "Rounds, other operations and allowances describe sizing scenarios, not experiment or resource caps.",
                  "Peak wire sizes are separate for each encoder; multiplication is a scenario allowance.",
                  "No future production native CLI, serialization/fsync, generation, physics or export was executed.",
                  "Capacity and other host activity must be rechecked at real phase boundaries."],
        "elapsed_seconds":time.monotonic()-start,"peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    with (OUT/"native-storage-projection.json").open("x") as stream:
        json.dump(output,stream,indent=2,allow_nan=False)
    print(json.dumps(output,allow_nan=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
