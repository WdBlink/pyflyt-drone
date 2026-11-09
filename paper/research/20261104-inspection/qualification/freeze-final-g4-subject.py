"""Freeze installed identities and actual qualification evidence for independent G4 review."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[4]
RUN=PROJECT/"paper/research/20261104-inspection"
Q=RUN/"qualification"
OUT=Q/"formal-freeze-final.v1"
RUNTIME=RUN/"runtime/autoresearch-document-snapshots-v2"
SDK=RUN/"runtime/autoresearch-state-reader-v1"
sys.path.insert(0,str(PROJECT))

from eval.inspection_formal_admission import Artifact, context_from_environment, file_sha, read_json
from eval.inspection_episode_receipts import durable_once


def ref(name):
    path=RUN/name
    value={"path":str(path.relative_to(RUN)),"sha256":file_sha(path)}
    Artifact.parse(value).checked_path(RUN)
    return value


def versions(names):
    return {name:importlib.metadata.version(name) for name in names}


def main():
    assert not OUT.exists() and not (RUN/"datasets").exists()
    context=context_from_environment(RUN,PROJECT,RUNTIME,SDK)
    owner=context.owner_reader(RUN)
    assert owner["active_round"]["id"]=="round-88" and owner["task_stop"] is None
    assert [item["stage"] for item in owner["receipts"]]==["intake","venue","plan","literature"]
    assert owner["brief"]["budget"]=={} and not owner["pending_round"]
    [plan]=[row["packet"] for row in owner["receipts"] if row["stage"]=="plan"]
    design=ref(plan["experiment_design"])
    registry=ref(plan["local_model_registry"])
    assert design["sha256"]==plan["experiment_design_sha256"]
    assert registry["sha256"]==plan["local_model_registry_sha256"]
    design_body=read_json(RUN/design["path"])
    population=design_body["split_contract"]["population"]
    Artifact.parse(population).checked_path(RUN)
    install=read_json(Q/"formal-installation.v1/verified-installed.v1.json")
    assert len(install["sources"])==44 and install["pytest_passed"]==970 and install["pytest_exit"]==0
    source_hashes={name:file_sha(PROJECT/name) for name in sorted(context.source_names)}
    assert all(source_hashes.get(name)==expected for name,expected in install["sources"].items())
    dependencies={name:file_sha(name) for name in sorted(context.dependency_names)}
    assert len(dependencies)==61
    registry_body=read_json(RUN/registry["path"])
    model_receipts={}
    for key, entry in registry_body["models"].items():
        receipt=ref(entry["download_receipt"])
        assert receipt["sha256"]==entry["download_receipt_sha256"]
        model_receipts[key]={"receipt":receipt,"generation_profile":entry["generation_profile"]}
    assert set(model_receipts)=={"M1","M2","M3"}
    hardware={name:subprocess.check_output(["/usr/sbin/sysctl","-n",name],text=True).strip()
              for name in ("hw.model","hw.memsize","hw.physicalcpu","hw.logicalcpu","machdep.cpu.brand_string")}
    model_env=json.loads(subprocess.check_output([str(PROJECT/".venv-inspection-llm/bin/python"),"-c",
        'import importlib.metadata as m,json,sys;print(json.dumps({"python":sys.version.split()[0],'
        '"packages":{k:m.version(k) for k in ("mlx","mlx-lm","transformers","tokenizers","numpy","huggingface-hub")}}))'],
        text=True))
    capacity=read_json(Q/"formal-capacity-final.v1/capacity-with-lossless-pack.v1.json")
    disk=shutil.disk_usage(PROJECT)
    assert disk.free>=capacity["scenarios"]["cap2_stress_shape"]["required_sample_shape_bytes"]
    OUT.mkdir()
    durable_once(OUT/"source-closure.json",{"schema":"inspection-final-source-closure-v1",
        "sources":source_hashes,"dependencies":dependencies,"absent_sources":sorted(context.absent_source_names)})
    durable_once(OUT/"hardware-profile.json",{"role":"actual declared serial local simulation host",
        "at":datetime.now(timezone.utc).isoformat(),"hardware":hardware,"machine_arch":platform.machine(),
        "os":subprocess.check_output(["/usr/bin/sw_vers"],text=True).strip().splitlines(),
        "physics_env":{"python":sys.version.split()[0],"packages":versions(("PyFlyt","pybullet","numpy","pytest"))},
        "model_env":model_env,"thread_limits":{k:"1" for k in
            ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS")},
        "clock":{"duration_seconds":600,"physics_hz":240,"control_hz":120,"max_lag_seconds":.1},
        "one_active_physical_episode":True,"free_bytes":disk.free,"filesystem_device":PROJECT.stat().st_dev,
        "readiness":"development engineering envelope only; no hardware flight or hard real-time guarantee"})
    evidence = [ref(name) for name in (
        "planning/gate-contract.v19.json",
        "qualification/g1-full.v4/manifest.json",
        "qualification/executor-full.v4/manifest.json",
        "qualification/protocol/retirement.v3.json",
        "qualification/reference/executor-realtime.v2.qualification.v3.json",
        "qualification/classic-six-root-cohort.v3/selection.json",
        "qualification/pilot-v4-complete.v1/binding.json",
        "qualification/formal-installation.v1/verified-installed.v1.json",
        "qualification/final-development-clean.v1/binding.json",
        "qualification/final-development-delay.v1/binding.json",
        "qualification/development-identity-catalog.v1/results.final.v1/binding.json",
        "qualification/formal-capacity-final.v1/capacity-with-lossless-pack.v1.json",
        "qualification/formal-capacity-final.v1/apfs-pack-software-witness.v2.json",
        "qualification/pack-closed-formal-phase.py",
        "review/evidence/formal-wholegraph-preflight-v1/binding.json",
        "review/formal-delta-runtime-qualification-plan.v1.md",
        "planning/statistical-protocol.ral.v1.md",
    )]
    catalog=ref("qualification/development-identity-catalog.v1/results.final.v1/catalog.json")
    evidence.extend([catalog,*[row["receipt"] for row in model_receipts.values()]])
    subject={"schema":"inspection-g4-subject-v1",
        "owner":{k:owner[k] for k in ("created_at","brief_sha256","frozen_plan")},
        "design":design,"population":population,"registry":registry,
        "source_manifest":ref("qualification/formal-freeze-final.v1/source-closure.json"),
        "hardware_profile":ref("qualification/formal-freeze-final.v1/hardware-profile.json"),
        "development_identity_catalog":catalog,"qualification_evidence":evidence,
        "model_receipts":model_receipts,
        "prompt_protocol_sources":{name:source_hashes[name] for name in (
            "eval/inspection_llm_actor.py","eval/inspection_llm_protocol.py","eval/run_inspection_llm_development.py",
            "eval/inspection_model_service.py","eval/inspection_classic.py")},
        "scope":{"status":"subject_for_independent_review_no_authorization",
            "venue":"RA-L","classical_comparator":"B1","episode_denominators":{k:sum(v) for k,v in {
                "main":(216,1080),"holdout":(36,180),"clean":(36,180),"ablation":(0,108),"latency":(0,180)}.items()},
            "max_attempts_per_slot":2,"budget":{},"local_pinned_models_only":True,
            "formal_datasets_exist":False,"paid_inference_or_external_compute_authorized":False,
            "development_is_not_formal_evidence":True,"first_real_formal_admission_pending":True,
            "capacity_requires_postphase_packing_and_fresh_available_space":True,
            "hardware_and_domain_human_review":"pending"}}
    durable_once(OUT/"subject.json",subject)
    result={"subject":ref("qualification/formal-freeze-final.v1/subject.json"),
        "sources":len(source_hashes),"dependencies":len(dependencies),"evidence":len(evidence),
        "disk_free":disk.free,"no_formal_data":not (RUN/"datasets").exists()}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
