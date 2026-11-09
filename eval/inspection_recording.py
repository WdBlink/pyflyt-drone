"""Eight-condition adapter using the supplied autoresearch recording SDK.

Classic and LLM episodes have separate rectangular inner records. The outer
record is one root/condition mean, not an extra flight or a provider random seed.
Primary root-paired inference belongs to a separate research statistics adapter.
"""

from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib
import math
from pathlib import Path
from statistics import mean

import numpy as np


CONDITION_REPEATS = {"B0":1,"B1":1,"B2":1,"B3-M1":3,"B3-M2":3,"B3-M3":3,"B4-M1":3,"B4-M2":3}


def episode_schedule(case_metadata, seed):
    """Freeze root blocks across strata and randomize their18 actual episodes."""
    if type(seed) is not int or not 0<=seed<2**32:
        raise ValueError("explicit schedule seed required")
    groups,seen = {},set()
    for row in case_metadata:
        case,domain,level = row["case_id"],row["domain"],row["level"]
        if (not isinstance(case,str) or not case or case in seen or domain not in ("ID","OOD")
                or type(level) is not int or level not in ((1,2,3) if domain=="ID" else (2,3))):
            raise ValueError("unique explicit root/domain/level metadata required")
        seen.add(case)
        groups.setdefault((domain,level),[]).append(case)
    if not seen:raise ValueError("nonempty root metadata required")
    rng = np.random.Generator(np.random.PCG64(seed))
    roots = []
    for key,cases in sorted(groups.items()):
        shuffled = rng.permutation(sorted(cases)).tolist()
        for i,case in enumerate(shuffled):
            # Every stratum spans the run, including unequal stratum sizes.
            roots.append(((i+float(rng.random()))/len(shuffled),case,key))
    slots = []
    for block,(_,case,(domain,level)) in enumerate(sorted(roots)):
        episodes = [(condition,repeat) for condition,count in CONDITION_REPEATS.items() for repeat in range(count)]
        for j in rng.permutation(len(episodes)):
            condition,repeat = episodes[int(j)]
            slots.append(dict(order_index=len(slots),root_block_index=block,case_id=case,
                domain=domain,level=level,condition=condition,repeat_ordinal=repeat))
    return dict(role="frozen_episode_order_not_a_measurement",seed=seed,rng="NumPy PCG64",
                numpy_version=np.__version__,cases=sorted(case_metadata,key=lambda r:r['case_id']),slots=slots)


def designs(cases, context, classic_best):
    if classic_best not in ("B1","B2"):
        raise ValueError("classic comparator must be frozen from development selection")
    if not cases or len(set(cases))!=len(cases):
        raise ValueError("nonempty unique independent root identities required")
    common = {**context,"cases":list(cases),"statistical_unit":"case","failure_policy":"penalty",
              "failure_value":0.,"metric":{"name":"Y","unit":"quality","direction":"higher"}}
    groups = {"classic":([c for c,n in CONDITION_REPEATS.items() if n==1],[0]),
              "llm":([c for c,n in CONDITION_REPEATS.items() if n==3],[0,1,2]),
              "root-means":(list(CONDITION_REPEATS),[0])}
    result = {}
    for group,(conditions,seeds) in groups.items():
        contrasts = ([{"id":"architecture-M1","minuend":"B4-M1","subtrahend":"B3-M1"},
                      {"id":"architecture-M2","minuend":"B4-M2","subtrahend":"B3-M2"},
                      {"id":"classic-M1","minuend":"B4-M1","subtrahend":classic_best}]
                     if group=="root-means" else
                     [{"id":"descriptive-inner","minuend":conditions[1],"subtrahend":conditions[0]}])
        result[group] = {**common,"experiment_id":context["experiment_id"]+"-"+group,
            "conditions":conditions,"seeds":seeds,"statistics":{"estimator":"mean","interval":"none",
            "paired":True,"contrasts":contrasts},"adapter_semantics":{
                "group":group,"seed_field":"one aggregate pass, not a flight" if group=="root-means"
                    else "episode repetition ordinal; actual provider seed, if any, belongs in evidence",
                "primary_inference":"not supplied by these SDK descriptive records"}}
    return result


def validate_episode_order(cases, execution_order):
    """Shared complete-matrix guard, without recording or random effects."""
    expected = {(case,repeat,c) for case in cases for c,n in CONDITION_REPEATS.items() for repeat in range(n)}
    slots = execution_order["slots"]
    actual = [(s["case_id"],s["repeat_ordinal"],s["condition"]) for s in slots]
    if (len(actual)!=len(expected) or set(actual)!=expected
            or any(type(s["repeat_ordinal"]) is not int or s["order_index"]!=i for i,s in enumerate(slots))
            or {r["case_id"] for r in execution_order["cases"]}!=set(cases)):
        raise ValueError("execution order must cover the full planned episode matrix exactly once")
    for block in range(len(cases)):
        part = slots[18*block:18*(block+1)]
        if len({s["case_id"] for s in part})!=1 or any(s["root_block_index"]!=block for s in part):
            raise ValueError("one complete18-episode block per independent root required")
    return slots


def record_matrix(root, cases, measure, *, context, classic_best, execution_order, recorder_type, write_json,
                  frozen_outer_design=None):
    """measure(root_id, repeat_ordinal, condition) returns actual Y and evidence.

    The caller must retain real episode traces, policy failures as scored
    outcomes, and operational failures as exceptions. All three SDK envelopes
    are opened before any flight so a failure retains the full planned matrix.
    """
    root = Path(root)
    specs = designs(cases,context,classic_best)
    if frozen_outer_design is not None:
        import json
        relative = Path(frozen_outer_design)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("frozen outer design must be inside the SDK recording root")
        canonical = json.loads((root/relative).read_text())
        if (canonical["cases"] != list(cases) or canonical["conditions"] != list(CONDITION_REPEATS)
                or canonical["seeds"] != [0]
                or canonical["recording_semantics"]["condition_repeats"] != CONDITION_REPEATS):
            raise ValueError("canonical outer design differs from the complete eight-condition grid")
        specs["root-means"] = canonical
    expected = {(case,repeat,c) for case in cases for c,n in CONDITION_REPEATS.items() for repeat in range(n)}
    slots = validate_episode_order(cases, execution_order)
    write_json(root,"episode-order.json",execution_order)
    schedule_hash = hashlib.sha256((root/"episode-order.json").read_bytes()).hexdigest()
    for group,spec in specs.items():
        if group == "root-means" and frozen_outer_design is not None:
            continue  # exact pre-existing bytes, no schedule or context rewrite
        spec["execution_schedule"] = {"path":"episode-order.json","sha256":schedule_hash}
        write_json(root,f"{group}/design.json",spec)
    measurements = {}
    records = {}
    with ExitStack() as stack:
        for group in specs:
            design_path = frozen_outer_design if group == "root-means" and frozen_outer_design is not None else f"{group}/design.json"
            records[group] = stack.enter_context(recorder_type(root,design=design_path,
                results=f"{group}/results.json",raw_directory=f"{group}/raw"))
        for slot in slots:
            case,repeat,condition = slot["case_id"],slot["repeat_ordinal"],slot["condition"]
            group = "classic" if CONDITION_REPEATS[condition]==1 else "llm"
            rec = records[group]
            with rec.attempt(case,repeat,condition) as complete:
                write_json(root,f"starts/{case}-{condition}-{repeat}.json",{
                    "role":"episode_attempt_start_not_completion","case_id":case,"condition":condition,
                    "repeat_ordinal":repeat,"status":"running","order_index":slot["order_index"],
                    "schedule_sha256":schedule_hash,"started_at":datetime.now(timezone.utc).isoformat()})
                value,evidence = measure(case,repeat,condition)
                if (type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1
                        or not isinstance(evidence,dict)):
                    raise ValueError("episode must supply finite scored Y in[0,1] and evidence")
                complete(value,{**evidence,"adapter_repeat_ordinal":repeat,"adapter_condition":condition,
                    "order_index":slot["order_index"],"schedule_sha256":schedule_hash})
                measurements[case,repeat,condition] = value
                write_json(root,f"order-receipts/{slot['order_index']:06d}.json",{
                    "role":"measurement_callback_returned_not_independent_flight_receipt",**slot,
                    "schedule_sha256":schedule_hash,"raw":rec.rows[case,repeat,condition]["raw"]})
        if hashlib.sha256((root/"episode-order.json").read_bytes()).hexdigest()!=schedule_hash:
            raise ValueError("frozen execution schedule changed")
        for group in ("classic","llm"):records[group].finish(outcome="inconclusive")
        outer = records["root-means"]
        for case,aggregation_pass,condition in outer.grid:
            group = "classic" if CONDITION_REPEATS[condition]==1 else "llm"
            inner = records[group]
            count = CONDITION_REPEATS[condition]
            bindings = []
            values = []
            for repeat in range(count):
                identity = (case,repeat,condition)
                if identity not in measurements or inner.rows[identity]["status"]!="completed":
                    raise ValueError("missing/operationally failed inner episode blocks aggregation")
                raw = inner.rows[identity]["raw"]
                bindings.append({"repeat_ordinal":repeat,"path":raw,
                    "sha256":hashlib.sha256((root/raw).read_bytes()).hexdigest()})
                values.append(measurements[identity])
            with outer.attempt(case,aggregation_pass,condition) as complete:
                complete(mean(values),{"aggregation":"arithmetic mean within independent root",
                    "actual_episode_count":count,"inner_attempts":bindings,
                    "inner_results":f"{group}/results.json",
                    "inner_results_sha256":hashlib.sha256((root/f"{group}/results.json").read_bytes()).hexdigest(),
                    "schedule_sha256":schedule_hash,
                    "not_an_additional_episode":True})
        outer.finish(outcome="inconclusive")
    write_json(root,"recording-completion-index.json",{
        "role":"recording_callbacks_complete_not_physical_or_scientific_acceptance",
        "episode_denominator":len(expected),"schedule_sha256":schedule_hash,
        "results_sha256":{g:hashlib.sha256((root/f"{g}/results.json").read_bytes()).hexdigest() for g in specs}})
    return {"independent_roots":len(cases),"actual_episode_denominator":len(cases)*sum(CONDITION_REPEATS.values()),
            "classic_episode_denominator":len(records["classic"].grid),
            "llm_episode_denominator":len(records["llm"].grid),"outer_aggregate_rows":len(outer.grid),
            "conditions":dict(CONDITION_REPEATS),"primary_inference_ready":False}
