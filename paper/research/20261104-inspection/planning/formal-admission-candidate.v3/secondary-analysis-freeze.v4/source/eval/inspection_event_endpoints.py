"""Offline event endpoints on original target windows; never actor input.

The caller must admit full source/input/clock/score receipts first. These pure
functions do not run a mission, change scores, declare G4 or infer independence.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import median

from envs.inspection.scoring import final_static_quality, grade_dynamic_bins
from eval.inspection_common_calendar import _score, reconstruct_reports
from eval.inspection_population import validate_development_case
from eval.inspection_cases import weight_at


def endpoint(event_time, deadline, observed_time):
    if (any(type(t) not in (int, float) or not math.isfinite(t) for t in (event_time, deadline))
            or not 0 <= event_time <= deadline <= 600
            or observed_time is not None and (type(observed_time) not in (int, float)
                or not math.isfinite(observed_time) or not event_time <= observed_time <= deadline)):
        raise ValueError("invalid original event follow-up window")
    return {"time": deadline if observed_time is None else observed_time,
        "elapsed": (deadline if observed_time is None else observed_time)-event_time,
        "censored": observed_time is None, "followup_horizon": deadline-event_time}


def static_recovery(versions, entity, event_time, deadline=600):
    """First qualifying report, with a full subsequent30s stable by deadline."""
    for index, candidate in enumerate(versions):
        stamp = candidate.arrived_at
        if not event_time <= stamp <= deadline-30:
            continue
        # Check each receipt prefix. A later good version with the same clock
        # stamp cannot erase an earlier bad version in the holding interval.
        checks = [(j, v.arrived_at) for j, v in enumerate(versions) if j >= index and v.arrived_at <= stamp+30]
        if all(final_static_quality(versions[:j+1], entity.origin_xy, entity.true_class, t) >= .6 for j, t in checks):
            return endpoint(event_time, deadline, stamp)
    return endpoint(event_time, deadline, None)


def dynamic_recovery(original_bins, versions, event_time, deadline):
    """Use exactly ten consecutive selected ORIGINAL bins, never reset a grid."""
    if [b["index"] for b in original_bins] != list(range(1, 61)):
        raise ValueError("all60 original required bins are mandatory")
    for i, b in enumerate(original_bins):
        if (type(b["index"]) is not int or any(type(b[k]) not in (int, float) or not math.isfinite(b[k])
                for k in ("starts_at", "ends_at", "quality")) or not 0 <= b["quality"] <= 1
                or b["ends_at"]-b["starts_at"] != 3
                or i > 0 and b["starts_at"] != original_bins[i-1]["ends_at"]):
            raise ValueError("original finite3s bin grid differs")
    measurements = {v.evidence_id: v for v in versions}
    times = []
    for start in range(51):
        window = original_bins[start:start+10]
        if window[0]["starts_at"] < event_time or any(b["quality"] < .5 for b in window):
            continue
        if any(b["evidence_id"] not in measurements for b in window):
            raise ValueError("selected bin has no original sealed measurement")
        stamp = max(window[-1]["ends_at"], *(measurements[b["evidence_id"]].arrived_at for b in window))
        if event_time <= stamp <= deadline:
            times.append(stamp)
    return endpoint(event_time, deadline, min(times) if times else None)


def event_calendar(case):
    if case.level != "L3":
        return []
    return sorted([{"type": kind, "time": stamp, "drone_id": drone} for kind, stamp, drone in (
        ("priority", case.priority_at, None), ("link_start", case.link_start, case.link_drone),
        ("link_end", case.link_end, case.link_drone), ("camera_start", case.camera_start, case.camera_drone),
        ("camera_end", case.camera_end, case.camera_drone), ("retirement", case.retire_at, case.retire_drone))],
        key=lambda e: (e["time"], e["type"]))


def _running_before(job, stamp):
    return (job["kind"] in ("inspect", "observe", "survey") and job["created_at"] < stamp
        and job["started_at"] is not None and job["started_at"] < stamp
        and (job["finished_at"] is None or job["finished_at"] >= stamp))


def _connected(case, drone, stamp):
    return not (case.retire_drone == drone and stamp >= case.retire_at or
        case.link_drone == drone and case.link_start <= stamp < case.link_end)


def survey_region(entity, stamp):
    """Fixed public responsibility partition, independent of sensing success."""
    point = entity.position(stamp)
    if point is None:
        return None
    centers = (-300., -100., 100., 300.)
    return "stripe-"+str(min(range(4), key=lambda i: (abs(point[1]-centers[i]), i)))


def packet_response(world, case, entity, event_time, deadline):
    hidden = {r["evidence_id"]: r["entity_key"] for r in world["sensor_audit"]}
    candidates = []
    for obs in world["delivered_observations"]:
        measured = obs["measured_at"]
        if (hidden.get(obs["evidence_id"]) != entity.key or obs["quality"] < .8
                or not event_time < measured <= deadline or obs["arrived_at"] > deadline
                or not _connected(case, obs["drone_id"], measured)):
            continue
        for job in world["jobs"]:
            params = job["params"]
            version = 2 if case.priority_at is not None and job["created_at"] >= case.priority_at else 1
            if (job["kind"] not in ("survey", "inspect", "observe") or job["drone_id"] != obs["drone_id"]
                    or job["created_at"] < event_time or job["started_at"] is None or job["started_at"] > measured
                    or job["finished_at"] is not None and job["finished_at"] < measured
                    or params.get("task_card_version") != version or params.get("rule_id") != entity.rule_id):
                continue
            if job["kind"] != "survey" and (params.get("candidate_id") != obs["candidate_id"] or obs["sensor_mode"] != "inspect"):
                continue
            if job["kind"] == "survey" and obs["sensor_mode"] != "survey":
                continue
            candidates.append((measured, obs["evidence_id"], job["job_id"]))
    return min(candidates) if candidates else None


def static_revision_response(report, entity, event_time):
    """An existing report ID must deliver a positive quality improvement."""
    if report is None or report.created_at >= event_time:
        return None
    versions = report.versions
    baseline = final_static_quality([v for v in versions if v.arrived_at < event_time],
        entity.origin_xy, entity.true_class, event_time)
    for index, version in enumerate(versions):
        if version.arrived_at < event_time:
            continue
        previous = final_static_quality(versions[:index], entity.origin_xy, entity.true_class, version.arrived_at)
        current = final_static_quality(versions[:index+1], entity.origin_xy, entity.true_class, version.arrived_at)
        if current > max(0., baseline, previous):
            return version.arrived_at
    return None


def _analyze_event_trace(world, case):
    """One row per event×required target, including every NA/censor outcome.

Survey uses the fixed public four-stripe responsibility partition at the
event position, without conditioning on detection. Inspect/observe use candidate↔truth
joins already delivered before the event. No future association supplies
eligibility. This operational interpretation is development-only until review.
    """
    if case.level != "L3" or world["private_case"] != case.as_dict() or world["case_sha256"] != case.digest():
        # JSON tuple/list round-trips are compared by the canonical digest below.
        import json
        if (case.level != "L3" or json.dumps(world["private_case"], sort_keys=True) != json.dumps(case.as_dict(), sort_keys=True)
                or world["case_sha256"] != case.digest()):
            raise ValueError("event analysis must use its actual complete L3 calendar")
    if _score(world, case, case) != world["score"]:
        raise ValueError("original score must reproduce before event analysis")
    ledger = reconstruct_reports(world, case)
    hidden = {r["evidence_id"]: r["entity_key"] for r in world["sensor_audit"]}
    output = []
    for event in event_calendar(case):
        stamp = event["time"]
        partial = reconstruct_reports({**world, "reports": [r for r in world["reports"] if r["t"] < stamp]}, case)
        for entity in case.entities:
            if not entity.rule_id:
                continue
            deadline = min(600., entity.appeared_at+186) if entity.object_type == "dynamic" else 600.
            report = next((r for r in ledger.reports.values() if r.associated_key == entity.key
                and r.object_type == entity.object_type and not r.withdrawn), None)
            current_report = next((r for r in partial.reports.values() if r.associated_key == entity.key
                and r.object_type == entity.object_type and not r.withdrawn), None)
            versions = current_report.versions if current_report else []
            if entity.object_type == "static":
                quality = final_static_quality(versions, entity.origin_xy, entity.true_class, stamp)
            else:
                quality, _ = grade_dynamic_bins(versions, entity.position, entity.true_class, entity.appeared_at, 600,
                    weight_at=lambda t: weight_at(case, entity, t))
            reason = None
            if entity.appeared_at > stamp:
                reason = "not_yet_active"
            elif deadline <= stamp:
                reason = "target_followup_ended"
            elif event["type"] == "priority":
                if entity.rule_id not in ("static-0", "dynamic-inspection"):
                    reason = "rule_unchanged"
                elif quality >= (.6 if entity.object_type == "static" else .5):
                    reason = "already_satisfied"
            else:
                jobs = [j for j in world["jobs"] if j["drone_id"] == event["drone_id"] and _running_before(j, stamp)]
                associated = {hidden.get(o["evidence_id"]) for j in jobs if j["kind"] != "survey"
                    for o in world["delivered_observations"] if o["arrived_at"] < stamp
                    and o["candidate_id"] == j["params"].get("candidate_id")}
                in_survey = any(j["kind"] == "survey" and j["params"].get("region_id") == survey_region(entity, stamp)
                    for j in jobs)
                if not in_survey and entity.key not in associated:
                    reason = "no_affected_active_workload"
                elif entity.object_type == "static" and quality >= .6:
                    reason = "already_satisfied"
            row = {"case_id": case.case_id, "lineage_group": case.lineage_group, "event_type": event["type"],
                "event_time": stamp, "affected_drone": event["drone_id"], "entity_key": entity.key,
                "object_type": entity.object_type, "deadline": deadline, "quality_received_at_event": quality,
                "eligible": reason is None, "NA_reason": reason}
            if reason is None:
                packet = packet_response(world, case, entity, stamp, deadline)
                revision = static_revision_response(report, entity, stamp) if entity.object_type == "static" else None
                observed = min([v for v in (packet[0] if packet else None, revision) if v is not None], default=None)
                row["response"] = endpoint(stamp, deadline, observed)
                row["response_packet_witness"] = packet
                row["response_static_revision_time"] = revision
                all_versions = report.versions if report else []
                if entity.object_type == "static":
                    row["recovery"] = static_recovery(all_versions, entity, stamp, deadline)
                else:
                    original = next(r for r in world["score"]["dynamic_entities"] if r["entity_key"] == entity.key)
                    row["recovery"] = dynamic_recovery(original["bins"], all_versions, stamp, deadline)
            else:
                row["response"] = row["recovery"] = None
            output.append(row)
    return {"role": "development offline event×target endpoints, not independent samples or G4",
        "case_sha256": case.digest(), "rows": output, "event_count": len(event_calendar(case))}


def analyze_event_trace(world, case):
    validate_development_case(case)
    return _analyze_event_trace(world, case)


def _event_apply_time(case, event):
    if event["type"] not in ("priority", "retirement"):
        return {"t_apply_tick": None, "apply_time_basis": "availability/quality evaluated at sample or delivery time; no mutation tick recorded"}
    index = math.floor(event["time"] * 120)
    while index / 120 < event["time"]:
        index += 1
    return {"t_apply_tick": index / 120,
        "apply_time_basis": "inferred first control boundary satisfying the registered time, from frozen head/tail event checks"}


def _creation_witness(world, job):
    kinds = {"assign_survey": "survey", "inspect_candidate": "inspect", "observe_track": "observe",
             "handoff_observation": "observe"}
    params = job["params"]
    if (type(job["drone_id"]) is not int or not 0 <= job["drone_id"] < 4
            or type(job["version"]) is not int or job["version"] < 1
            or type(job["job_id"]) is not str or type(params.get("task_card_version")) is not int):
        return None
    same = lambda left, right: type(left) is type(right) and left == right
    matches = []
    for index, call in enumerate(world["tool_calls"]):
        response, args = call["response"], call["args"]
        if (kinds.get(call["name"]) != job["kind"] or response.get("status") != "accepted"
                or not same(response.get("job_id"), job["job_id"])
                or not same(response.get("version"), job["version"])
                or not same(call["t"], job["created_at"])):
            continue
        drone = args.get("to_drone") if call["name"] == "handoff_observation" else args.get("drone_id")
        expected = args.get("expected_to_version") if call["name"] == "handoff_observation" else args.get("expected_version")
        if (type(drone) is not int or type(expected) is not int or type(args.get("task_card_version")) is not int
                or drone != job["drone_id"] or expected != job["version"] - 1
                or any(not same(args.get(key), params.get(key)) for key in ("rule_id", "task_card_version"))):
            continue
        if call["name"] == "handoff_observation" and (type(args.get("from_drone")) is not int
                or not 0 <= args["from_drone"] < 4 or args["from_drone"] == drone
                or type(args.get("expected_from_version")) is not int or args["expected_from_version"] < 0):
            continue
        keys = ("region_id", "duration") if job["kind"] == "survey" else (
            ("candidate_id",) if job["kind"] == "inspect" else ("candidate_id", "duration", "report_id"))
        if all(same(args.get(key), params.get(key)) for key in keys):
            matches.append(index)
    return matches[0] if len(matches) == 1 else None


def _serves_entity(world, case, entity, job, event_time):
    if job["kind"] == "survey":
        return job["params"].get("region_id") == survey_region(entity, event_time)
    visible = [row for row in world["delivered_observations"]
        if row["candidate_id"] == job["params"].get("candidate_id") and row["arrived_at"] <= job["created_at"]]
    if not visible:
        return None
    hidden = {row["evidence_id"]: row["entity_key"] for row in world["sensor_audit"]}
    evidence_id = visible[-1]["evidence_id"]
    return hidden[evidence_id] == entity.key if evidence_id in hidden else None


def _adoption_response(world, case, entity, row, coverage):
    """Conditional source-chain inference, never direct job/setpoint telemetry."""
    stamp, deadline = row["event_time"], row["deadline"]
    candidates, gaps, excluded = [], [], []
    audit, transitions = world["execution_audit"], world["job_transitions"]
    for job in world["jobs"]:
        created, params = job["created_at"], job["params"]
        if (job["kind"] not in ("survey", "inspect", "observe") or not stamp <= created <= deadline
                or params.get("rule_id") != entity.rule_id):
            continue
        serves = _serves_entity(world, case, entity, job, stamp)
        if serves is False:
            continue
        creation = _creation_witness(world, job)
        version = 2 if case.priority_at is not None and created >= case.priority_at else 1
        if serves is None or creation is None or params.get("task_card_version") != version:
            gaps.append({"job_id": job["job_id"], "reason": "causal candidate/tool/card binding not identifiable"})
            continue
        running = [entry for entry in transitions if entry["job_id"] == job["job_id"] and entry["status"] == "running"]
        commits = [(index, entry) for index, entry in enumerate(audit)
                   if entry["type"] == "program_committed" and entry["job_id"] == job["job_id"]]
        for index, commit in commits:
            t = commit["at"]
            if not created <= t <= deadline:
                continue
            if sum(entry["type"] == "program_committed" and entry.get("serial") == commit["serial"]
                   for entry in audit) != 1:
                gaps.append({"job_id": job["job_id"], "reason": "commit serial is not unique in the original audit"})
                continue
            n = round(t * 120)
            terminal_same_advance = any(entry["job_id"] == job["job_id"] and entry["t"] == t
                and entry["status"] in ("succeeded", "failed", "cancelled") for entry in transitions)
            if terminal_same_advance or (n + 1) / 120 > world["simulated_seconds"]:
                excluded.append({"job_id": job["job_id"], "at": t,
                    "reason": "same-advance terminal" if terminal_same_advance else "no subsequent actual control step"})
                continue
            requests = [(i, entry) for i, entry in enumerate(audit[:index])
                if entry["type"] == "plan_request" and entry["serial"] == commit["serial"]]
            responses = [(i, entry) for i, entry in enumerate(audit[:index])
                if entry["type"] == "plan_response" and entry["serial"] == commit["serial"]]
            active = [entry for entry in transitions if entry["job_id"] == job["job_id"] and entry["t"] <= t]
            if len(requests) != 1 or len(responses) != 1:
                gaps.append({"job_id": job["job_id"], "reason": "ordered plan-request/response/commit chain incomplete"})
                continue
            request_index, request = requests[0]; response_index, response = responses[0]
            tokens = request.get("tokens", [])
            program_sha = commit.get("program_sha256")
            if (not request_index < response_index < index or not created <= request["now"] <= response["received_at"] <= t
                    or request["job_id"] != job["job_id"] or request["drone"] != job["drone_id"]
                    or commit["drone"] != job["drone_id"] or response.get("fresh") is not True
                    or response.get("program_sha256") != program_sha
                    or type(program_sha) is not str or len(program_sha) != 64 or set(program_sha) - set("0123456789abcdef")
                    or commit["revision"] != request["revision"] + 1 or len(tokens) != 4
                    or list(tokens[job["drone_id"]]) != [job["version"], job["job_id"]]
                    or abs(t - n / 120) > 1e-9 or request["commit_at"] > t + 1e-8
                    or not active or active[-1]["status"] != "running"
                    or job["started_at"] is None or job["started_at"] > t
                    or coverage["physics_samples"] != 144000):
                gaps.append({"job_id": job["job_id"], "reason": "commit program/phase/version/actual-step binding not identifiable"})
                continue
            candidates.append({"time": t, "kind": "program_commit_control_tick_inference", "job_id": job["job_id"],
                "tool_call_index": creation, "job_version": job["version"], "drone_id": job["drone_id"],
                "request_audit_index": request_index, "response_audit_index": response_index,
                "commit_audit_index": index, "serial": commit["serial"], "program_sha256": program_sha,
                "first_subsequent_physics_hook": (2 * n + 1) / 240, "control_step_end": (n + 1) / 120,
                "precision": "120Hz control inference; tracking rows contain no job_id or setpoint"})
        for transition in running:
            if transition.get("reason") != "actual local acquisition on committed flight":
                continue
            t = transition["t"]
            if (not created < t <= deadline or params.get("service_started_at") != t
                    or t != round(t * 2) / 2):
                gaps.append({"job_id": job["job_id"], "reason": "local acquisition creation/sample phase ambiguous"})
                continue
            frame = [entry for entry in world["sensor_audit"]
                if entry["t"] == t and entry["drone_id"] == job["drone_id"] and entry["mode"] == "inspect"]
            if len(frame) != 1 or frame[0]["quality"] < .8:
                gaps.append({"job_id": job["job_id"], "reason": "unique qualified narrow local frame unavailable"})
                continue
            obs = next((entry for entry in world["delivered_observations"]
                if entry["evidence_id"] == frame[0]["evidence_id"]), None)
            if obs is not None and (obs["candidate_id"] != params["candidate_id"]
                    or obs["object_type"] != params.get("object_type") or obs["measured_at"] != t):
                gaps.append({"job_id": job["job_id"], "reason": "local frame and candidate/object-type packet disagree"})
                continue
            candidates.append({"time": t, "kind": "local_acquisition_on_existing_continuation", "job_id": job["job_id"],
                "tool_call_index": creation, "job_version": job["version"], "drone_id": job["drone_id"],
                "evidence_id": frame[0]["evidence_id"], "service_started_at": t,
                "candidate_binding": "delivered observation" if obs is not None else "unique narrow frame under frozen active-job candidate/type gate",
                "precision": "actual 2Hz local sampling transition; does not wait for ground upload"})
        if running and not commits and not any(entry.get("reason") == "actual local acquisition on committed flight" for entry in running):
            gaps.append({"job_id": job["job_id"], "reason": "running job lacks an identifiable adoption path"})
    if candidates:
        earliest = min(candidates, key=lambda item: (item["time"], item["kind"], item["job_id"]))
        # An unresolved earlier path can invalidate a claim of FIRST adoption.
        if gaps:
            return {"status": "adoption_not_identifiable_from_trace", "endpoint": None,
                "identified_candidates": candidates, "trace_gaps": gaps, "excluded_markers": excluded}
        return {"status": "identified", "endpoint": endpoint(stamp, deadline, earliest["time"]),
            "witness": earliest, "excluded_markers": excluded}
    return {"status": "adoption_not_identifiable_from_trace" if gaps else "right_censored",
        "endpoint": None if gaps else endpoint(stamp, deadline, None), "trace_gaps": gaps, "excluded_markers": excluded}


def analyze_formal_event_trace(world, case, admitted_parent_sha256):
    """Explicit formal adapter; development validation is never bypassed."""
    from eval.inspection_population import validate_formal_population_case
    from eval.inspection_formal_intervention import SECONDARY_ROOTS
    from eval.inspection_tracking import recount_tracking
    validate_formal_population_case(case, "main", case.case_id)
    if case.case_id not in SECONDARY_ROOTS or case.digest() != admitted_parent_sha256 or case.level != "L3":
        raise ValueError("the fixed admitted original L3 event parent is required")
    if (world["simulated_seconds"] != 600 or world["physics_steps_executed"] != 144000
            or world["physics_hz"] != 240 or world["control_hz"] != 120):
        raise ValueError("full original 600s/240Hz physics/120Hz control coverage required")
    coverage = recount_tracking(world, expected_xy_tube_m=10.)
    if coverage["physics_samples"] != 144000:
        raise ValueError("complete actual tracking coverage required for adoption inference")
    result = _analyze_event_trace(world, case)
    reports = reconstruct_reports(world, case)
    entities = {entity.key: entity for entity in case.entities}
    calendar = {event["type"]: event for event in event_calendar(case)}
    for row in result["rows"]:
        row["information_response"] = row.pop("response")
        row["information_response_basis"] = "first qualifying measured observation or positive static revision; not job adoption"
        if row["NA_reason"] == "already_satisfied":
            row["NA_reason"] = "pre_event_threshold_met"
        row.update(tau_nominal=row["event_time"], t_actor_known=None,
            actor_knowledge_status="exact actor receipt time is not recorded in this world trace",
            **_event_apply_time(case, calendar[row["event_type"]]))
        row["timing_records"] = dict.fromkeys(("job_adoption", "information_response", "recovery"))
        if not row["eligible"]:
            row["job_adoption"] = None
            continue
        entity = entities[row["entity_key"]]
        row["job_adoption"] = _adoption_response(world, case, entity, row, coverage)
        recovery = row["recovery"]
        if recovery["censored"]:
            row["recovery_witness"] = None
        elif entity.object_type == "static":
            row["recovery_witness"] = {"recovery_start": recovery["time"], "confirmed_at": recovery["time"] + 30,
                "basis": "offline complete30s stable follow-up; not online knowledge at recovery_start"}
        else:
            report = next((report for report in reports.reports.values() if report.associated_key == entity.key
                and report.object_type == entity.object_type and not report.withdrawn), None)
            versions = {version.evidence_id: version for version in report.versions} if report else {}
            bins = next(entry["bins"] for entry in world["score"]["dynamic_entities"] if entry["entity_key"] == entity.key)
            windows = [bins[start:start + 10] for start in range(51)]
            selected = next(window for window in windows if window[0]["starts_at"] >= row["event_time"]
                and all(entry["quality"] >= .5 for entry in window)
                and max(window[-1]["ends_at"], *(versions[entry["evidence_id"]].arrived_at for entry in window)) == recovery["time"])
            row["recovery_witness"] = {"original_bin_indices": [entry["index"] for entry in selected],
                "evidence_ids": [entry["evidence_id"] for entry in selected],
                "recovery_start": selected[0]["starts_at"], "confirmed_at": recovery["time"],
                "basis": "ten consecutive original3s bins and their original arrival witnesses; no grid reset"}
        endpoints = {"job_adoption": row["job_adoption"]["endpoint"],
            "information_response": row["information_response"], "recovery": row["recovery"]}
        row["timing_records"] = {name: {"U": value["elapsed"] if value else None,
            "H": row["deadline"] - row["event_time"], "indicator": int(not value["censored"]) if value else None}
            for name, value in endpoints.items()}
    result.update(role="formal offline event/target diagnostics after upstream receipt admission; not independent samples or new tests",
        adoption_scope="conditional frozen single-writer call-chain inference; no direct per-step job/setpoint telemetry; no actor intent/optimality proof",
        tracking_raw_sha256=world["execution_tracking_trace"]["raw_sha256"])
    return result


def summarize_formal_events(episodes):
    """Dependent target/event counts; complete individual timing records stay in ZIP."""
    groups = {}
    for episode in episodes:
        events = episode.get("events")
        if events is None:
            continue
        for row in events["rows"]:
            for root in (episode["case_id"], None):
                key = root, episode["domain"], episode["method"], row["event_type"]
                groups.setdefault(key, []).append(row)

    def distribution(values):
        return {"n": len(values), "min": min(values), "median": median(values), "max": max(values)} if values else {
            "n": 0, "min": None, "median": None, "max": None}

    summaries = []
    for (root, domain, method, event), rows in sorted(groups.items(), key=lambda pair: str(pair[0])):
        eligible = [row for row in rows if row["eligible"]]
        count = len(eligible)
        adoption = dict(Counter(row["job_adoption"]["status"] for row in eligible))
        timing = {}
        for name in ("job_adoption", "information_response", "recovery"):
            values = [row["timing_records"][name] for row in eligible]
            observed = [value["U"] for value in values if value["indicator"] == 1]
            censored = sum(value["indicator"] == 0 for value in values)
            unknown = sum(value["indicator"] is None for value in values)
            timing[name] = {"identified_observed": len(observed), "right_censored": censored,
                "not_identifiable": unknown, "right_censored_rate_among_eligible": censored / count if count else None,
                "not_identifiable_rate_among_eligible": unknown / count if count else None,
                "observed_only_conditional_elapsed_seconds": distribution(observed)}
        summaries.append({"case_id": root, "domain": domain, "method": method, "event_type": event,
            "full_event_target_repeat_denominator": len(rows), "eligible": count,
            "NA_counts": dict(Counter(row["NA_reason"] for row in rows if not row["eligible"])),
            "adoption_status_counts": adoption,
            "adoption_witness_counts": dict(Counter(row["job_adoption"]["witness"]["kind"] for row in eligible
                if row["job_adoption"]["status"] == "identified")),
            "followup_horizon_seconds": distribution([row["deadline"] - row["event_time"] for row in eligible]),
            "timing": timing})
    return {"role": "descriptive_event_counts_with_all_NA_censor_and_trace_gaps_no_new_tests", "groups": summaries,
        "sampling_unit": "original parent; event/target/repeat rows are dependent diagnostics",
        "eligibility": "method-state-dependent; all denominators shown, no paired success/eligibility filtering",
        "timing": "U/H/indicator retained per original row; censored U is observation cutoff, not response time",
        "intervals": "not_computed", "p_values": "not_computed"}
