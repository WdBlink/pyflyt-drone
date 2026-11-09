"""Offline event endpoints on original target windows; never actor input.

The caller must admit full source/input/clock/score receipts first. These pure
functions do not run a mission, change scores, declare G4 or infer independence.
"""

from __future__ import annotations

import math

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
    candidates = sorted({v.arrived_at for v in versions if event_time < v.arrived_at <= deadline-30})
    for stamp in candidates:
        checks = [stamp, *(v.arrived_at for v in versions if stamp < v.arrived_at <= stamp+30)]
        if all(final_static_quality(versions, entity.origin_xy, entity.true_class, t) >= .6 for t in checks):
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


def packet_response(world, case, entity, event_time, deadline):
    hidden = {r["evidence_id"]: r["entity_key"] for r in world["sensor_audit"]}
    candidates = []
    for obs in world["delivered_observations"]:
        measured = obs["measured_at"]
        if (hidden.get(obs["evidence_id"]) != entity.key or obs["quality"] < .8
                or not event_time < measured <= deadline or obs["arrived_at"] > deadline
                or not _connected(case, obs["drone_id"], measured)):
            continue
        version = 2 if case.priority_at is not None and measured >= case.priority_at else 1
        for job in world["jobs"]:
            params = job["params"]
            if (job["kind"] not in ("survey", "inspect", "observe") or job["drone_id"] != obs["drone_id"]
                    or job["created_at"] <= event_time or job["started_at"] is None or job["started_at"] > measured
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
    for index, version in enumerate(versions):
        if version.arrived_at <= event_time:
            continue
        previous = final_static_quality(versions[:index], entity.origin_xy, entity.true_class, version.arrived_at)
        current = final_static_quality(versions[:index+1], entity.origin_xy, entity.true_class, version.arrived_at)
        if current > max(0., previous):
            return version.arrived_at
    return None


def analyze_event_trace(world, case):
    """One row per event×required target, including every NA/censor outcome.

Survey is area work: an affected running survey represents every currently
active outstanding target obligation. Inspect/observe use only candidate↔truth
joins already delivered before the event. No future association supplies
eligibility. This operational interpretation is development-only until review.
    """
    validate_development_case(case)
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
        partial = reconstruct_reports({**world, "reports": [r for r in world["reports"] if r["t"] <= stamp]}, case)
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
                    for o in world["delivered_observations"] if o["arrived_at"] <= stamp
                    and o["candidate_id"] == j["params"].get("candidate_id")}
                if not jobs or not any(j["kind"] == "survey" for j in jobs) and entity.key not in associated:
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
