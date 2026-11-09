"""Task tools and evaluator-owned evidence. Responses contain no truth verdict."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Callable

from envs.inspection.scoring import GradingMeasurement


def audit_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_numeric": repr(value)}
    if isinstance(value, dict):
        return {str(k): audit_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [audit_value(v) for v in value]
    return value


@dataclass(frozen=True)
class Observation:
    evidence_id: str
    candidate_id: str
    drone_id: int
    measured_at: float
    arrived_at: float
    estimate_xy: tuple[float, float]
    estimated_class: str
    object_type: str
    quality: float
    covariance_xy: tuple[float, float]
    sensor_mode: str
    aircraft_xy: tuple[float, float]
    camera_xyz: tuple[float, float, float] | None = None

    def public(self):
        return asdict(self)


@dataclass(frozen=True)
class Evidence:
    observation: Observation
    entity_key: str | None  # evaluator-only; never serialized to actor


@dataclass
class Report:
    report_id: str
    object_type: str
    associated_key: str | None
    created_at: float
    versions: list[GradingMeasurement] = field(default_factory=list)
    withdrawn: bool = False


class ReportLedger:
    def __init__(self, budget: int, classes: set[str], truth_at: Callable):
        self.budget, self.classes, self.truth_at = budget, classes, truth_at
        self.evidence: dict[str, Evidence] = {}
        self.reports: dict[str, Report] = {}
        self.audit = []
        self.association_ambiguities = 0

    def register(self, evidence: Evidence):
        key = evidence.observation.evidence_id
        if key in self.evidence and self.evidence[key] != evidence:
            raise ValueError("minted evidence is immutable")
        self.evidence[key] = evidence

    def submit(self, request: dict, now: float) -> dict:
        required = {"report_id", "object_type", "candidate_id", "measured_at", "estimate_xy", "claimed_class", "evidence_ids"}
        try:
            if set(request) != required:
                raise ValueError("report schema fields do not match")
            report_id = request["report_id"]
            if not isinstance(report_id, str) or not 1 <= len(report_id) <= 64:
                raise ValueError("invalid report ID")
            if request["object_type"] not in ("static", "dynamic") or request["claimed_class"] not in self.classes:
                raise ValueError("unknown object type or class")
            if not isinstance(request["candidate_id"], str):
                raise ValueError("candidate handle must be a string")
            xy, stamp = request["estimate_xy"], request["measured_at"]
            if not isinstance(xy, (list, tuple)) or len(xy) != 2:
                raise ValueError("planar estimate required")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (*xy, stamp)):
                raise ValueError("finite numerical estimate and time required")
            ids = request["evidence_ids"]
            if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
                raise ValueError("evidence IDs must be a unique string list")
            existing = self.reports.get(report_id)
            if existing and existing.object_type != request["object_type"]:
                raise ValueError("a report cannot change object type")
            if not existing and len(self.reports) >= self.budget:
                raise ValueError("report capacity exhausted")
        except (ValueError, TypeError) as error:
            response = {"status": "rejected", "reason": str(error)}
            self.audit.append(audit_value({"t": now, "request": request, "response": response}))
            return response
        references = [self.evidence[i] for i in ids if i in self.evidence]
        causal = (bool(ids) and len(references) == len(ids) and 0 <= stamp <= now
                  and all(e.observation.measured_at <= stamp and e.observation.arrived_at <= now for e in references)
                  and any(e.observation.measured_at == stamp for e in references))
        if existing is None:
            occupied = {r.associated_key for r in self.reports.values() if r.associated_key is not None}
            matches = []
            if 0 <= stamp <= now:
                for key, position in self.truth_at(stamp).items():
                    distance = math.hypot(xy[0]-position[0], xy[1]-position[1])
                    if distance <= 20 and key not in occupied:
                        matches.append((distance, key))
            matches.sort()
            self.association_ambiguities += int(len(matches) > 1)
            existing = Report(report_id, request["object_type"], matches[0][1] if matches else None, now)
            self.reports[report_id] = existing
        identity_correct = (existing.associated_key is not None and bool(references)
                            and all(e.entity_key == existing.associated_key
                                    and e.observation.candidate_id == request["candidate_id"] for e in references))
        quality_valid = bool(references) and all(e.observation.quality >= .8 for e in references)
        # An invalid/future claim consumes capacity but cannot enter a score.
        safe_stamp = max(0.0, min(float(stamp), now))
        new_evidence_ids = sorted(e.observation.evidence_id for e in references if e.observation.measured_at == stamp)
        measurement_id = new_evidence_ids[0] if new_evidence_ids else "unsupported-"+str(len(existing.versions))
        version = GradingMeasurement(measurement_id, safe_stamp, now, tuple(xy), request["claimed_class"],
                                     identity_correct, quality_valid, causal, len(existing.versions)+1)
        if existing.object_type == "dynamic":
            previous = next((v for v in existing.versions if v.evidence_id == measurement_id), None)
            if previous is not None:
                same = (previous.measured_at == version.measured_at and previous.estimate_xy == version.estimate_xy
                        and previous.claimed_class == version.claimed_class)
                response = {"status": "accepted" if same else "rejected", "report_id": report_id,
                            "reason": "idempotent frozen measurement" if same else "dynamic measurements are sealed"}
                self.audit.append(audit_value({"t": now, "request": request, "response": response}))
                return response
        existing.versions.append(version)
        existing.withdrawn = False
        response = {"status": "accepted", "report_id": report_id,
                    "slots_used": len(self.reports), "slots_remaining": self.budget-len(self.reports)}
        self.audit.append(audit_value({"t": now, "request": request, "response": response}))
        return response

    def withdraw(self, report_id):
        if report_id not in self.reports:
            return {"status": "rejected", "reason": "unknown report"}
        self.reports[report_id].withdrawn = True
        return {"status": "accepted", "slots_used": len(self.reports)}


@dataclass
class Job:
    job_id: str
    drone_id: int
    kind: str
    version: int
    created_at: float
    params: dict
    status: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    reason: str | None = None


class JobBook:
    def __init__(self, count):
        self.versions = [0]*count
        self.jobs: dict[str, Job] = {}
        self.active: dict[int, str] = {}
        self.transitions = []

    def create(self, drone_id, kind, params, expected_version, now, *, preempt=False):
        if isinstance(drone_id, bool) or not isinstance(drone_id, int) or not 0 <= drone_id < len(self.versions):
            return {"status": "rejected", "reason": "unknown aircraft"}
        if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version != self.versions[drone_id]:
            return {"status": "rejected", "reason": "stale version", "version": self.versions[drone_id]}
        if drone_id in self.active:
            old = self.jobs[self.active[drone_id]]
            if old.status in ("queued", "running"):
                if not preempt:
                    return {"status": "rejected", "reason": "aircraft has an active job"}
                self.transition(old, "cancelled", now, "preempted by safe exit")
        self.versions[drone_id] += 1
        job_id = f"job-{len(self.jobs)+1:04d}"
        job = Job(job_id, drone_id, kind, self.versions[drone_id], now, dict(params))
        self.jobs[job_id] = job
        self.active[drone_id] = job_id
        self.transitions.append({"t": now, "job_id": job_id, "status": "queued"})
        return {"status": "accepted", "job_id": job_id, "version": job.version}

    def transition(self, job, status, now, reason=None):
        legal = {"queued": {"running", "cancelled", "failed"},
                 "running": {"succeeded", "failed", "cancelled"}}
        if status not in legal.get(job.status, set()):
            raise ValueError("illegal job state transition")
        job.status, job.reason = status, reason
        if status == "running":
            job.started_at = now
        else:
            job.finished_at = now
        self.transitions.append({"t": now, "job_id": job.job_id, "status": status, "reason": reason})

    def cancel(self, job_id, expected_version, now):
        job = self.jobs.get(job_id)
        if job is None:
            return {"status": "rejected", "reason": "unknown job"}
        if expected_version != self.versions[job.drone_id] or isinstance(expected_version, bool) or not isinstance(expected_version, int):
            return {"status": "rejected", "reason": "stale version"}
        if job.status not in ("queued", "running"):
            return {"status": "rejected", "reason": "job already terminal"}
        self.transition(job, "cancelled", now, "actor cancellation")
        self.versions[job.drone_id] += 1
        return {"status": "accepted", "version": self.versions[job.drone_id]}

    def public(self):
        return [asdict(job) for job in self.jobs.values()]
