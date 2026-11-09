"""Staged read-only G4 proof validation. This is not installed or phase access.

A G4Proof alone never opens a dataset or authorizes a holdout draw. Namespace,
dataset, episode and holdout-opening validation must still be wired at every
side-effect boundary before this candidate can be qualified.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


REQUIRED_SOURCES = frozenset({
    "envs/inspection/mission.py", "envs/inspection/scenario.py", "envs/inspection/protocol.py",
    "envs/inspection/scoring.py", "envs/inspection/execution.py", "envs/inspection/control.py",
    "envs/inspection/flight.py", "envs/inspection/planning.py", "envs/inspection/paths.py",
    "envs/inspection/guidance.py", "envs/inspection/reference.py",
    "eval/inspection_cases.py", "eval/inspection_population.py", "eval/inspection_scenarios.py",
    "eval/inspection_formal_admission.py", "eval/prepare_inspection_formal_population.py",
    "eval/run_inspection_formal.py", "eval/run_inspection_reference.py",
    "eval/run_inspection_llm_development.py", "eval/run_inspection_llm_episode.py",
    "eval/inspection_llm_actor.py", "eval/inspection_llm_protocol.py", "eval/inspection_classic.py",
    "eval/inspection_episode_receipts.py", "eval/inspection_llm_episode_receipts.py",
    "eval/inspection_model_service.py", "eval/inspection_model_client.py",
    "eval/inspection_model_receipts.py", "eval/inspection_measurement_loader.py",
    "eval/inspection_process_cleanup.py", "eval/inspection_recording.py",
})


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path, maximum=16 * 1024 * 1024):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate admission JSON field")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite admission JSON")

    path = Path(path)
    if path.stat().st_size > maximum:
        raise ValueError("admission JSON exceeds its explicit input bound")
    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=nonfinite)


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str

    @classmethod
    def parse(cls, value):
        if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
                or type(value["path"]) is not str or not value["path"]
                or Path(value["path"]).is_absolute() or ".." in Path(value["path"]).parts
                or type(value["sha256"]) is not str or re.fullmatch("[0-9a-f]{64}", value["sha256"]) is None):
            raise ValueError("a run-relative artifact and exact SHA256 are required")
        return cls(**value)

    def checked_path(self, root):
        root = Path(root).resolve()
        supplied = root/self.path
        path = supplied.resolve()
        if not path.is_relative_to(root) or not path.is_file() or supplied.is_symlink():
            raise ValueError("missing or escaping admission artifact")
        if file_sha(path) != self.sha256:
            raise ValueError("admission artifact changed")
        return path

    def json(self, root):
        return read_json(self.checked_path(root))


@dataclass(frozen=True)
class G4Proof:
    """Only a validated qualification subject; not a namespace capability."""

    acceptance: Artifact
    subject: Artifact
    review: Artifact
    source_manifest: Artifact
    frozen_plan: str
    design: Artifact
    population: Artifact
    registry: Artifact
    native_request_sha256: str


def require_native_binding(owner, finding_id, artifacts):
    if type(finding_id) is not str or not finding_id:
        raise ValueError("explicit native finding identity required")
    matches = []
    for row in owner.get("rounds", []):
        request, decision = row.get("request", {}), row.get("decision", {})
        if decision.get("finding", {}).get("id") != finding_id:
            continue
        issued = request.get("request_sha256")
        subject = {k: v for k, v in request.items() if k != "request_sha256"}
        if (type(issued) is not str or issued != canonical_sha(subject)
                or decision.get("request_sha256") != issued
                or row.get("validation") != "structure_and_binding_only"
                or decision.get("status") != "active" or decision.get("progress") != "verified_progress"
                or any(request.get("evidence", {}).get(a.path) != a.sha256 for a in artifacts)):
            raise ValueError("qualification is not bound by its actual applied native round")
        matches.append(issued)
    if len(matches) != 1:
        raise ValueError("a unique applied qualification finding is required")
    return matches[0]


def read_g4_proof(run, project_root, acceptance_value, *, for_dispatch=True):
    """Validate immutable subjects and their native application without writes.

    Dispatch additionally requires an active owner round. Offline inspection
    can omit that liveness check; neither mode generates or opens any dataset.
    """
    root, project = Path(run).resolve(), Path(project_root).resolve()
    if not root.is_relative_to(project) or type(for_dispatch) is not bool:
        raise ValueError("explicit project/run identity and dispatch mode required")
    acceptance = Artifact.parse(acceptance_value)
    value = acceptance.json(root)
    if (set(value) != {"schema", "status", "subject", "review", "native_finding_id"}
            or value["schema"] != "inspection-g4-acceptance-v1" or value["status"] != "qualified"):
        raise ValueError("a typed G4 qualification acceptance is required")
    subject_ref, review_ref = Artifact.parse(value["subject"]), Artifact.parse(value["review"])
    subject, review = subject_ref.json(root), review_ref.json(root)
    if (subject.get("schema") != "inspection-g4-subject-v1"
            or review.get("schema") != "inspection-g4-independent-review-v1"
            or review.get("decision") != "accept_declared_simulation_scope"
            or review.get("subject_sha256") != subject_ref.sha256
            or review.get("unresolved_blocking_findings") != []
            or type(review.get("reviewer")) is not str or not review["reviewer"]):
        raise ValueError("independent review must bind this exact G4 subject without open blockers")
    owner = read_json(root/".paper/control.json", 128 * 1024 * 1024)
    plans = [row["packet"] for row in owner.get("receipts", []) if row.get("stage") == "plan"]
    active = owner.get("active_round")
    if (len(plans) != 1 or owner.get("frozen_plan") != canonical_sha(plans[0])
            or owner.get("task_stop") is not None
            or for_dispatch and (type(active) is not dict or type(active.get("id")) is not str
                                 or not active["id"] or type(active.get("stage")) is not str or not active["stage"])):
        raise ValueError("accepted plan and live dispatch owner required")
    identity = {k: owner[k] for k in ("created_at", "brief_sha256", "frozen_plan")}
    if subject.get("owner") != identity:
        raise ValueError("qualification belongs to a different owner or scientific contract")
    plan = plans[0]
    design_ref = Artifact(plan["experiment_design"], plan["experiment_design_sha256"])
    registry_ref = Artifact(plan["local_model_registry"], plan["local_model_registry_sha256"])
    design = design_ref.json(root)
    registry_ref.checked_path(root)
    population_ref = Artifact.parse(design["split_contract"]["population"])
    population_ref.checked_path(root)
    for key, expected in (("design", design_ref), ("population", population_ref), ("registry", registry_ref)):
        if Artifact.parse(subject.get(key)) != expected:
            raise ValueError("G4 qualification changed an accepted scientific subject")
    source_ref = Artifact.parse(subject.get("source_manifest"))
    closure = source_ref.json(root)
    if (closure.get("schema") != "inspection-final-source-closure-v1"
            or type(closure.get("sources")) is not dict or not closure["sources"]
            or not REQUIRED_SOURCES <= closure["sources"].keys()
            or type(closure.get("dependencies")) is not dict or not closure["dependencies"]):
        raise ValueError("complete qualified source/dependency closure required")
    for name, expected in closure["sources"].items():
        Artifact.parse({"path": name, "sha256": expected}).checked_path(project)
    for name, expected in closure["dependencies"].items():
        if (type(name) is not str or not Path(name).is_absolute()
                or type(expected) is not str or re.fullmatch("[0-9a-f]{64}", expected) is None
                or file_sha(name) != expected):
            raise ValueError("qualified dependency changed")
    evidence = subject.get("qualification_evidence")
    if type(evidence) is not list or not evidence:
        raise ValueError("qualification evidence cannot be replaced by a status label")
    for item in evidence:
        Artifact.parse(item).checked_path(root)
    Artifact.parse(subject.get("hardware_profile")).checked_path(root)
    request = require_native_binding(owner, value["native_finding_id"], [acceptance, subject_ref, review_ref])
    return G4Proof(acceptance, subject_ref, review_ref, source_ref, identity["frozen_plan"],
                   design_ref, population_ref, registry_ref, request)
