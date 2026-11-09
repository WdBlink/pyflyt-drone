"""Staged read-only G4 proof validation. This is not installed or phase access.

A G4Proof alone never opens a dataset or authorizes a holdout draw. Namespace,
dataset, episode and holdout-opening validation must still be wired at every
side-effect boundary before this candidate can be qualified.
"""

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys
import types
from typing import Callable


REQUIRED_SOURCES = frozenset({
    "envs/__init__.py", "envs/inspection/__init__.py",
    "envs/inspection/mission.py", "envs/inspection/scenario.py", "envs/inspection/protocol.py",
    "envs/inspection/scoring.py", "envs/inspection/execution.py", "envs/inspection/control.py",
    "envs/inspection/flight.py", "envs/inspection/planning.py", "envs/inspection/paths.py",
    "envs/inspection/guidance.py", "envs/inspection/reference.py",
    "eval/inspection_cases.py", "eval/inspection_population.py", "eval/inspection_scenarios.py",
    "eval/inspection_formal_admission.py", "eval/prepare_inspection_formal_population.py",
    "eval/inspection_formal_namespace.py", "eval/inspection_namespace_admission.py",
    "eval/inspection_dataset_admission.py", "eval/inspection_formal_population.py",
    "eval/inspection_development_catalog.py",
    "eval/inspection_formal_case.py",
    "eval/prepare_inspection_development_population.py",
    "eval/inspection_formal_execution.py", "eval/prepare_inspection_formal_execution.py",
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


def parse_json_bytes(raw):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate admission JSON field")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite admission JSON")

    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("overflowed admission JSON number")
        return value

    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                      parse_constant=nonfinite, parse_float=finite_float)


def bounded_bytes(path, maximum):
    with Path(path).open("rb") as stream:
        raw = stream.read(maximum+1)
    if len(raw) > maximum:
        raise ValueError("admission JSON exceeds its explicit input bound")
    return raw


def read_json(path, maximum=16 * 1024 * 1024):
    return parse_json_bytes(bounded_bytes(path, maximum))


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

    def path_in(self, root):
        root = Path(root).resolve()
        supplied = root/self.path
        path = supplied.resolve()
        current = root
        for part in Path(self.path).parts:
            current = current/part
            if current.is_symlink():
                raise ValueError("admission artifacts cannot traverse symlinks")
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("missing or escaping admission artifact")
        return path

    def checked_path(self, root):
        path = self.path_in(root)
        if file_sha(path) != self.sha256:
            raise ValueError("admission artifact changed")
        return path

    def json(self, root):
        raw = bounded_bytes(self.path_in(root), 16 * 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError("admission JSON artifact changed")
        return parse_json_bytes(raw)


@dataclass(frozen=True)
class VerificationContext:
    """Trusted caller's independently resolved inventory, never read from a G4 claim.

    Production must construct this from the actual source/runtime environment.
    Fixtures explicitly provide a synthetic resolver. Wiring that production
    resolver and all phase boundaries is still a separate staged component.
    """

    project_root: Path
    owner_reader: Callable
    source_names: frozenset
    dependency_names: frozenset
    absent_source_names: frozenset

    def __post_init__(self):
        if (type(self.source_names) is not frozenset or type(self.dependency_names) is not frozenset
                or type(self.absent_source_names) is not frozenset
                or not callable(self.owner_reader) or not REQUIRED_SOURCES <= self.source_names
                or not self.dependency_names or self.source_names & self.absent_source_names
                or "eval/__init__.py" not in self.source_names | self.absent_source_names):
            raise ValueError("independently resolved source/dependency/package inventory required")


def native_owner_reader(runtime_package):
    """Use the sole owner's actual lock/load protocol; do not duplicate its state logic."""
    package = Path(runtime_package).resolve()
    namespace = "_inspection_native_owner_"+hashlib.sha256(str(package).encode()).hexdigest()[:16]
    if namespace not in sys.modules:
        module = types.ModuleType(namespace)
        module.__path__ = [str(package/"lib")]
        sys.modules[namespace] = module
    runtime = importlib.import_module(namespace+".runtime")

    def load(run):
        controller = runtime.Controller(run, package=package)
        # Initialized runs have all three. Never create a synthetic state or
        # recovery copy here. Native lock/load validates schema, Brief digest,
        # implementation pin and the durable pair under the writer's lock.
        for name in ("control.json", "checkpoint.json", "lock"):
            path = controller.home/name
            if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
                raise ValueError("initialized native owner pair/lock required")
        with controller.lock():
            return controller.load()
    return load


def context_from_environment(run, project_root, runtime_package, sdk_package):
    """Resolve this study's defined consumers independently of a G4 JSON claim.

    Scope is repository code, the provided runtime/evaluator, PyFlyt sources/
    assets and the PyBullet binary. Numerical/model-engine versions and full
    weight trees still have their existing separate qualification/service
    checks; this is not an OS or every-transitive-import attestation.
    """
    project, root, package, sdk = map(lambda p: Path(p).resolve(),
                                     (project_root, run, runtime_package, sdk_package))
    if (not root.is_relative_to(project) or not package.is_relative_to(project)
            or not sdk.is_relative_to(project)):
        raise ValueError("study/runtime roots must belong to the qualified project")
    sources = set(REQUIRED_SOURCES)
    files = [*sorted((project/"envs/inspection").glob("*.py"))]
    for consumer in {package, sdk}:
        files += [*sorted((consumer/"lib").glob("*.py")), consumer/"scripts/research.py",
                  *sorted((consumer/"references").rglob("*.md")),
                  *sorted((consumer/"vendor").rglob("scripts/*.py"))]
    experiment = root/"experiment"
    manifest_path = experiment/"evaluator-source-manifest.v2.json"
    manifest_ref = Artifact(str(manifest_path.relative_to(project)),
        "ecc2069fc8795adb9096173ab2d859b6540a94309ecf99310755cb7b3177bf80")
    manifest = manifest_ref.json(project)
    closure = experiment/"evaluator-source.v2"
    expected = set(manifest["sources"])
    if ({str(p.relative_to(closure)) for p in closure.rglob("*") if p.is_file()} != expected
            or any(p.is_symlink() for p in closure.rglob("*"))):
        raise ValueError("accepted evaluator closure file set changed")
    for name, expected_sha in manifest["sources"].items():
        Artifact(str((closure/name).relative_to(project)), expected_sha).checked_path(project)
    files += [experiment/"evaluator.v2.py", manifest_path, *[closure/name for name in sorted(expected)]]
    sources.update(str(path.relative_to(project)) for path in files)
    absent = set()
    if (project/"eval/__init__.py").exists() or (project/"eval/__init__.py").is_symlink():
        sources.add("eval/__init__.py")
    else:
        absent.add("eval/__init__.py")
    packages = list((project/".venv/lib").glob("python*/site-packages/PyFlyt"))
    if len(packages) != 1:
        raise ValueError("the qualified flight environment must resolve one PyFlyt package")
    package_root = packages[0]
    binaries = list(package_root.parent.glob("pybullet*.so"))
    if len(binaries) != 1:
        raise ValueError("the qualified flight environment must resolve one PyBullet binary")
    dependencies = {str(p) for p in package_root.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".py", ".yaml", ".urdf", ".obj", ".stl", ".dae"}}
    dependencies.add(str(binaries[0]))
    return VerificationContext(project, native_owner_reader(package), frozenset(sources),
                               frozenset(dependencies), frozenset(absent))


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
    rounds = owner.get("rounds", [])
    for index, row in enumerate(rounds):
        request, decision = row.get("request", {}), row.get("decision", {})
        finding = decision.get("finding") or {}
        if not isinstance(finding, dict):
            raise ValueError("invalid native finding shape")
        if finding.get("id") != finding_id:
            continue
        issued = request.get("request_sha256")
        subject = {k: v for k, v in request.items() if k != "request_sha256"}
        if (type(issued) is not str or issued != canonical_sha(subject)
                or decision.get("request_sha256") != issued
                or row.get("validation") != "structure_and_binding_only"
                or decision.get("status") != "active" or decision.get("progress") != "verified_progress"
                or any(request.get("evidence", {}).get(a.path) != a.sha256 for a in artifacts)):
            raise ValueError("qualification is not bound by its actual applied native round")
        matches.append((index, issued))
    if len(matches) != 1:
        raise ValueError("a unique applied qualification finding is required")
    index, issued = matches[0]
    for row in rounds[index+1:]:
        # Native finding.revises is one existing finding ID, not free text.
        # Any explicit revision supersedes the old acceptance. A new acceptance
        # must bind its own new finding; unrelated later rounds are harmless.
        finding = row.get("decision", {}).get("finding") or {}
        if not isinstance(finding, dict):
            raise ValueError("invalid native finding revision shape")
        if finding.get("revises") == finding_id:
            raise ValueError("qualification finding has been superseded")
    return issued


def read_g4_proof(run, project_root, acceptance_value, *, verification, for_dispatch=True):
    """Validate immutable subjects and their native application without writes.

    Dispatch additionally requires an active owner round. Offline inspection
    can omit that liveness check; neither mode generates or opens any dataset.
    """
    root, project = Path(run).resolve(), Path(project_root).resolve()
    if (not root.is_relative_to(project) or type(for_dispatch) is not bool
            or not isinstance(verification, VerificationContext)
            or verification.project_root.resolve() != project):
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
    owner = verification.owner_reader(root)
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
    # The accepted grader is a scientific input independent of a later G4
    # manifest. A self-consistent new closure cannot silently replace it.
    Artifact.parse(design["evaluator"]).checked_path(root)
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
            or set(closure["sources"]) != verification.source_names
            or type(closure.get("dependencies")) is not dict
            or set(closure["dependencies"]) != verification.dependency_names
            or type(closure.get("absent_sources")) is not list
            or set(closure["absent_sources"]) != verification.absent_source_names
            or len(closure["absent_sources"]) != len(verification.absent_source_names)):
        raise ValueError("complete qualified source/dependency closure required")
    for name in verification.absent_source_names:
        if (project/name).exists() or (project/name).is_symlink():
            raise ValueError("qualified package absence changed")
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
