"""Worker-only formal input decoding after fresh dataset admission.

This component starts no simulator and grants no runner/service dispatch. The
private copy is hashed and decoded from the same buffer before any warmup;
actors and model services must not call this content reader.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np

from eval.inspection_dataset_admission import DatasetCase, DatasetProof, read_dataset_proof
from eval.inspection_development_catalog import physical_configuration_digest
from eval.inspection_formal_admission import Artifact, bounded_bytes, context_from_environment, parse_json_bytes
from eval.inspection_population import PopulationCase, decode_formal_population_case


@dataclass(frozen=True)
class FormalCaseInput:
    case: PopulationCase
    dataset: DatasetProof
    record: DatasetCase
    private_copy_sha256: str

    def binding(self):
        ref = lambda value: {'path': value.path, 'sha256': value.sha256}
        return {'phase': self.dataset.namespace.phase, 'case_id': self.record.case_id,
            'dataset_manifest': ref(self.dataset.manifest), 'dataset_review': ref(self.dataset.review),
            'namespace_intent': ref(self.dataset.namespace.intent),
            'g4_acceptance': ref(self.dataset.namespace.g4_acceptance),
            'native_dataset_request_sha256': self.dataset.native_request_sha256,
            'native_namespace_request_sha256': self.dataset.namespace.native_request_sha256,
            'private_copy_sha256': self.private_copy_sha256,
            'case_sha256': self.record.case_sha256,
            'configuration_sha256': self.record.configuration_sha256}


def load_formal_worker_input(project_root, payload, private_copy_path, expected_copy_sha256):
    """Rebuild trusted-runner context inside a spawned worker before warmup.

    Payload paths originate in the privileged runner's admitted dispatch,
    never in a private case or actor/tool message. Verification is constructed
    in this process; callable verification objects are not pickled across it.
    """
    fields = {'schema', 'run', 'runtime', 'sdk', 'g4', 'namespace', 'dataset',
              'dataset_review', 'case_id', 'opening'}
    if (type(payload) is not dict or set(payload) != fields
            or payload['schema'] != 'inspection-formal-worker-input-v1'
            or any(type(payload[key]) is not str or not payload[key] for key in ('run','runtime','sdk','case_id'))):
        raise ValueError('complete trusted-runner formal worker input required')
    for key in ('g4', 'namespace', 'dataset', 'dataset_review'):
        Artifact.parse(payload[key])
    if payload['opening'] is not None: Artifact.parse(payload['opening'])
    verification = context_from_environment(payload['run'], project_root, payload['runtime'], payload['sdk'])
    return load_formal_case(payload['run'], project_root, payload['g4'], payload['namespace'],
        payload['dataset'], payload['dataset_review'], payload['case_id'], private_copy_path,
        expected_copy_sha256, verification=verification, holdout_opening_value=payload['opening'])


def load_formal_case(run, project_root, acceptance_value, intent_value, manifest_value,
                     review_value, case_id, private_copy_path, expected_copy_sha256,
                     *, verification, holdout_opening_value=None):
    if np.__version__ != '1.26.4':
        raise ValueError('worker content requires the qualified NumPy1.26.4 environment')
    proof = read_dataset_proof(run, project_root, acceptance_value, intent_value,
        manifest_value, review_value, verification=verification,
        holdout_opening_value=holdout_opening_value)
    rows = [row for row in proof.records if row.case_id == case_id]
    if type(case_id) is not str or len(rows) != 1:
        raise ValueError('worker input requires one exact admitted dataset root')
    row = rows[0]
    project = Path(project_root).resolve()
    path = Path(private_copy_path)
    if not path.is_absolute(): path = project/path
    if '..' in path.parts or not path.is_relative_to(project):
        raise ValueError('private worker copy must stay inside the qualified project')
    for part in (path, *path.parents):
        if part.is_symlink(): raise ValueError('private worker copy cannot use symbolic aliases')
        if part == project: break
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError('one regular private copy without hard links required')
    if expected_copy_sha256 != row.case.sha256:
        raise ValueError('worker private-copy expectation differs from the admitted input')
    raw = bounded_bytes(path, 16 * 1024 * 1024)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_copy_sha256:
        raise ValueError('private worker copy changed before decoding')
    case = decode_formal_population_case(parse_json_bytes(raw), proof.namespace.phase, row.case_id)
    if (case.domain != row.domain or case.level != 'L'+str(row.level)
            or case.seed != row.sensor_seed or case.digest() != row.case_sha256
            or physical_configuration_digest(case) != row.configuration_sha256):
        raise ValueError('private content differs from its admitted semantic/configuration/noise identity')
    fresh = read_dataset_proof(run, project_root, acceptance_value, intent_value,
        manifest_value, review_value, verification=verification,
        holdout_opening_value=holdout_opening_value)
    if fresh != proof:
        raise ValueError('formal dataset changed during worker content validation')
    return FormalCaseInput(case, proof, row, digest)
