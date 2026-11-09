"""Secondary packet-to-worker and full SDK adapters; all effects are Fake."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent/'eval'
spec = importlib.util.spec_from_file_location('secondary_runner_helpers', HERE/'test_formal_secondary.py')
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
dispatch = helper.helper.module
primary = helper.load('eval.run_inspection_formal', STAGE/'run_inspection_formal.py')
driver = helper.load('secondary_full_driver_candidate', STAGE/'run_inspection_formal_secondary.py')
worker = helper.load('secondary_worker_content_candidate', STAGE/'inspection_formal_case.py')
SDK = HERE.parent/'paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1'


@pytest.fixture(autouse=True)
def matching_types(monkeypatch):
    monkeypatch.setitem(sys.modules, 'eval.inspection_population', helper.shared)
    monkeypatch.setitem(sys.modules, 'eval.inspection_formal_intervention', helper.intervention)
    monkeypatch.setitem(sys.modules, 'eval.inspection_formal_secondary', helper.module)
    monkeypatch.setitem(sys.modules, 'eval.inspection_formal_dispatch', dispatch)


def prepared(tmp_path, monkeypatch, track):
    f = helper.helper.helper.ExecutionFixture(tmp_path)
    original_dataset = f.dataset
    def with_literal_content(phase='main', opening=None):
        refs = original_dataset(phase, opening)
        value = f.read(refs[1])
        row = next(row for row in value['records'] if row['case_id'] == 'main-ID-L3-00')
        # Rename one existing development body solely for this software reader.
        body = json.loads((HERE.parent/'paper/research/20261104-inspection'/'qualification/dev-population.v1/devext-ID-L3-00.json').read_text())
        body.update(split='main', case_id=row['case_id'], lineage_group=row['case_id'], seed=row['sensor_seed'])
        case = helper.shared.decode_formal_population_case(body, phase, row['case_id'])
        from eval.inspection_development_catalog import physical_configuration_digest
        row['case'] = f.put(row['case']['path'], case.as_dict())
        row['case_sha256'] = case.digest()
        row['configuration_sha256'] = physical_configuration_digest(case)
        manifest, review = f.bind_dataset(value)
        return refs[0], manifest, review
    f.dataset = with_literal_content
    refs, execution = f.execution()
    ref = helper.module.prepare_secondary_execution(f.run, f.project, f.acceptance, *refs,
        execution, track, verification=f.verification)
    value = f.read(ref)
    f.bind(helper.module.FINDINGS[track], [ref, value['episode_order']])
    proof = helper.module.read_secondary_execution(f.run, f.project, f.acceptance, *refs,
        execution, ref, verification=f.verification)
    monkeypatch.setattr(dispatch, 'context_from_environment', lambda *args: f.verification)
    monkeypatch.setattr(worker, 'context_from_environment', lambda *args: f.verification)
    return f, refs, execution, ref, proof


@pytest.mark.parametrize('track,condition', [('clean', 'B0'), ('clean', 'B4-M1'),
                                            ('ablation', 'B4-M1'), ('latency', 'B4-M1')])
def test_secondary_packet_plan_argv_and_worker_transform_agree(tmp_path, monkeypatch, track, condition):
    f, refs, execution, ref, proof = prepared(tmp_path, monkeypatch, track)
    arm = helper.module.ARMS[track][-1]
    slot = next(row for row in proof.slots if row.condition == condition and row.arm == arm)
    packet = dispatch.write_secondary_dispatch(f.run, f.project, f.acceptance, *refs, execution,
        ref, slot.case_id, condition, slot.repeat_ordinal, track, arm,
        runtime=f.project/'runtime', sdk=f.project/'sdk', verification=f.verification)
    admitted = dispatch.admit_dispatch(f.run, f.project, packet)
    plan = primary.episode_plan(admitted, f.run, f.project, sys.executable)
    command = primary.episode_command(plan, f.project/'flight', project_root=f.project, reservation='Fake')
    if track == 'ablation':
        assert command[command.index('--ablation')+1] == arm
    if track == 'latency':
        assert command[command.index('--injected-delay-seconds')+1] == '20'
    private = f.project/'flight/private-case.json'
    private.parent.mkdir()
    private.write_bytes(admitted.episode.case.case.checked_path(f.run).read_bytes())
    content = worker.load_formal_worker_input(f.project, admitted.worker_input, private,
                                             admitted.episode.case.case.sha256)
    assert content.binding() == admitted.case_binding()
    assert content.case.level == 'L3'
    assert content.case.schema == (helper.intervention.CLEAN_SCHEMA if track == 'clean' else helper.shared.SCHEMA)


@pytest.mark.parametrize('track,count,means', [('clean', 216, 96), ('ablation', 108, 36), ('latency', 180, 60)])
def test_full_secondary_driver_uses_fixed_order_and_real_sdk_denominators(tmp_path, monkeypatch, track, count, means):
    f, refs, execution, ref, proof = prepared(tmp_path, monkeypatch, track)
    request = {'schema': 'inspection-secondary-run-v1', 'run': str(f.run),
        'runtime': str(f.project/'runtime'), 'sdk': str(SDK), 'g4': f.acceptance,
        'namespace': refs[0], 'dataset': refs[1], 'dataset_review': refs[2],
        'execution': execution, 'secondary_execution': ref, 'opening': None}
    seen, pending, values = [], {}, {}
    monkeypatch.setattr(driver, 'admitted_request', lambda *args: (f.verification, proof.primary))
    monkeypatch.setattr(driver, 'read_secondary_execution', lambda *args, **kwargs: proof)
    def packet(run, project, g4, namespace, dataset, review, execution, secondary,
               case, condition, repeat, track, arm, **kwargs):
        slot = next(row for row in proof.slots if (row.case_id, row.condition, row.repeat_ordinal, row.arm)
                    == (case, condition, repeat, arm))
        assert slot.order_index == len(seen)
        pending['slot'] = slot
        return {'path': 'literal-software-packet', 'sha256': 'a'*64}
    monkeypatch.setattr(driver, 'write_secondary_dispatch', packet)
    monkeypatch.setattr(driver, 'admit_dispatch', lambda *args: SimpleNamespace(
        episode=SimpleNamespace(slot=pending['slot'])))
    monkeypatch.setattr(driver, 'episode_plan', lambda admitted, *args: {
        'case_id': admitted.episode.slot.case_id, 'condition': admitted.episode.slot.condition,
        'repeat_ordinal': admitted.episode.slot.repeat_ordinal})
    def execute(directory, plan, command, **kwargs):
        seen.append(pending['slot'])
        values[str(directory)] = plan['repeat_ordinal']/4
    monkeypatch.setattr(driver, 'execute_local_episode', execute)
    monkeypatch.setattr(driver, 'execute_development_llm_episode', execute)
    monkeypatch.setattr(driver, 'measure_completed_episode', lambda directory, expected: (
        values[str(directory)], {'role': 'literal software Fake, no flight or model'}))
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        monkeypatch.setenv(key, '1')
    out = f.project/'logs/secondary-sdk'
    result = driver.run_secondary(request, out, project_root=f.project)
    assert seen == list(proof.slots) and len(seen) == count
    outer = json.loads((out/'root-means/results.json').read_text())
    assert outer['execution_status'] == 'completed' and len(outer['runs']) == means
    assert result['additional_actual_episode_denominator' if track != 'clean' else 'actual_episode_denominator'] == count
