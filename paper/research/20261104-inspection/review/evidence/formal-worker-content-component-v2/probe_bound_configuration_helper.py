"""Bounded synthetic source-closure replay; no formal RNG or physical constructors."""
from pathlib import Path
from dataclasses import replace
import hashlib,importlib.util,json,sys,tempfile
W=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/planning/formal-admission-candidate.v3')
F=W/'worker-content-component-freeze.v2';repo=W.parents[4];R=W.parents[1]
B=json.loads((F/'binding.json').read_text())
for name,expected in B['working_sources'].items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==expected, 'use the matching frozen v2 in isolation'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
h=load('independent_worker_v2_helpers',W/'tests/test_formal_case.py')
helper_name='eval/prepare_inspection_development_population.py';good=(repo/helper_name).read_bytes();results=[]
for wrong in [False,True]:
    project=Path(tempfile.mkdtemp(prefix='bound-helper-',dir='/tmp/inspection-formal-worker-review-v2')).resolve()
    f=h.helper.DatasetFixture(project);hp=f.project/helper_name;hp.write_bytes(good)
    # Qualify the actual good helper bytes before namespace/dataset fixture creation.
    closure=f.read(f.source);closure['sources'][helper_name]=h.helper.helper.admission.file_sha(hp)
    f.source=f.put(f.source['path'],closure)
    subject=f.read(f.subject);subject['source_manifest']=f.source;f.subject=f.put(f.subject['path'],subject)
    review=f.read(f.review);review['subject_sha256']=f.subject['sha256'];f.review=f.put(f.review['path'],review)
    acceptance=f.read(f.acceptance);acceptance.update(subject=f.subject,review=f.review);f.acceptance=f.put(f.acceptance['path'],acceptance)
    f.owner['rounds']=[];f.bind(acceptance['native_finding_id'],[f.acceptance,f.subject,f.review])
    omitted=False
    try:replace(f.verification,source_names=frozenset(f.verification.source_names-{helper_name}))
    except ValueError:omitted=True
    assert omitted
    intent,manifest,review=f.dataset();value=f.read(manifest)
    row=next(x for x in value['records'] if x['case_id']=='main-ID-L2-00')
    body=json.loads((h.RUN/'qualification/dev-population.v1/devext-ID-L2-00.json').read_text())
    body.update(split='main',case_id=row['case_id'],lineage_group=row['case_id'],seed=row['sensor_seed'])
    case=h.shared.decode_formal_population_case(body,'main',row['case_id'])
    row['case']=f.put(row['case']['path'],case.as_dict());row['case_sha256']=case.digest()
    row['configuration_sha256']='0'*64 if wrong else h.catalog.physical_configuration_digest(case)
    manifest,review=f.bind_dataset(value)
    private=f.project/'flight/private-case.json';private.parent.mkdir();private.write_bytes((f.run/row['case']['path']).read_bytes())
    def reader():
        load('eval.prepare_inspection_development_population',hp)
        sys.modules['eval.inspection_population']=h.shared
        load('eval.inspection_development_catalog',W/'source/eval/inspection_development_catalog.py')
        sys.modules['eval.inspection_dataset_admission']=h.helper.module
        return load('bound_helper_formal_reader',W/'source/eval/inspection_formal_case.py')
    def attempt(module):
        try:
            value=module.load_formal_case(f.run,f.project,f.acceptance,intent,manifest,review,row['case_id'],private,row['case']['sha256'],verification=f.verification)
            return {'accepted':True,'configuration_sha256':value.record.configuration_sha256}
        except ValueError as error:return {'accepted':False,'reason':str(error)}
    before=attempt(reader());assert before['accepted'] is (not wrong)
    if wrong:assert 'configuration/noise identity' in before['reason']
    hp.write_text("def configuration_digest(case):\n    return '0'*64\n")
    changed=reader();used=[];original_decode=changed.decode_formal_population_case
    def spy(*args,**kwargs):used.append('decode');return original_decode(*args,**kwargs)
    changed.decode_formal_population_case=spy
    drift=attempt(changed);assert not drift['accepted'] and not used and 'artifact changed' in drift['reason']
    hp.write_bytes(good);restored=attempt(reader());assert restored['accepted'] is (not wrong)
    if wrong:assert 'configuration/noise identity' in restored['reason']
    results.append({'wrong_configuration':wrong,'omitted_helper_context_constructor_rejected':omitted,
        'bound_helper_sha256':hashlib.sha256(good).hexdigest(),'before':before,'after_actual_helper_drift':drift,
        'decode_calls_during_drift':len(used),'after_restoring_exact_bytes':restored})
print(json.dumps(results,indent=2));Path('/tmp/inspection-formal-worker-review-v2/bound-helper-results.json').write_text(json.dumps(results,indent=2)+'\n')
