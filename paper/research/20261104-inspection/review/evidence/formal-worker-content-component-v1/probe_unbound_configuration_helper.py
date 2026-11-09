"""Serialization/synthetic-owner negative only; no formal RNG or constructor."""
from pathlib import Path
import hashlib,importlib.util,json,sys,tempfile
W=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/planning/formal-admission-candidate.v3')
F=W/'worker-content-component-freeze.v1'
B=json.loads((F/'binding.json').read_text())
repo=W.parents[4]
for name,expected in B['working_sources'].items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==expected, 'use the matching frozen v1 in isolation'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
h=load('independent_worker_content_helpers',W/'tests/test_formal_case.py')
project=Path(tempfile.mkdtemp(prefix='unbound-config-',dir='/tmp/inspection-formal-worker-review-v1')).resolve()
f,refs,row,private,case=h.selected_fixture(project)
actual_configuration=row['configuration_sha256'];intent,manifest,review=refs
value=f.read(manifest);target=next(x for x in value['records'] if x['case_id']==row['case_id']);target['configuration_sha256']='0'*64
manifest,review=f.bind_dataset(value)
helper_name='eval/prepare_inspection_development_population.py';helper_path=f.project/helper_name
assert helper_name not in h.helper.helper.admission.REQUIRED_SOURCES
assert helper_name not in f.verification.source_names
helper_path.write_bytes((repo/helper_name).read_bytes());before=hashlib.sha256(helper_path.read_bytes()).hexdigest()
def content_reader():
    load('eval.prepare_inspection_development_population',helper_path)
    sys.modules['eval.inspection_population']=h.shared
    load('eval.inspection_development_catalog',W/'source/eval/inspection_development_catalog.py')
    sys.modules['eval.inspection_dataset_admission']=h.helper.module
    return load('unbound_config_formal_case',W/'source/eval/inspection_formal_case.py')
def validate(module):
    return module.load_formal_case(f.run,f.project,f.acceptance,intent,manifest,review,row['case_id'],private,
                                  row['case']['sha256'],verification=f.verification)
baseline=content_reader()
try:validate(baseline);raise AssertionError('bad configuration descriptor must initially reject')
except ValueError as e:
    baseline_reason=str(e)
    assert 'semantic/configuration/noise identity' in baseline_reason
# Simulate a fresh child's natural import of a modified, unqualified repository dependency.
helper_path.write_text("def configuration_digest(case):\n    return '0'*64\n")
after=hashlib.sha256(helper_path.read_bytes()).hexdigest();changed=content_reader();result=validate(changed)
assert result.binding()['configuration_sha256']=='0'*64 and actual_configuration!='0'*64
out={'role':'synthetic serialization/copy and native-shaped owner fixture, not actual formal dataset',
     'helper':helper_name,'helper_in_required_sources':False,'helper_in_verification_sources':False,
     'helper_before_sha256':before,'helper_after_sha256':after,'baseline_bad_config_rejected':baseline_reason,
     'after_only_untracked_helper_changed_content_guard_accepted':True,
     'actual_configuration_sha256':actual_configuration,'accepted_wrong_configuration_sha256':result.binding()['configuration_sha256'],
     'dataset_case_semantic_and_raw_copy_unchanged':True,'real_source_owner_or_physics_modified':False}
print(json.dumps(out,indent=2));Path('/tmp/inspection-formal-worker-review-v1/configuration-helper-results.json').write_text(json.dumps(out,indent=2)+'\n')
