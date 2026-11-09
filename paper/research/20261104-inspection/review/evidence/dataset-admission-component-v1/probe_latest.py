from pathlib import Path
from dataclasses import replace
import importlib.util,json,sys,tempfile
W=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/planning/formal-admission-candidate.v3')
# Refuse to reinterpret these historical probes against a later working candidate.
R=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection')
F=R/'planning/formal-admission-candidate.v3/dataset-component-freeze.v1'
B=json.loads((F/'binding.json').read_text())
for name, expected in B['working_sources'].items():
    import hashlib
    assert hashlib.sha256((R.parents[2]/name).read_bytes()).hexdigest()==expected, 'materialize matching frozen v1 in an isolated layout before reproducing'
spec=importlib.util.spec_from_file_location('independent_dataset_helpers',W/'tests/test_dataset_admission.py');t=importlib.util.module_from_spec(spec);sys.modules[spec.name]=t;spec.loader.exec_module(t)
outputs=[]
for fault in ['manifest','review','report','case','audit','extra_file']:
    p=Path(tempfile.mkdtemp(prefix='dataset-latest-',dir='/tmp/inspection-dataset-review-v1'));f=t.DatasetFixture(p);intent,manifest,review=f.dataset()
    value=f.read(manifest);row=value['records'][0]
    refs={'manifest':manifest,'review':review,'report':f.read(review)['report'],'case':row['case'],'audit':row['generation_audit']}
    reader=f.verification.owner_reader;calls=[0]
    def latest(run):
        calls[0]+=1
        if calls[0]==4:
            path=f.run/(refs[fault]['path'] if fault!='extra_file' else 'datasets/main.v1/unbound-late-file.json')
            if fault=='extra_file':path.write_text('{}')
            else:
                with path.open('a') as stream:stream.write(' ')
        return reader(run)
    try:
        proof=t.module.read_dataset_proof(f.run,f.project,f.acceptance,intent,manifest,review,
            verification=replace(f.verification,owner_reader=latest))
        result={'accepted':True,'records':len(proof.records)}
    except Exception as error:result={'accepted':False,'reason':str(error)}
    result.update(fault=fault,raw_owner_reads=calls[0])
    if fault!='extra_file':
        try:t.helper.admission.Artifact.parse(refs[fault]).checked_path(f.run);result['current_artifact_matches']=True
        except ValueError:result['current_artifact_matches']=False
    outputs.append(result)
print(json.dumps(outputs,indent=2))
Path('/tmp/inspection-dataset-review-v1/latest-results.json').write_text(json.dumps(outputs,indent=2)+'\n')
