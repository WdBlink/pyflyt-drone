"""Explicit mock exhaustion, no scientific root generated or physics/model."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO))
import eval.prepare_inspection_development_population as prepare
from eval.inspection_population import PopulationGenerationExhausted
BASE=Path('/tmp/inspection-development-population-review-v1')
root=Path(tempfile.mkdtemp(prefix='controlled-exhaustion-',dir=BASE));prior=root/'prior';prior.mkdir()
old=REPO/'paper/research/20261104-inspection/qualification/dev-cases.v2';record=json.loads((old/'manifest.json').read_text())['records'][0]
shutil.copy2(old/record['file'],prior/record['file']);(prior/'manifest.json').write_text(json.dumps({'split':'dev','records':[record]}))
out=root/'new';order=[];real_write=prepare.durable_once
def durable(path,value):
    real_write(path,value);order.append(Path(path).name)
def exhausted(case_id,domain,level,seed):
    assert order==['namespace-intent.json']
    intent=json.loads((out/'namespace-intent.json').read_text());assert seed==intent['root_generation_seeds'][0]
    raise PopulationGenerationExhausted([{'candidate_index':i,'status':'rejected','reason':'controlled fixture','placements':[]} for i in range(1000)])
raised=None
with patch.object(prepare.secrets,'randbits',return_value=12345) as entropy,patch.object(prepare,'generate_development_population_case',side_effect=exhausted),patch.object(prepare,'durable_once',side_effect=durable):
    try:prepare.prepare(out,[prior])
    except PopulationGenerationExhausted:raised='PopulationGenerationExhausted'
    first=entropy.call_count
    try:prepare.prepare(out,[prior]);retry_rejected=False
    except ValueError:retry_rejected=True
    count=entropy.call_count
failure=json.loads((out/'failure.json').read_text());audit=json.loads((out/'devext-ID-L1-00-exhaustion.json').read_text())
result={'role':'controlled mocked exhaustion only, not a real new population','raised':raised,'order':order,
    'one_entropy_call_only':first==count==1,'retry_same_output_rejected':retry_rejected,
    'all1000_failure_records_preserved':len(audit['attempts'])==1000,'failure':failure,'no_final_manifest':not(out/'manifest.json').exists(),
    'passed':raised=='PopulationGenerationExhausted' and first==count==1 and retry_rejected and len(audit['attempts'])==1000
        and failure['status']=='namespace_blocked_no_automatic_reseed' and failure['completed_cases']==0 and not(out/'manifest.json').exists()}
(BASE/'failure-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
