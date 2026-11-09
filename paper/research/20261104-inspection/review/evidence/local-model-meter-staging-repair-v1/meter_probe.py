"""Independent synthetic accepted-invalid meter boundaries, no model/physics."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile

REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('meter_tests',REPO/'tests/test_inspection_model_receipts.py')
test=importlib.util.module_from_spec(spec);spec.loader.exec_module(test)
from eval.inspection_model_receipts import validate_model_operation
BASE=Path('/tmp/inspection-model-meter-staging-repair-review-v1')
rows=[]
for mode in ('normal','bool_completion','float_completion','duplicate_token_field','impossible_headroom','unknown_receipt_dir'):
    root=Path(tempfile.mkdtemp(prefix=mode+'-',dir=BASE));expected=test.fixture(root)
    if mode in ('bool_completion','float_completion'):
        p=root/'service-completion.json';d=json.loads(p.read_text())
        d['usage']['calls']=True if mode=='bool_completion' else 1.0;p.write_text(json.dumps(d))
    if mode=='duplicate_token_field':
        p=root/'call-0001/tokens.jsonl';p.write_text(p.read_text().replace('"token": 7','"token": 999, "token": 7'))
    if mode=='impossible_headroom':
        expected['generation_profile']['max_total_tokens']=4;expected['reserved']['tokens']=4
        p=root/'service-intent.json';d=json.loads(p.read_text());d['reserved']=expected['reserved'];d['request_budget']=expected['generation_profile'];p.write_text(json.dumps(d))
    if mode=='unknown_receipt_dir':
        p=root/'unexpected-call';p.mkdir();(p/'request.json').write_text('{"input_tokens_submitted":123}')
    try:
        result=validate_model_operation(root,expected);row={'mode':mode,'accepted':True,'usage':result['usage']}
    except BaseException as e:row={'mode':mode,'accepted':False,'error':type(e).__name__+':'+str(e)}
    row['path']=str(root);rows.append(row)
(BASE/'meter-probe-repeat.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
