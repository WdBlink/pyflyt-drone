"""Actual owner read-only negative gate. No weights or MLX import reachable."""
from pathlib import Path
import json,hashlib,sys
import eval.inspection_model_service as s
R=Path('paper/research/20261104-inspection');p=R/'.paper/control.json';before=p.read_bytes();state=json.loads(before);reservation='review-negative-no-reservation';assert reservation not in state.get('usage_ledger',{})
old=sys.argv;result={'real_generation':0,'real_gpu':0,'frozen_plan_is_null':state.get('frozen_plan') is None,'reservation_missing':True}
try:
 sys.argv=['service','--run',str(R),'--reservation',reservation,'--model-receipt','/tmp/local-model-wire-review/not-a-real-model-receipt.json','--max-calls','1','--max-total-tokens','1']
 try:s.main();raise AssertionError('actual negative gate unexpectedly admitted')
 except ValueError as e:assert 'generation requires' in str(e);result['refused_before_model_receipt_or_MLX']=True;result['reason']=str(e)
finally:sys.argv=old
result['control_unchanged']=before==p.read_bytes();result['operation_directory_absent']=not (R/'resource-operations'/reservation).exists();assert result['control_unchanged'] and result['operation_directory_absent'];Path('/tmp/local-model-wire-review/negative.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
