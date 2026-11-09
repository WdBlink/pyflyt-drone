"""Fix audit's episode-plan canonical convention; no repeat score/model work."""
from pathlib import Path
import hashlib
import json
BASE=Path('/tmp/inspection-event-scope-pair-review-v1')
p=BASE/'pair-audit.json';d=json.loads(p.read_text());(BASE/'pair-audit-before-hash-spec-correction.json').write_bytes(p.read_bytes())
pair=json.loads((BASE/'source/pair.json').read_text());rows=[]
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
for a in d['attempts']:
    attempt=Path(a['attempt']);plan=json.loads((attempt.parent/'plan.json').read_text());term=json.loads((attempt/'terminal.json').read_text());h=canonical(plan)
    a['plan_canonical_sha256']=h;a['checks']['plan_canonical_binding']=h==term['plan_sha256'];a['passed']=all(a['checks'].values())
    rows.append({'label':a['label'],'canonical_plan_sha256':h,'terminal_equal':a['checks']['plan_canonical_binding']})
d['root_checks']['paired_semantic_plan_receipts']=d['attempts'][0]['plan_canonical_sha256']==pair['event_episode']['plan_sha256'] and d['attempts'][-1]['plan_canonical_sha256']==pair['clean_episode']['plan_sha256']
d['passed']=all(d['root_checks'].values()) and all(a['passed'] for a in d['attempts']) and all(d['archive_checks'].values()) and all(d['tokenizer_only_checks'].values())
d['audit_hash_spec_correction']='episode canonical_digest uses sorted JSON with default separators; owner compact canonical is a different convention. Only plan hashes corrected; three frozen score outputs retained.'
p.write_text(json.dumps(d,indent=2)+'\n')
out={'role':'audit fixture convention correction, not a source/receipt repair or new score run','rows':rows,'pair_plan_binding':d['root_checks']['paired_semantic_plan_receipts'],'passed':d['passed']}
(BASE/'hash-spec-correction.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
