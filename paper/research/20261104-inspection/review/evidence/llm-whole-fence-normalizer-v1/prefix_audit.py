"""Only closed response texts: framing sensitivity, no rollout or scoring."""
import hashlib
import importlib.util
import json
from pathlib import Path

REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
QUAL=REPO/'paper/research/20261104-inspection/qualification'
BASE=Path('/tmp/inspection-llm-whole-fence-review-v1')
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
strict=load('strict',BASE/'initial-source/inspection_llm_protocol.py')
operation=REPO/'logs/inspection-model-resource-operations.v1/dev-model-realtime-M1-B3-L3-00-v1-20261106'
versions=[]
for v in (1,2):
    candidate_path=QUAL/f'llm-wire-normalizer-candidate.v{v}.py'
    artifact_path=QUAL/f'llm-wire-normalization-prefix.v{v}.json'
    candidate=load('candidate'+str(v),candidate_path);artifact=json.loads(artifact_path.read_text());rows=[]
    for expected in artifact['rows']:
        p=operation/f"call-{expected['submitted_ordinal']:04d}/response.json";response=json.loads(p.read_text());raw=response['text']
        body,changed=candidate.unwrap_json_fence(raw)
        def verdict(text):
            try:strict.parse_reply(text);return 'accepted'
            except strict.ModelReplyError as e:return str(e)
        row={'submitted_ordinal':expected['submitted_ordinal'],'raw_text_sha256':hashlib.sha256(raw.encode()).hexdigest(),
            'whole_wrapper':changed,'old_parser':verdict(raw),'candidate_parser':verdict(body)}
        rows.append({'expected_row_exact':row==expected,'response_status':response['status'],'raw_length':len(raw),'row':row,'source_response_sha256':sha(p)})
    versions.append({'version':v,'candidate_sha256':sha(candidate_path),'artifact_sha256':sha(artifact_path),
        'candidate_binding_matches':artifact['candidate_sha256']==sha(candidate_path),
        'active_protocol_binding_matches':artifact['active_protocol_sha256']==sha(REPO/'eval/inspection_llm_protocol.py'),
        'all_rows_match_closed_originals':all(x['expected_row_exact'] and x['response_status']=='completed' for x in rows),
        'count':len(rows),'fences':sum(x['row']['whole_wrapper'] for x in rows),
        'old_accepted':sum(x['row']['old_parser']=='accepted' for x in rows),
        'candidate_accepted':sum(x['row']['candidate_parser']=='accepted' for x in rows),
        'max_raw_length':max(x['raw_length'] for x in rows),'rows':rows})
data={'role':'offline framing verification on existing completed texts; not new rollout, model call, tool ACK or Y','versions':versions}
(BASE/'prefix-audit.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:v for k,v in data.items() if k!='versions'}))
for d in versions:print(json.dumps({k:v for k,v in d.items() if k!='rows'}))
