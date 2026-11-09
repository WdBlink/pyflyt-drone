"""Whole-fence framing only, no model/flight/score. Reuse the actual strict checker."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO))
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
BASE=Path('/tmp/inspection-llm-whole-fence-review-v1')
candidate=load('candidate',BASE/'initial-source/llm-wire-normalizer-candidate.v1.py')
strict=load('frozen_strict',BASE/'initial-source/inspection_llm_protocol.py')
valid='{"commands":[],"note":"ready"}'
def fence(s,head='```json',eol='\n'):return head+eol+s+eol+'```'
examples={
 'plain':valid,'LF':fence(valid),'bare':fence(valid,'```'),'CRLF':fence(valid,eol='\r\n'),
 'whitespace_outer':' \t\r\n'+fence(valid)+'\r\n\t ',
 'U2028_in_note':fence('{"commands":[],"note":"a\u2028b"}'),
 'backticks_in_note':fence('{"commands":[],"note":"```json"}'),
 'prose_before':'Here is JSON:\n'+fence(valid),'prose_after':fence(valid)+'\nDone.',
 'multi_block':fence(valid)+'\n'+fence(valid),'nested':fence(fence(valid)),
 'wrong_language':fence(valid,'```python'),'opening_inline':'```json '+valid+'\n```',
 'duplicate_key':fence('{"commands":[],"commands":[]}'),'NaN':fence('{"commands":[],"note":NaN}'),
 'unknown_tool':fence('{"commands":[{"name":"admin","args":{}}]}'),
 'multi_object':fence(valid+'\n'+valid),'nine_commands':fence(json.dumps({'commands':[{'name':'enter_safe_exit','args':{'drone_id':0,'expected_version':0}}]*9})),
 'wrong_fields':fence('{"commands":[{"name":"cancel_job","args":{"job_id":"j"}}]}'),
 'bad_json':fence('{"commands":[],}'),'oversize_note':fence(json.dumps({'commands':[],'note':'x'*513})),
 'non_string':None,
 'raw_overlimit_fence':fence(valid+' '*(24000-len(valid))),
 'raw_overlimit_outer_spaces':' '*24000+fence(valid),
}
rows=[]
for name,text in examples.items():
    normalized,changed=candidate.unwrap_json_fence(text)
    row={'name':name,'whole_wrapper':changed,'raw_length':len(text) if isinstance(text,str) else None,
        'normalized_length':len(normalized) if isinstance(normalized,str) else None,'accepted':False,'raw_accepted':False}
    try:strict.parse_reply(text);row['raw_accepted']=True
    except strict.ModelReplyError:pass
    try:
        reply,cmd=strict.parse_reply(normalized);row.update(accepted=True,reply=reply)
    except strict.ModelReplyError as e:row['error']=str(e)
    rows.append(row)
u=next(x for x in rows if x['name']=='U2028_in_note')
result={'role':'pure CPU offline wrapper/checker composition, not a rollout/Y','candidate_sha256':hashlib.sha256((BASE/'initial-source/llm-wire-normalizer-candidate.v1.py').read_bytes()).hexdigest(),
    'strict_sha256':hashlib.sha256((BASE/'initial-source/inspection_llm_protocol.py').read_bytes()).hexdigest(),
    'unicode_string_preserved':u['reply']['note']=='a\u2028b','rows':rows}
(BASE/'probe.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
