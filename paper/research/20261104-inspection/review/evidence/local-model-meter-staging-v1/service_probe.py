"""Isolated FakeMLX service CLI/staging regression, zero real generation."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
sys.path.insert(0,str(REPO))
import eval.inspection_model_service as service
from eval.inspection_model_receipts import validate_model_operation
BASE=Path('/tmp/inspection-model-meter-staging-review-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,d):p.write_text(json.dumps(d))

def run(mode):
    root=Path(tempfile.mkdtemp(prefix=mode+'-',dir=BASE));run=root/'owner';(run/'.paper').mkdir(parents=True)
    src=root/'repo/eval/inspection_model_service.py';src.parent.mkdir(parents=True);src.write_bytes(Path(service.__file__).read_bytes())
    model=root/'fake-blob';model.mkdir();(model/'fixture.bin').write_bytes(b'not a model')
    receipt=run/'receipt.json';write(receipt,{'destination':str(model),'files':[{'path':'fixture.bin','bytes':11,'sha256':sha(model/'fixture.bin')}]})
    limits={'max_calls':32,'max_total_tokens':180000,'max_input_tokens':8192,'max_output_tokens':1024,
        'max_request_seconds':60,'temperature':.2,'top_p':.95,'enable_thinking':False}
    profile={'generation_profile':limits,'file_count':1,'tokenizer_config':{},'download_receipt_sha256':sha(receipt)}
    reg=run/'registry.json';write(reg,{'models':{'synthetic':profile}})
    plan={'stage':'plan','local_model_registry':'registry.json','local_model_registry_sha256':sha(reg)}
    held={'calls':32,'tokens':180000}
    purpose={'kind':'inspection_local_generation','phase':'development','model_registry_sha256':sha(reg),'model_receipt_sha256':sha(receipt)}
    state={'frozen_plan':canon(plan),'receipts':[{'stage':'plan','packet':plan}],
        'active_round':{'id':'fake-round','stage':'fake-stage'},'task_stop':None,
        'usage_ledger':{'synthetic':{'status':'reserved','reserved':held,'purpose':json.dumps(purpose)}}}
    expected={'reservation':'synthetic','owner_frozen_plan':state['frozen_plan'],'model_receipt_sha256':sha(receipt),
        'reserved':held,'service_sha256':sha(src),'generation_profile':limits}
    statepath=run/'.paper/control.json'
    if mode=='no_plan':state['frozen_plan']=None
    write(statepath,state)
    counts={'fake_load_calls':0,'fake_stream_calls':0,'fake_chunks':0,'real_generation_calls':0}
    mx=ModuleType('mlx.core');mx.random=SimpleNamespace(seed=lambda v:None)
    mlx=ModuleType('mlx');mlx.__path__=[];mlx.core=mx
    lm=ModuleType('mlx_lm');samplers=ModuleType('mlx_lm.sample_utils');samplers.make_sampler=lambda **k:None
    class Tokenizer:
        def apply_chat_template(self,*a,**k):
            assert k['enable_thinking'] is False;return 'synthetic prompt'
        def encode(self,p):return [10,11]
    def load(*a,**k):
        counts['fake_load_calls']+=1
        if mode=='revoke_after_load':state['usage_ledger']['synthetic']['status']='known';write(statepath,state)
        return object(),Tokenizer()
    def stream(*a,**k):
        counts['fake_stream_calls']+=1;counts['fake_chunks']+=1
        yield SimpleNamespace(token=7,text='{}',prompt_tokens=2,generation_tokens=1)
    lm.load=load;lm.stream_generate=stream
    original_verify=service.verify_model_files
    def verify(*a):
        original_verify(*a)
        if mode=='revoke_after_verify':state['task_stop']={'reason':'synthetic revocation'};write(statepath,state)
    argv=['service','--run',str(run),'--reservation','synthetic','--model-receipt',str(receipt),
        '--max-calls','32','--max-total-tokens','180000']
    modules={'mlx':mlx,'mlx.core':mx,'mlx_lm':lm,'mlx_lm.sample_utils':samplers}
    request=json.dumps({'messages':[{'role':'user','content':'synthetic public prompt'}],'seed':1})+'\n'
    output=io.StringIO();error=None;second_error=None;unchanged=None;recount=None
    with patch.dict(sys.modules,modules),patch.object(service,'__file__',str(src)),\
        patch.object(service,'verify_model_files',side_effect=verify),patch.object(sys,'argv',argv),\
        patch.object(sys,'stdin',io.StringIO(request)),patch.object(sys,'stdout',output):
        operation=service.operation_directory('synthetic')
        try:service.main()
        except BaseException as e:error=type(e).__name__+': '+str(e)
        if mode=='normal':
            recount=validate_model_operation(operation,expected)
            before={str(p.relative_to(operation)):sha(p) for p in operation.rglob('*') if p.is_file()}
            with patch.object(sys,'stdin',io.StringIO(request)):
                try:service.main()
                except BaseException as e:second_error=type(e).__name__+': '+str(e)
            after={str(p.relative_to(operation)):sha(p) for p in operation.rglob('*') if p.is_file()}
            unchanged=before==after
    intent=json.loads((operation/'service-intent.json').read_text()) if (operation/'service-intent.json').exists() else None
    return {'mode':mode,'role':'all modules are explicit FakeMLX; 0 real generation/physics','path':str(root),
        **counts,'error':error,'second_invocation_error':second_error,'existing_operation_files_unchanged':unchanged,
        'operation':str(operation),'operation_exists':operation.exists(),'completion_exists':(operation/'service-completion.json').exists(),
        'usage':recount['usage'] if recount else None,'intent_pid':intent.get('pid') if intent else None,
        'intent_process_group':intent.get('process_group') if intent else None,'stdout':output.getvalue()}

if __name__=='__main__':
    invalid=[]
    for rid in ('','../x','a/b','a.b',True):
        try:service.operation_directory(rid);invalid.append({'input':rid,'accepted':True})
        except ValueError:invalid.append({'input':rid,'accepted':False})
    result={'actual_path_read_only':str(service.operation_directory('synthetic-read-only-probe')),
        'invalid_reservation_ids':invalid,'cases':[run(x) for x in ('normal','no_plan','revoke_after_verify','revoke_after_load')]}
    (BASE/'service-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
