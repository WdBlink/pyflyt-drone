"""New smoke main over an admitted synthetic owner and fake local client.

No MLX imports, no subprocess backend and no physical world are executed.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
sys.path.insert(0,str(REPO))
import eval.probe_inspection_model_generation as smoke
from eval.inspection_model_client import ModelOperationUnknown
from eval.inspection_model_service import admitted_limits, verify_model_files
BASE=Path('/tmp/inspection-generation-smoke-binding-review-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def run(mode):
    root=Path(tempfile.mkdtemp(prefix=mode+'-',dir=BASE));repo=root/'repo';(repo/'eval').mkdir(parents=True)
    for name in ('inspection_model_service.py','inspection_model_client.py','inspection_model_receipts.py','inspection_llm_protocol.py'):
        shutil.copy2(REPO/'eval'/name,repo/'eval'/name)
    owner=root/'owner';(owner/'planning').mkdir(parents=True);(owner/'.paper').mkdir()
    limits={'max_calls':32,'max_total_tokens':180000,'max_input_tokens':8192,'max_output_tokens':1024,
        'max_request_seconds':60,'temperature':.2,'top_p':.95,'enable_thinking':False}
    profiles={}
    for alias in ('M1','M2'):
        model=root/('fake-blob-'+alias);model.mkdir();blob=model/'blob.bin';blob.write_bytes(alias.encode())
        receipt=owner/(alias+'-receipt.json');write(receipt,{'destination':str(model),
            'files':[{'path':'blob.bin','bytes':2,'sha256':sha(blob)}]})
        profiles[alias]={'generation_profile':limits,'tokenizer_config':{},'file_count':1,
            'download_receipt':receipt.name,'download_receipt_sha256':sha(receipt)}
    registry=owner/'planning/local-model-registry.v1.json';write(registry,{'models':profiles})
    plan={'stage':'plan','local_model_registry':'planning/local-model-registry.v1.json','local_model_registry_sha256':sha(registry)}
    chosen='M2' if mode=='alias_mismatch' else 'M1'
    purpose={'kind':'inspection_local_generation','phase':'development','model_registry_sha256':sha(registry),
        'model_receipt_sha256':profiles[chosen]['download_receipt_sha256']}
    purpose.update(model_alias='M1',owner_frozen_plan=digest(plan),service_source_sha256=sha(repo/'eval/inspection_model_service.py'),
        actual_request_budget={**limits,'max_calls':1,'max_total_tokens':16384,'max_output_tokens':64})
    changes={'wrong_kind':('kind','other'),'wrong_phase':('phase','main'),'wrong_alias':('model_alias','M2'),
        'wrong_registry':('model_registry_sha256','f'*64),'wrong_model':('model_receipt_sha256','f'*64),
        'wrong_owner_plan':('owner_frozen_plan','f'*64),'wrong_source':('service_source_sha256','f'*64)}
    if mode in changes:
        k,v=changes[mode];purpose[k]=v
    if mode=='wrong_config':purpose['actual_request_budget']['max_output_tokens']=32
    if mode=='float_config':purpose['actual_request_budget']['max_calls']=1.0
    held={'calls':1,'tokens':16384}
    state={'frozen_plan':digest(plan),'receipts':[{'stage':'plan','packet':plan}],
        'active_round':{'id':'synthetic-round','stage':'synthetic-stage'},'task_stop':None,
        'usage_ledger':{'synthetic':{'status':'reserved','reserved':held,'purpose':json.dumps(purpose)}}}
    write(owner/'.paper/control.json',state)
    if mode=='alias_mismatch':(owner/'M1-receipt.json').write_bytes((owner/'M2-receipt.json').read_bytes())
    operation=root/'staging/synthetic'
    counters={'fake_init':0,'fake_submit':0,'fake_finish':0,'real_generation':0,'admitted_alias':None}
    class FakeClient:
        def __init__(self,argv,logs):
            counters['fake_init']+=1
            self.alive=True;self.process=SimpleNamespace(poll=lambda:None if self.alive else 0)
            self.ready={'kind':'ready','reservation':'synthetic'}
            logs=Path(logs);logs.mkdir();(logs/'stderr.log').write_text('');(logs/'stdout.log').write_text('fake transport\n')
            args={argv[i]:argv[i+1] for i in range(3,len(argv)-1,2)}
            receipt=Path(args['--model-receipt'])
            admitted=admitted_limits(owner,'synthetic',1,16384,receipt)
            data=json.loads(receipt.read_text());verify_model_files(data['destination'],data['files'])
            counters['admitted_alias']=next(k for k,v in profiles.items() if v['download_receipt_sha256']==sha(receipt))
            actual={**limits,'max_calls':1,'max_total_tokens':16384,'max_output_tokens':64}
            operation.mkdir(parents=True,exist_ok=False)
            write(operation/'service-intent.json',{'reservation':'synthetic','owner_frozen_plan':state['frozen_plan'],
                'reserved':held,'request_budget':actual,'model_receipt_sha256':sha(receipt),
                'source_sha256':sha(repo/'eval/inspection_model_service.py')})
        def submit(self,messages,seed):
            counters['fake_submit']+=1
            if mode=='unknown':return
            call=operation/'call-0001';call.mkdir()
            write(call/'request.json',{'messages':messages,'seed':seed,'input_tokens_submitted':3,
                'max_output_tokens':64,'temperature':.2,'top_p':.95,'enable_thinking':False})
            if mode in ('wrong_request','wrong_seed','bool_seed'):
                request=json.loads((call/'request.json').read_text())
                if mode=='wrong_request':request['messages']=[{'role':'user','content':'different public prompt'}]
                if mode=='wrong_seed':request['seed']=seed+1
                if mode=='bool_seed':request['seed']=True
                write(call/'request.json',request)
            generated=0 if mode=='zero_tokens' else 1
            text='{"commands":[],"note":"ready"}' if generated else ''
            (call/'tokens.jsonl').write_text(json.dumps({'token':7,'text':text,
                'runtime_prompt_tokens':3,'runtime_generation_tokens':1})+'\n' if generated else '')
            status='runtime_failure' if mode in ('known_failure','response_mismatch') else 'completed'
            self.response={'kind':'response','status':status,'text':text,'wall_seconds':.01,
                'deadline_exceeded':False,'input_tokens_submitted':3,'generated_tokens_observed':generated,
                'emitted_tokens_journaled':generated,'usage':{'calls':1,'tokens':3+generated},
                'receipt':str((call/'response.json').resolve())}
            write(call/'response.json',self.response)
            write(operation/'service-completion.json',{'usage':{'calls':1,'tokens':3+generated}})
        def poll(self):
            if mode=='unknown':raise ModelOperationUnknown('synthetic transport unknown')
            return {**self.response,'status':'completed' if mode=='response_mismatch' else self.response['status'],
                'parent_end_to_end_wall_seconds':.01,'submitted_ordinal':1}
        def finish(self):counters['fake_finish']+=1;self.alive=False;return None
    real_validator=smoke.validate_model_operation
    def recount(*a):
        result=real_validator(*a)
        if mode=='request_sha_mismatch':
            p=operation/'call-0001/request.json';p.write_text(p.read_text()+' ')
        return result
    out=root/'out';argv=['probe','--run',str(owner),'--model','M1','--reservation','synthetic','--seed','1','--out',str(out)]
    console=io.StringIO();raised=None;code=None
    with patch.object(smoke,'ROOT',repo),patch.object(smoke,'LocalModelClient',FakeClient),\
        patch.object(smoke,'operation_directory',return_value=operation),patch.object(smoke,'validate_model_operation',side_effect=recount),patch.object(sys,'argv',argv),contextlib.redirect_stdout(console):
        try:code=smoke.main()
        except BaseException as e:raised=type(e).__name__+':'+str(e)
    saved=out/'probe.json'
    result=json.loads(saved.read_text()) if saved.exists() else {
        'status':'pre_dispatch_rejection_no_probe_receipt','model_alias':'M1','error':raised or ''}
    return {'mode':mode,'path':str(root),'return_code':code,'raised':raised,'status':result['status'],
        'declared_alias':result.get('model_alias'),'probe_receipt_saved':saved.exists(),**counters,'usage':result.get('resource_recount',{}).get('usage'),
        'generated':result.get('resource_recount',{}).get('generated_tokens_runtime_observed'),
        'call_status':result.get('resource_recount',{}).get('calls',[{}])[0].get('status'),
        'operation_intent_saved':(operation/'service-intent.json').exists(),
        'completion_saved':(operation/'service-completion.json').exists(),'error_tail':result.get('error','').splitlines()[-1:]}

if __name__=='__main__':
    rows=[run(m) for m in ('normal','wrong_kind','wrong_phase','wrong_alias','wrong_registry','wrong_model','wrong_owner_plan',
        'wrong_source','wrong_config','float_config','wrong_request','wrong_seed','bool_seed','request_sha_mismatch')]
    (BASE/'smoke-probe.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
