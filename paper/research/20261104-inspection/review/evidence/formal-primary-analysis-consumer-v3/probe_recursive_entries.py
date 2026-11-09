from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys,json,hashlib,tempfile,io,contextlib,os,socket,shutil
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';W=R/'planning/formal-admission-candidate.v3';F=W/'primary-analysis-freeze.v4';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();b=json.loads((F/'binding.json').read_text());assert sha(F/'binding.json')=='1eb375245f88fc9c29a21c828a75a04e3112937ec02c61a1705d0d2b8e4e5216'
for n,v in b['members'].items():assert sha(F/n)==v['sha256'] and sha(W/n)==v['sha256']
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('adjacent_analysis_fixture',W/'tests/test_formal_analysis.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h);a=h.analysis
spec=importlib.util.spec_from_file_location('independent_original_meter',W/'secondary-consumer-runtime-freeze.v1/source/eval/inspection_model_receipts.py');meter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=meter;spec.loader.exec_module(meter)
project=Path(tempfile.mkdtemp(prefix='pa4-',dir='/tmp')).resolve();run=project/'run';episode=project/'logs/receipt';operation=project/'logs/operation';rid='formal-main-'+('a'*12)+'-0000-a0';resource=run/'qualification/local-model-episodes'/rid
write=lambda p,v:(p.parent.mkdir(parents=True,exist_ok=True),p.write_text(json.dumps(v,sort_keys=True)+'\n'))
write(episode/'plan.json',{'role':'synthetic'});write(episode/'attempt-00/terminal.json',{'role':'synthetic'});write(resource/'usage-receipt.json',{'role':'synthetic'})
reserve=run/'resource-requests'/f'{rid}.reserve.json';settle=run/'resource-requests'/f'{rid}.settle.json';write(reserve,{'role':'synthetic'});write(settle,{'role':'synthetic'})
profile={'max_calls':32,'max_total_tokens':180000,'max_input_tokens':8192,'max_output_tokens':1024,'max_request_seconds':60,'temperature':.2,'top_p':.95,'enable_thinking':False};expected={'reservation':rid,'owner_frozen_plan':'b'*64,'model_receipt_sha256':'c'*64,'service_sha256':'d'*64,'reserved':{'calls':32,'tokens':180000},'generation_profile':profile}
write(operation/'service-intent.json',{**{k:expected[k] for k in ['reservation','owner_frozen_plan','model_receipt_sha256','reserved']},'source_sha256':expected['service_sha256'],'request_budget':profile});write(operation/'service-completion.json',{'usage':{'calls':1,'tokens':4}})
call=operation/'call-0001';write(call/'request.json',{'input_tokens_submitted':3,'max_output_tokens':1024,'temperature':.2,'top_p':.95,'enable_thinking':False});write(call/'response.json',{'kind':'response','status':'completed','text':'x','receipt':str(call/'response.json'),'usage':{'calls':1,'tokens':4},'input_tokens_submitted':3,'generated_tokens_observed':1,'emitted_tokens_journaled':1,'wall_seconds':.01,'deadline_exceeded':False});write(call/'tokens.jsonl',{'token':1,'text':'x','runtime_prompt_tokens':3,'runtime_generation_tokens':1})
assert meter.validate_model_operation(operation,expected)['usage']=={'calls':1,'tokens':4}
seen=[]
def directory(reservation):assert reservation==rid;seen.append(reservation);return operation
a.operation_directory=directory;slot=SimpleNamespace(sampler_seed=1,order_index=0);request={'run':str(run),'opening':None,'g4':{'sha256':'a'*64}};before=a._receipt_snapshot(episode,slot,request,project);snap=[(slot,episode,before)];a._check_snapshots(snap,request,project)

assert sha(W/'secondary-consumer-runtime-freeze.v1/source/eval/inspection_model_receipts.py') == 'f68f149960254ef9e50a47354722d7f64d83417adda26a845fb076847dab6a49'
result = {'normal_valid_one_call_meter': True, 'normal_snapshot_passed': True,
          'subject_binding_sha256': sha(F/'binding.json'),
          'scope': 'synthetic valid one-call meter and exact frozen v4 snapshot/check helpers only; no model/physics/native operation/formal RNG'}

def rejected(action, needle):
    try:
        action()
    except ValueError as error:
        assert needle in str(error), str(error)
        return str(error)
    raise AssertionError('expected ValueError')

late=operation/'call-0002';late.mkdir()
after=a._receipt_snapshot(episode,slot,request,project)
assert after != before and str(late.relative_to(project)) in after['directories']
assert after['files'] == before['files']
result['new_empty_call_directory_in_bound_set'] = True
result['new_empty_call_directory_rejected'] = rejected(lambda:a._check_snapshots(snap,request,project),'closed receipt input changed')
try:
    meter.validate_model_operation(operation,expected)
except meter.UnresolvedModelOperation as error:
    result['actual_meter_rejects_new_empty_call_directory'] = str(error)
else:
    raise AssertionError('actual meter must reject empty call directory')
late.rmdir();a._check_snapshots(snap,request,project)
assert meter.validate_model_operation(operation,expected)['usage']=={'calls':1,'tokens':4}

fifo=operation/'unexpected-fifo';os.mkfifo(fifo)
result['FIFO_rejected_before_file_read'] = rejected(lambda:a._receipt_snapshot(episode,slot,request,project),'regular files or directories')
result['FIFO_rejected_by_final_guard'] = rejected(lambda:a._check_snapshots(snap,request,project),'regular files or directories')
fifo.unlink();a._check_snapshots(snap,request,project)

# The same nonregular-entry branch must also reject a bound local UNIX socket.
sockpath=operation/'unexpected-socket';sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
try:
    sock.bind(str(sockpath))
    result['UNIX_socket_rejected'] = rejected(lambda:a._check_snapshots(snap,request,project),'regular files or directories')
finally:
    sock.close();sockpath.unlink(missing_ok=True)

for label,target in [('file',call/'request.json'),('directory',call),('broken',operation/'absent')]:
    link=operation/('unexpected-'+label+'-symlink');link.symlink_to(target)
    try:
        result['symlink_'+label+'_rejected'] = rejected(lambda:a._check_snapshots(snap,request,project),'cannot traverse symlinks')
    finally:
        link.unlink()
a._check_snapshots(snap,request,project)
assert meter.validate_model_operation(operation,expected)['usage']=={'calls':1,'tokens':4}
result['restored_normal_meter_and_snapshot_pass'] = True
for n,v in b['members'].items():
    assert sha(F/n)==v['sha256'] and sha(W/n)==v['sha256']
result['five_frozen_and_working_member_hashes_stable'] = True
shutil.rmtree(project)
(OUT/'delta-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
