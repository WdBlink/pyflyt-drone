from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys,json,hashlib,tempfile,io,contextlib
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';W=R/'planning/formal-admission-candidate.v3';F=W/'primary-analysis-freeze.v3';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();b=json.loads((F/'binding.json').read_text());assert sha(F/'binding.json')=='e24c021d1ddf297712c756bab2a06d2fe9c0eb3a4d67bb2c8f76e5b6d276186e'
for n,v in b['members'].items():assert sha(F/n)==v['sha256'] and sha(W/n)==v['sha256']
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('adjacent_analysis_fixture',W/'tests/test_formal_analysis.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h);a=h.analysis
spec=importlib.util.spec_from_file_location('independent_original_meter',W/'secondary-consumer-runtime-freeze.v1/source/eval/inspection_model_receipts.py');meter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=meter;spec.loader.exec_module(meter)
project=Path(tempfile.mkdtemp(prefix='closure-unit-',dir=OUT)).resolve();run=project/'run';episode=project/'logs/receipt';operation=project/'logs/operation';rid='formal-main-'+('a'*12)+'-0000-a0';resource=run/'qualification/local-model-episodes'/rid
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
late=operation/'call-0002';late.mkdir();same=a._receipt_snapshot(episode,slot,request,project)==before
a._check_snapshots(snap,request,project)
try:meter.validate_model_operation(operation,expected)
except meter.UnresolvedModelOperation as e:meter_reason=str(e)
else:raise AssertionError('actual meter must reject empty call directory')
assert same
result={'normal_valid_one_call_meter':True,'normal_snapshot_passed':True,'new_empty_call_directory_snapshot_unchanged':same,'new_empty_call_directory_check_incorrectly_passed':True,'actual_meter_rejects_new_empty_call_directory':meter_reason,'scope':'fully synthetic one-call journal, no generation/native resource'}
late.rmdir();new=operation/'late.json';write(new,{'new':True})
try:a._check_snapshots(snap,request,project)
except ValueError as e:assert 'closed receipt input changed' in str(e);result['new_operation_file_rejected']=True
else:raise AssertionError('file addition must reject')
new.unlink();old=settle.read_bytes();write(settle,{'changed':True})
try:a._check_snapshots(snap,request,project)
except ValueError as e:assert 'closed receipt input changed' in str(e);result['canonical_settlement_bytes_rejected']=True
else:raise AssertionError('settlement change must reject')
settle.write_bytes(old);a._check_snapshots(snap,request,project);assert meter.validate_model_operation(operation,expected)['usage']=={'calls':1,'tokens':4}
cli=project/'request.json';write(cli,{'role':'literal CLI fixture'});a.analyze_primary=lambda *args:{'role':'subject-not-lock','accepted':False,'note':'中文'};sys.argv=['analysis','--request',str(cli),'--recording',str(project/'unused')]
with contextlib.redirect_stdout(io.StringIO()) as out:a.main()
assert json.loads(out.getvalue())=={'role':'subject-not-lock','accepted':False,'note':'中文'};result['CLI_strict_JSON_passed']=True
(OUT/'adjacent-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
