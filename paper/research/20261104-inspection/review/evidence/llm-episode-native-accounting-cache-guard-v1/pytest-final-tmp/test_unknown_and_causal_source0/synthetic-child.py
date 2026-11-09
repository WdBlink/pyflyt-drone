
from pathlib import Path
import hashlib,json,sys,subprocess
flight,op,planfile,reservation,mode,counter=map(Path,sys.argv[1:])
p=json.loads(planfile.read_text());reservation=str(reservation);mode=str(mode)
count=int(counter.read_text())+1 if counter.exists() else 1;counter.write_text(str(count))
if mode=='orphan':
 leak=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 Path(str(counter)+'.orphan').write_text(str(leak.pid))
flight.mkdir();op.mkdir()
def put(path,v):path.write_text(json.dumps(v)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
(flight/'private-case.json').write_bytes(Path(p['case_path']).read_bytes())
world={'status':'completed','clock_mode':'wallclock_1x','simulated_seconds':600,
 'physics_steps_executed':144000,'physics_hz':240,'control_hz':120,'max_clock_lag_seconds':.02,
 'private_case_file_sha256':p['case_file_sha256'],
 'score':{'C':.5,'S':.5,'D':0,'Q':.3333333333333333,'Y':0.,'H':1,'E':0,'SR':0}}
badclock=mode in ('always-clock','clock-and-seed') or mode=='first-clock' and count==1
if badclock:world['max_clock_lag_seconds']=.2
put(flight/'world.json',world)
put(op/'service-intent.json',{'reservation':reservation,'owner_frozen_plan':p['owner_frozen_plan'],
 'model_receipt_sha256':p['model_receipt_sha256'],'reserved':p['reservation_usage'],
 'source_sha256':p['service_sha256'],'request_budget':p['generation_profile']})
state={'sim_time':0,'task_card':{'task_card_version':1},'aircraft':[],'jobs':[],'events':[],
 'coverage':[],'observations':[],'report_slots_used':0}
messages=[{'role':'system','content':'synthetic unit fixture only'},
 {'role':'user','content':json.dumps({'public_state':state,'recent_tool_results':[],'memory':{}})}]
mh=hashlib.sha256(json.dumps(messages,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
before=[.1,0,0,80];after=[1.,20,0,80]
delay=p.get('injected_delay_seconds',0)
injected=delay if mode in ('injected2','injected20','delay-underwait') else 0
if mode=='delay-underwait':injected=1.5
reply_progress=list(after)
after=[after[0]+injected,*after[1:]]
attempt=2 if mode=='context-gap' else 1
request={'request_ordinal':attempt,'source_sim_time':0,'messages':messages,'messages_sha256':mh}
public={'kind':'public_model_request',**request,'sampler_seed':p['sampler_seed']+attempt-1,'worker_before_request':before}
usage={'calls':0,'tokens':0}
if mode=='not-started':
 response={'kind':'not_started','reason':'frozen local request budget/context exhausted'}
 audit={'status':'ignored_noncompleted_or_late','request':request}
else:
 call=op/'call-0001';call.mkdir();text='{"commands":[]}'
 put(call/'request.json',{'messages':messages,'seed':p['sampler_seed']+attempt-1+(mode in ('wrong-seed','clock-and-seed')),
  'input_tokens_submitted':3,'max_output_tokens':p['generation_profile']['max_output_tokens'],
  'temperature':.2,'top_p':.95,'enable_thinking':False})
 (call/'tokens.jsonl').write_text(json.dumps({'token':1,'text':text,'runtime_prompt_tokens':3,'runtime_generation_tokens':1})+'\n')
 late=mode=='policy-timeout'
 response={'kind':'response','status':'policy_timeout' if late else 'completed','text':text,
  'generated_tokens_observed':1,'emitted_tokens_journaled':1,'input_tokens_submitted':3,
  'usage':{'calls':1,'tokens':4},'wall_seconds':61. if late else 1.,'deadline_exceeded':late,
  'receipt':str((call/'response.json').resolve())}
 put(call/'response.json',response)
 usage={'calls':1,'tokens':4}
 audit={'status':'ignored_noncompleted_or_late','request':request} if late else {
  'status':'decoded_not_tool_acceptance','request':request,'reply':{'commands':[]},'wire_framing':{'json_fence_removed':False}}
if mode!='missing-completion':put(op/'service-completion.json',{'usage':usage})
rows=[]
if mode=='context-gap':rows.append({'kind':'policy_context_failure','decision_attempt':1,'model_request_dispatched':False})
parent_wall=59. if mode=='injected20' else 1.1
rows += [public,{'kind':'actual_model_response','response':{**response,'submitted_ordinal':1,'parent_end_to_end_wall_seconds':parent_wall},
 'worker_before_request':before,'worker_after_response':after,'worker_at_model_reply':reply_progress,
 'injected_delay_requested_seconds':delay,'injected_delay_wall_seconds':injected,'final_after_flight':False},
 {'kind':'decoded_reply','audit':audit,'commands_enqueued_not_accepted':[],'worker_progress_at_enqueue':after,
  'dispatch_age_sim_seconds':after[0],'dispatch_age_wall_seconds':parent_wall+injected}]
if mode=='bad-enqueue':rows[-1]['worker_progress_at_enqueue']=[601,0,0,80]
(flight/'actor.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
run={'status':'completed','source_hashes':p['source_hashes'],'dependency_hashes':p['dependency_hashes'],
 'sources':{name:Path(name).read_text() for name in p['source_hashes']},
 'injected_delay_seconds':20 if mode=='wrong-delay' else delay,
 'sources_unchanged':mode!='source-fault','dependencies_unchanged':True,
 'private_case_sha256':p['case_file_sha256'],'private_case_unchanged':True,'policy':p['condition'],
 'runtime_thread_environment':p['thread_environment'],'requested_seconds':600,'realtime':True,
 'world_sha256':sha(flight/'world.json'),'reservation':reservation,'sampler_base_seed':p['sampler_seed'],
 'ablation':None,'model_registry_sha256':p['model_registry_sha256'],'model_receipt_sha256':p['model_receipt_sha256'],
 'model_generation_profile':p['generation_profile'],'actor_mutating_calls':0,
 'model_operation':{'usage':usage,'sha256':sha(op/'service-completion.json') if (op/'service-completion.json').exists() else None}}
if mode=='embedded-source':run['sources']={name:'mismatched embedded source' for name in p['source_hashes']}
put(flight/'run.json',run)
