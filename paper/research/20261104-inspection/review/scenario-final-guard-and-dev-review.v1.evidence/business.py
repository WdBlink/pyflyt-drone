"""Independent saved-business audit; no simulator/scorer imports or physics."""
import collections,hashlib,json,math
from pathlib import Path
ROOT=Path.cwd();R=ROOT/'paper/research/20261104-inspection';Q=R/'qualification';P=Q/'dev-receipts.v2/dev-L3-00-B0-0/attempt-00/flight'
w=json.loads((P/'world.json').read_text());run=json.loads((P/'run.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={'input_sha':{'run':sha(P/'run.json'),'world':sha(P/'world.json'),'contract':sha(R/'planning/gate-contract.v19.json')},'completed':run['status']==w['status']=='completed','source_flags':[run['sources_unchanged'],run['dependencies_unchanged']]}
archive=Q/'source/actual-dev-L3-v2';source={}
for p,s in run['sources'].items():
 rel=Path(p).relative_to(ROOT); h=hashlib.sha256(s.encode()).hexdigest()
 source[str(rel)]={'sha':h,'embedded_match':h==run['source_hashes'][p],'archive_match':(archive/rel).read_text()==s,'current_match':Path(p).read_text()==s}
out['source']=source
out['archive_scaffold']={p:((archive/p/'__init__.py').read_bytes()==b'' if (archive/p/'__init__.py').exists() else None) for p in ['envs','eval']}
out['archive_binding']=json.loads((archive/'binding.json').read_text())
entities={e['key']:e for e in w['hidden_entities']};sensors={e['evidence_id']:e for e in w['sensor_audit']};obs={e['evidence_id']:e for e in w['delivered_observations']}
assert len(sensors)==len(w['sensor_audit']) and len(obs)==len(w['delivered_observations'])
def truth(e,t):
 phase=e['speed']*(t-e['appeared_at'])/e['radius'];dx=e['radius']*math.sin(phase);dy=e['bend']*e['radius']*(1-math.cos(phase));co,si=math.cos(e['heading']),math.sin(e['heading']);return (e['origin_xy'][0]+co*dx-si*dy,e['origin_xy'][1]+si*dx+co*dy)
def wt(e,t):return 3 if t>=w['private_case']['priority_at'] and e['rule_id']=='static-0' else 2 if t>=w['private_case']['priority_at'] and e['rule_id']=='dynamic-inspection' else 1
def quality(v,e):
 if not v['identity']:return 0.,'incorrect_identity'
 if v['cls']!=e['true_class']:return 0.,'incorrect_class'
 if not v['quality']:return 0.,'insufficient_quality'
 er=math.dist(v['xy'],truth(e,v['t']))
 return (1. if er<=3 else 2/3 if er<=5 else 1/3 if er<=10 else 0.),None if er<=10 else 'localization_error'
reports={};idempotents=0;invalid=[]
for idx,a in enumerate(w['reports']):
 if a['response']['status']!='accepted':continue
 q=a['request'];rid=q['report_id'];t=a['t'];refs=[obs[k] for k in q['evidence_ids'] if k in obs]
 if rid not in reports:
  occupied={z['associated'] for z in reports.values() if z['associated'] is not None}
  match=sorted((math.dist(q['estimate_xy'],truth(e,q['measured_at'])),e['key']) for e in entities.values() if e['appeared_at']<=q['measured_at'] and e['key'] not in occupied and math.dist(q['estimate_xy'],truth(e,q['measured_at']))<=20)
  reports[rid]={'associated':match[0][1] if match else None,'kind':q['object_type'],'versions':[]}
 r=reports[rid];causal=bool(q['evidence_ids']) and len(refs)==len(q['evidence_ids']) and 0<=q['measured_at']<=t and all(o['measured_at']<=q['measured_at'] and o['arrived_at']<=t for o in refs) and any(o['measured_at']==q['measured_at'] for o in refs)
 mids=sorted(o['evidence_id'] for o in refs if o['measured_at']==q['measured_at']);mid=mids[0] if mids else 'unsupported-'+str(len(r['versions']))
 v={'id':mid,'t':max(0,min(float(q['measured_at']),t)),'arrival':t,'xy':q['estimate_xy'],'cls':q['claimed_class'],'identity':r['associated'] is not None and bool(refs) and all(sensors[o['evidence_id']]['entity_key']==r['associated'] and o['candidate_id']==q['candidate_id'] for o in refs),'quality':bool(refs) and all(o['quality']>=.8 for o in refs),'provenance':causal,'order':len(r['versions'])+1,'report_audit_index':idx}
 if r['kind']=='dynamic' and any(z['id']==mid for z in r['versions']):
  prior=next(z for z in r['versions'] if z['id']==mid);assert (prior['t'],prior['xy'],prior['cls'])==(v['t'],v['xy'],v['cls']);idempotents+=1;continue
 if not causal:invalid.append((rid,idx))
 r['versions'].append(v)
assert len(reports)<=10
static=[];dynamic=[];allbins=[]
for e in entities.values():
 if not e['rule_id']:continue
 rp=next((r for r in reports.values() if r['associated']==e['key'] and r['kind']==e['object_type']),None);versions=rp['versions'] if rp else []
 if e['object_type']=='static':
  legal=[v for v in versions if v['provenance'] and v['arrival']<=600];v=min(legal,key=lambda z:(-z['arrival'],-z['order'],z['id']),default=None);val,reason=quality(v,e) if v else (0.,'no_measurement')
  static.append({'entity_key':e['key'],'weight':wt(e,600),'quality':val,'selected':v,'reason':reason})
 else:
  tracks={}
  for track in ['delivered','observed']:
   bins=[]
   for i in range(60):
    start=e['appeared_at']+3*i;end=start+3
    legal=[v for v in versions if (start<=v['t']<=end if i==0 else start<v['t']<=end) and v['provenance'] and v['arrival']<=(end+6 if track=='delivered' else 600)]
    v=min(legal,key=lambda z:(-z['t'],z['id']),default=None);val,reason=quality(v,e) if v else (0.,'no_legal_timely_measurement')
    bins.append({'index':i+1,'starts_at':start,'ends_at':end,'weight':float(wt(e,start)),'evidence_id':v['id'] if v else None,'quality':val,'zero_reason':reason})
   tracks[track]={'bins':bins,'numerator':math.fsum(z['weight']*z['quality'] for z in bins),'denominator':math.fsum(z['weight'] for z in bins),'quality':math.fsum(z['weight']*z['quality'] for z in bins)/math.fsum(z['weight'] for z in bins),'reasons':dict(collections.Counter(z['zero_reason'] for z in bins))}
  dynamic.append({'entity_key':e['key'],'tracks':tracks});allbins.extend(tracks['delivered']['bins'])
covered=set();badcoverage=[]
for p in w['coverage_evidence']:
 if not 0<=p['measured_at']<=p['arrived_at']<=600:badcoverage.append(p)
 covered.update(p['cells'])
assert covered<=set(w['required_cells']) and not badcoverage
C=len(covered)/len(w['required_cells']);S=sum(z['weight']*z['quality'] for z in static)/sum(z['weight'] for z in static);D=math.fsum(z['weight']*z['quality'] for z in allbins)/math.fsum(z['weight'] for z in allbins);Q=(C+S+D)/3;H=int(not w['safety']['violations']);E=int(all(v>=20-1e-6 for v in w['safety']['exit_dwell_seconds']) and all(v>=1000 for v in w['score']['remaining_range_m']))
score={'C':C,'S':S,'D':D,'Q':Q,'H':H,'E':E,'Y':H*E*Q,'SR':int(H==E==1 and C>=.8 and S>=.6 and D>=.5)}
out['score_recount']=score;out['score_exact']=all(v==w['score'][k] for k,v in score.items());out['coverage']={'covered':len(covered),'required':len(w['required_cells']),'packets':len(w['coverage_evidence'])}
out['static']=static;out['dynamic']=dynamic;out['hidden_denominators']={'all_entities':len(entities),'required_static':len(static),'normal_distractors':sum(not e['rule_id'] for e in entities.values()),'required_dynamic':len(dynamic),'dynamic_bins':len(allbins),'static_weight':sum(z['weight'] for z in static),'dynamic_weight':math.fsum(z['weight'] for z in allbins)}
out['score_entity_exact']={}
for z in static:
 saved=next(a for a in w['score']['static_entities'] if a['entity_key']==z['entity_key']);out['score_entity_exact'][z['entity_key']]=all(saved[k]==z[k] for k in ['entity_key','weight','quality'])
for z in dynamic:
 saved=next(a for a in w['score']['dynamic_entities'] if a['entity_key']==z['entity_key']);out['score_entity_exact'][z['entity_key']]=saved['quality']==z['tracks']['delivered']['quality'] and saved['observed']==z['tracks']['observed']['quality'] and saved['bins']==z['tracks']['delivered']['bins']
out['report_summary']={'slots':len(reports),'versions':sum(len(r['versions']) for r in reports.values()),'idempotents':idempotents,'invalid_provenance':invalid,'associations':{k:{'entity':v['associated'],'kind':v['kind'],'versions':len(v['versions'])} for k,v in reports.items()}}
out['truth_same_measurement']=all(z['truth_xy'] is None or math.dist(z['truth_xy'],truth(entities[z['entity_key']],z['t']))<1e-12 for z in sensors.values())
out['observation_audit_match']=all(o['measured_at']==sensors[k]['t'] and o['estimate_xy']==sensors[k]['estimate_xy'] and o['estimated_class']==sensors[k]['predicted_class'] and o['quality']==sensors[k]['quality'] for k,o in obs.items())
# Actual link outage: distinguish acquired-in-outage from strictly delayed packets.
out['outage']={}
for key,stamp in [('delivered_observations','measured_at'),('coverage_evidence','measured_at')]:
 packets=[z for z in w[key] if z['drone_id']==1 and z['arrived_at']==279 and z[stamp]<279]
 delayed=[z for z in packets if z[stamp]+.5<279]
 out['outage'][key]={'arrived_at_recovery':len(packets),'strictly_delayed':len(delayed),'measurement_range':[min(z[stamp] for z in packets),max(z[stamp] for z in packets)],'ready_but_pending_at_278':sum(z[stamp]<=278 and z[stamp]+.5<=278 for z in packets),'pending_at_278':sum(z[stamp]<=278 for z in packets)}
 if key=='delivered_observations':out['outage']['observations']=packets
# Exposure-route check from accepted observe job intervals and report audit identity.
routed=[];route_errors=[]
for a in w['reports']:
 q=a['request'];mid=q['evidence_ids'][0]
 if any(c['name']=='submit_or_revise_report' and c['t']==a['t'] and c['args']==q for c in w['tool_calls']):continue
 o=obs[mid]
 jobs=[j for j in w['jobs'] if j['kind']=='observe' and j['drone_id']==o['drone_id'] and j['params']['candidate_id']==o['candidate_id'] and j['created_at']<=o['measured_at']<=j['finished_at'] and j['params']['report_id']==q['report_id']]
 okay=bool(jobs) and a['t']==o['arrived_at'] and q['measured_at']==o['measured_at'] and q['estimate_xy']==o['estimate_xy'] and q['claimed_class']==o['estimated_class']
 routed.append({'id':mid,'report_id':q['report_id'],'jobs':[j['job_id'] for j in jobs],'measured':o['measured_at'],'arrived':o['arrived_at']})
 if not okay:route_errors.append((mid,q))
out['autoroute']={'count':len(routed),'errors':route_errors,'during_outage':sum(244<=z['measured']<279 and obs[z['id']]['drone_id']==1 for z in routed),'post_job_delivery':[(z,next(j['finished_at'] for j in w['jobs'] if j['job_id']==z['jobs'][0])) for z in routed if z['arrived']>next(j['finished_at'] for j in w['jobs'] if j['job_id']==z['jobs'][0])]}
out['retirement']={'post384':{name:sum(z['drone_id']==3 and z[stamp]>=384 for z in w[key]) for name,key,stamp in [('target','sensor_audit','t'),('coverage','coverage_evidence','measured_at'),('opportunities','sensor_opportunities','t'),('background','background_trials','t')]},'old_late':[o for o in obs.values() if o['drone_id']==3 and o['measured_at']<384<=o['arrived_at']]}
# All state transitions and terminal jobs; no ACK is counted as success.
state={};bad=[];active={};overlap=[]
for tr in w['job_transitions']:
 prev=state.get(tr['job_id']);new=tr['status'];j=next(j for j in w['jobs'] if j['job_id']==tr['job_id'])
 if not ((prev is None and new=='queued') or new in {'queued':{'running','cancelled','failed'},'running':{'succeeded','cancelled','failed'}}.get(prev,set())):bad.append(tr)
 if new=='queued':
  if j['drone_id'] in active:overlap.append(tr)
  active[j['drone_id']]=j['job_id']
 if new in {'succeeded','cancelled','failed'}:active.pop(j['drone_id'],None)
 state[tr['job_id']]=new
out['jobs']={'count':len(w['jobs']),'transition_count':len(w['job_transitions']),'illegal':bad,'overlap':overlap,'nonterminal':active,'final_state_match':all(state[j['job_id']]==j['status'] for j in w['jobs']),'kind_status':{str(k):v for k,v in collections.Counter((j['kind'],j['status']) for j in w['jobs']).items()},'observe':[j for j in w['jobs'] if j['kind']=='observe'],'exits':[j for j in w['jobs'] if j['kind']=='exit']}
out['tool_summary']={'count':len(w['tool_calls']),'names':dict(collections.Counter(c['name'] for c in w['tool_calls'])),'all_accepted':all(c['response']['status']=='accepted' for c in w['tool_calls']),'handoff_count':sum(c['name']=='handoff_observation' for c in w['tool_calls'])}
Path('/tmp/scenario-final-review/business.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in out.items() if k in ['input_sha','completed','score_recount','score_exact','coverage','hidden_denominators','score_entity_exact','report_summary','truth_same_measurement','observation_audit_match','tool_summary','autoroute']},indent=2))
for z in dynamic:print('dynamic',z['entity_key'],{t:{k:v for k,v in a.items() if k!='bins'} for t,a in z['tracks'].items()})
print('outage',{k:v for k,v in out['outage'].items() if k!='observations'})
print('retirement',out['retirement']);print('jobs',{k:v for k,v in out['jobs'].items() if k not in ['observe','exits']})
