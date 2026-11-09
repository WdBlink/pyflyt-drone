"""Readonly existing completed event flight. No clean partial/model/physics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO))
BASE=Path('/tmp/inspection-event-endpoint-review-v1');RUN=REPO/'paper/research/20261104-inspection'
flight=REPO/'logs/inspection-population-dev-flights.v1/devext-OOD-L3-00-B0-0/attempt-00/flight'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
s=importlib.util.spec_from_file_location('frozen_event_endpoints',BASE/'source/inspection_event_endpoints.py');ep=importlib.util.module_from_spec(s);s.loader.exec_module(ep)
from eval.inspection_population import load_development_case
world=read(flight/'world.json');run=read(flight/'run.json');stored=read(BASE/'source/event-endpoints-ood-B0.candidate.v1.json')
case=load_development_case(world['private_case']);result=ep.analyze_event_trace(world,case)
canonical=lambda x:json.dumps(x,sort_keys=True,allow_nan=False)
reproduction=all(canonical(result[k])==canonical(stored[k]) for k in ('role','case_sha256','rows','event_count'))
ledger=ep.reconstruct_reports(world,case);entities={e.key:e for e in case.entities}
manual=[]
for row in result['rows']:
    if not row['eligible']:continue
    e=entities[row['entity_key']]
    report=next((r for r in ledger.reports.values() if r.associated_key==e.key and not r.withdrawn),None)
    versions=report.versions if report else []
    if e.object_type=='dynamic':
        bins=next(r for r in world['score']['dynamic_entities'] if r['entity_key']==e.key)['bins']
        selected={v.evidence_id:v for v in versions};ends=[]
        for k in range(51):
            window=bins[k:k+10]
            if window[0]['starts_at']<row['event_time'] or any(b['quality']<.5 for b in window):continue
            complete=max(window[-1]['ends_at'],*[selected[b['evidence_id']].arrived_at for b in window])
            if complete<=row['deadline']:ends.append(complete)
        expected=min(ends) if ends else None
    else:
        ordered=sorted(versions,key=lambda v:(v.arrived_at,v.received_order,v.evidence_id));expected=None
        for i,v in enumerate(ordered):
            if not row['event_time']<v.arrived_at<=row['deadline']-30:continue
            if all(ep.final_static_quality(ordered[:j+1],e.origin_xy,e.true_class,ordered[j].arrived_at)>=.6
                for j in range(i,len(ordered)) if ordered[j].arrived_at<=v.arrived_at+30):
                expected=v.arrived_at;break
    manual.append({'event_type':row['event_type'],'entity_key':e.key,'object_type':e.object_type,
        'manual_recovery_time':row['deadline'] if expected is None else expected,
        'manual_censored':expected is None,'matches_candidate':row['recovery']['time']==(row['deadline'] if expected is None else expected)
            and row['recovery']['censored']==(expected is None)})
summary={}
for kind in sorted({r['event_type'] for r in result['rows']}):
    rows=[r for r in result['rows'] if r['event_type']==kind];eligible=[r for r in rows if r['eligible']]
    summary[kind]={'rows':len(rows),'eligible':len(eligible),'NA':len(rows)-len(eligible),
        'response_censored':sum(r['response']['censored'] for r in eligible),'recovery_censored':sum(r['recovery']['censored'] for r in eligible)}
same_time_report_stamps=sorted({x['t'] for x in world['reports'] if sum(y['t']==x['t'] for y in world['reports'])>1})
events=[x['time'] for x in ep.event_calendar(case)]
at_event_delivered={str(t):sum(o['arrived_at']==t for o in world['delivered_observations']) for t in events}
check={'original_flight_completed':world['status']==run['status']=='completed' and world['simulated_seconds']==600 and world['physics_steps_executed']==144000,
    'clock_record_admitted':world['clock_mode']=='wallclock_1x' and world['max_clock_lag_seconds']<=.1,
    'source_case_guards_recorded':all(run[k] is True for k in ('sources_unchanged','dependencies_unchanged','private_case_unchanged')),
    'world_sha_bound':stored['world_sha256']==run['world_sha256']==sha(flight/'world.json'),
    'case_bytes_bound':sha(flight/'private-case.json')==run['private_case_sha256'],
    'candidate_source_bound':stored['source_sha256']==sha(BASE/'source/inspection_event_endpoints.py'),
    'candidate_rows_exact_reproduced':reproduction,'manual_recoveries_match_this_trace':all(x['matches_candidate'] for x in manual)}
data={'role':'read-only single actual event trace reproduction, not clean pair/effect/qualification','checks':check,'passed':all(check.values()),
    'summary':summary,'eligible_rows':[r for r in result['rows'] if r['eligible']],'manual':manual,
    'report_timestamps_with_multiple_entries':same_time_report_stamps,'delivered_at_exact_event':at_event_delivered,
    'input_sha256':{str(p):sha(p) for p in (flight/'run.json',flight/'world.json',flight/'private-case.json',BASE/'source/design.ral.v3.json',BASE/'source/event-endpoints-ood-B0.candidate.v1.json')}}
(BASE/'audit.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({k:v for k,v in data.items() if k not in ('eligible_rows','input_sha256')},indent=2))
