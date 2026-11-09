"""Read stored acquisition/arrival/report-route joins; no physics or replay."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
R=OUT.parents[2]
D=R/'qualification/dev-same-level-event-pair.v2'
pair=json.loads((D/'pair.json').read_text())
w=json.loads((Path(pair['flights']['event']['root'])/'attempt-00/flight/world.json').read_text())
c=w['private_case'];sensor={s['evidence_id']:s for s in w['sensor_audit']};obs={s['evidence_id']:s for s in w['delivered_observations']}
buffer=[s for s in obs.values() if s['drone_id']==c['link_drone'] and c['link_start']<=s['measured_at']<c['link_end']]
target=[s for s in buffer if sensor[s['evidence_id']]['entity_key'] is not None]
dynamic=[s for s in target if s['object_type']=='dynamic']
entities={e['key']:e for e in c['entities']}
required=[s for s in target if entities[sensor[s['evidence_id']]['entity_key']]['rule_id']]
routes=[]
for entry in w['reports']:
    if entry['t']!=c['link_end']:continue
    req=entry['request']
    for evidence_id in req['evidence_ids']:
        o=obs[evidence_id]
        assert entry['t']>=o['arrived_at'] and req['measured_at']==o['measured_at'] and req['candidate_id']==o['candidate_id'] and req['estimate_xy']==o['estimate_xy']
        matches=[j for j in w['jobs'] if j['kind']=='observe' and j['drone_id']==o['drone_id'] and j['params']['report_id']==req['report_id']
                 and j['started_at'] is not None and j['started_at']<=o['measured_at'] and (j['finished_at'] is None or o['measured_at']<j['finished_at'])]
        assert len(matches)==1
        job=matches[0]
        routes.append({'evidence_id':evidence_id,'measured_at':o['measured_at'],'arrived_at':o['arrived_at'],'report_t':entry['t'],
            'report_id':req['report_id'],'acquisition_observe_job_id':job['job_id'],'job_finished_at':job['finished_at'],'job_final_status':job['status'],
            'finished_before_upload':job['finished_at'] is not None and job['finished_at']<o['arrived_at']})
old=[s for s in obs.values() if s['drone_id']==c['retire_drone'] and s['measured_at']<c['retire_at']<=s['arrived_at']]
retirement_jobs=[j for j in w['jobs'] if j['drone_id']==c['retire_drone'] and (j['created_at']==c['retire_at'] or j['finished_at']==c['retire_at'])]
cancelled=next(j for j in retirement_jobs if j['kind']=='observe' and j['status']=='cancelled')
exit_job=next(j for j in retirement_jobs if j['kind']=='exit' and j['status']=='succeeded')
assert exit_job['params']['origin']=='exogenous_capacity_exit' and exit_job['version']==cancelled['version']+1
old_ids={s['evidence_id'] for s in old}
retirement_reports=[r for r in w['reports'] if old_ids&set(r['request']['evidence_ids'])]
assert len(retirement_reports)==1
assert retirement_reports[0]['t']==c['retire_at'] and retirement_reports[0]['request']['report_id']==cancelled['params']['report_id']
assert retirement_reports[0]['request']['measured_at']==371.5 and retirement_reports[0]['response']['status']=='accepted'
result={'passed':True,'window':[c['link_start'],c['link_end']],'buffered_observations':len(buffer),'buffered_nonbackground_entity_observations':len(target),
 'buffered_required_task_observations':len(required),
 'buffered_dynamic_observations':len(dynamic),'all_original_measurement_timestamps_join_sensor':all(s['measured_at']==sensor[s['evidence_id']]['t'] for s in buffer),
 'all_arrived_at_recovery':all(s['arrived_at']==c['link_end'] for s in buffer),'recovery_report_routes':routes,
 'recovery_report_entries_from_disconnected_drone':sum(obs[s['evidence_id']]['drone_id']==c['link_drone'] for s in routes),
 'postretire_old_observations':old,'retirement_jobs':retirement_jobs,'same_tick_postretirement_old_route_reports':retirement_reports}
(OUT/'link-route-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('recovery_report_routes','postretire_old_observations')},indent=2))
print('routes',len(routes),'finished_before_upload',sum(s['finished_before_upload'] for s in routes))
