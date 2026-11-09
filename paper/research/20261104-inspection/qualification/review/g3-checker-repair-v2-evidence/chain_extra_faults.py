from copy import deepcopy
from pathlib import Path
import json
from eval.qualify_inspection_reference import causal_chain_witness
def view(e,t,x):
 return dict(evidence_id=e,candidate_id='c',drone_id=0,quality=.95,sensor_mode='inspect',measured_at=t,arrived_at=t+.5,aircraft_xy=[x,0.])
original=dict(simulated_seconds=600.,delivered_observations=[view('a',1.5,0.),view('b',2.5,25.),view('local',4.5,50.)],jobs=[dict(job_id='inspect',drone_id=0,kind='inspect',status='succeeded',created_at=1.,finished_at=3.,params=dict(candidate_id='c')),dict(job_id='observe',drone_id=0,kind='observe',status='running',created_at=4.,params=dict(candidate_id='c',report_id='r',service_started_at=4.5))],reports=[dict(t=3.5,request=dict(candidate_id='c',report_id='r',object_type='dynamic',measured_at=2.5,evidence_ids=['b']),response=dict(status='accepted'))])
results={}
for name in ['future_measurement_claim','no_same_time_evidence','report_uses_old_survey']:
 w=deepcopy(original)
 if name=='future_measurement_claim':w['reports'][0]['request']['measured_at']=999.
 elif name=='no_same_time_evidence':w['reports'][0]['request']['measured_at']=3.
 else:
  old=deepcopy(w['delivered_observations'][0]);old.update(evidence_id='old-survey',sensor_mode='survey',measured_at=.5,arrived_at=1.);w['delivered_observations'].append(old);w['reports'][0]['request'].update(measured_at=.5,evidence_ids=['old-survey'])
 witness=causal_chain_witness(w);assert witness is not None
 results[name]={'input':w,'accepted_witness':witness}
Path(__file__).with_name('chain-extra-inputs-and-results.json').write_text(json.dumps(results,indent=2)+'\n');print('3 isolated false-witness counterexamples confirmed on70dff checker')
