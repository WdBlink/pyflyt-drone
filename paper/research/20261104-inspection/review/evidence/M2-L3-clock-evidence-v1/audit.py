"""Independent offline artifact recount. No production imports, model, physics or owner writes."""
from array import array
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
import argparse
import base64
import hashlib
import json
import math
import statistics
import sys
import zipfile
import zlib


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p): return json.loads(p.read_bytes())
def canonical(v): return sha_bytes(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def summary(values):
    s=sorted(values)
    return {'count':len(s),'sum':math.fsum(values),'min':s[0],'median':statistics.median(s),
            'p95_floor':s[int(.95*(len(s)-1))],'max':s[-1]}
def doubles(p,byte_order):
    a=array('d');a.frombytes(p.read_bytes())
    if byte_order!=sys.byteorder:a.byteswap()
    assert all(math.isfinite(x) for x in a),str(p)
    return a

def clock_recount(raw):
    meta=read(raw/'flight/clock-profile.json');data=doubles(raw/'flight/clock-ticks.f64',meta['byte_order'])
    fields=meta['fields'];width=len(fields);ix={f:i for i,f in enumerate(fields)}
    assert width==30 and meta['rows']==72000 and len(data)==72000*width
    for name,entry in meta['files'].items():
        f=raw/'flight'/name;assert sha(f)==entry['sha256'] and f.stat().st_size==entry['bytes']
    at=lambda n,k:data[n*width+ix[k]]
    stages=['begin','wake','rpc_end','step_end','progress_end','stats_end','pub_end','log_end']
    epoch=meta['anchor']['worker_monotonic_epoch'];lags=[];sensor=0;rpc=0
    durations={f'{a}_to_{b}':{k:[] for k in ('wall','thread_cpu','process_cpu')} for a,b in zip(stages,stages[1:])}
    for n in range(72000):
        if n:
            assert all(at(n,'begin_'+k)>=at(n-1,'log_end_'+k)-1e-9 for k in ('wall','thread_cpu','process_cpu'))
        assert abs(at(n,'sim_before')-n/120)<1e-9 and abs(at(n,'sim_after')-(n+1)/120)<1e-9
        assert at(n,'sleep_requested')>=0
        assert at(n,'rpc_count').is_integer() and 0<=at(n,'rpc_count')<=16
        assert at(n,'sensor_tick')==int((n+1)%60==0)
        sensor+=int(at(n,'sensor_tick'));rpc+=int(at(n,'rpc_count'));lag=at(n,'lag');lags.append(lag)
        assert at(n,'progress_end_wall')-epoch-at(n,'sim_after')-1e-7<=lag<=at(n,'stats_end_wall')-epoch-at(n,'sim_after')+1e-7
        for a,b in zip(stages,stages[1:]):
            for kind in ('wall','thread_cpu','process_cpu'):
                delta=at(n,b+'_'+kind)-at(n,a+'_'+kind)
                assert delta>=-1e-9,(n,a,b,kind,delta)
                durations[f'{a}_to_{b}'][kind].append(delta)
    s=sorted(lags);out={'rows':72000,'fields':30,'tick_bytes':len(data)*8,'sim_first_before':at(0,'sim_before'),
        'sim_last_after':at(71999,'sim_after'),'max_lag':max(lags),'p95_lag':s[int(.95*(len(s)-1))],
        'over_0p1':sum(x>.1 for x in lags),'peak_index_zero_based':lags.index(max(lags)),
        'sensor_ticks':sensor,'rpc_envelopes':rpc,'stage_intervals':{k:{q:summary(v) for q,v in z.items()} for k,z in durations.items()},
        'gc_start_configuration':meta['gc_configuration_at_start'],'gc_overflow':meta['gc_overflow'],
        'epoch':epoch,'anchor':meta['anchor'],'initial_publication_wall_seconds':meta['initial_publication_wall_seconds']}
    assert out['max_lag']==meta['max_lag'] and out['p95_lag']==meta['p95_lag'] and out['over_0p1']==meta['ticks_over_0p1']
    peak=out['peak_index_zero_based'];out['peak_tick']={k:at(peak,k) for k in fields}
    gcmeta=meta['gc_fields'];gcdata=doubles(raw/'flight/clock-gc-events.f64',meta['byte_order']);assert len(gcdata)==meta['gc_rows']*len(gcmeta)
    events=[dict(zip(gcmeta,gcdata[n:n+len(gcmeta)])) for n in range(0,len(gcdata),len(gcmeta))]
    pending={};pairs=[]
    for e in events:
        key=(e['thread_id'],e['generation']);assert e['event'] in (1.,2.)
        if e['event']==1.:
            assert key not in pending;pending[key]=e
        else:
            assert key in pending;start=pending.pop(key);assert e['wall']>=start['wall']
            pairs.append({'start_event':start,'stop_event':e,'generation':int(e['generation']),'thread_id':int(e['thread_id']),
                'wall_seconds':e['wall']-start['wall'],'thread_cpu_seconds':e['thread_cpu']-start['thread_cpu'],
                'process_cpu_seconds':e['process_cpu']-start['process_cpu'],'start_tick':int(start['tick_index']),
                'stop_tick':int(e['tick_index']),'start_stage':int(start['stage_index']),
                'start_control_active':bool(start['control_active']),'collected':e['collected'],'uncollectable':e['uncollectable']})
    assert not pending
    out['gc']={'events':len(events),'pairs':len(pairs),'unmatched':0,'generations':dict(Counter(x['generation'] for x in pairs)),
        'main_thread_pairs':sum(x['thread_id']==meta['main_thread_id'] for x in pairs),
        'control_active_at_start':sum(x['start_control_active'] for x in pairs),
        'wall_intervals':summary([x['wall_seconds'] for x in pairs]),'longest5':sorted(pairs,key=lambda x:x['wall_seconds'],reverse=True)[:5]}
    out['gc']['all_pairs']=pairs
    rm=read(raw/'flight/clock-rusage-profile.json');rd=doubles(raw/'flight/clock-rusage.f64',rm['byte_order']);rw=len(rm['fields']);ri={k:i for i,k in enumerate(rm['fields'])}
    assert rm['rows']==72000 and len(rd)==72000*rw and rw==30
    assert sha(raw/'flight/clock-rusage.f64')==rm['binary_sha256'] and len(rd)*8==rm['binary_bytes']
    assert sha(raw/'flight/clock-profile.json')==rm['base_clock_profile_sha256']
    counters=['ru_utime','ru_stime','ru_minflt','ru_majflt','ru_nswap','ru_nvcsw','ru_nivcsw','ru_maxrss']
    samples={k:[] for k in counters};brackets=[];last_rusage_end=float('-inf')
    for n in range(72000):
        for st in ('begin','wake','log_end'):
            b=rd[n*rw+ri[st+'_wall_before']];e=rd[n*rw+ri[st+'_wall_after']];assert e>=b>=last_rusage_end
            assert b>=at(n,st+'_wall')
            following='wake' if st=='begin' else 'rpc_end' if st=='wake' else None
            if following:assert e<=at(n,following+'_wall')
            elif n<71999:assert e<=at(n+1,'begin_wall')
            last_rusage_end=e;brackets.append(e-b)
            for k in counters:samples[k].append(rd[n*rw+ri[st+'_'+k]])
    changes={}
    for key,values in samples.items():
        delta=[b-a for a,b in zip(values,values[1:])]
        changes[key]={'samples':len(values),'first':values[0],'last':values[-1],'signed_net':values[-1]-values[0],
            'adjacent_negative_count':sum(x<0 for x in delta),'minimum_delta':min(delta),'maximum_delta':max(delta),
            'signed_delta_sum':math.fsum(delta)}
    out['rusage']={'rows':72000,'snapshot_count':216000,'bytes':len(rd)*8,'counter_changes_raw_unclipped':changes,
        'syscall_wall_brackets':summary(brackets),'qos':rm['requested_qos_before_epoch']}
    world=read(raw/'flight/world.json')
    wc=dict(world['clock_profile']);rr=wc.pop('rusage_profile');assert wc==meta
    assert rr=={'path':'clock-rusage-profile.json','sha256':sha(raw/'flight/clock-rusage-profile.json'),'rows':72000}
    assert world['maximum_clock_lag_witness']['sim_time']==out['peak_tick']['sim_after']
    assert world['maximum_clock_lag_witness']['lag_seconds']==out['max_lag']
    assert world['status']=='completed' and world['simulated_seconds']==600 and world['physics_steps_executed']==144000
    assert world['physics_hz']==240 and world['control_hz']==120
    assert world['safety']['physics_audits']==144000 and world['execution_tracking_240hz']['physics_samples']==144000
    assert world['max_clock_lag_seconds']==out['max_lag'] and world['clock_lag_p95_seconds']==out['p95_lag'] and world['clock_lag_over_0p1_seconds_ticks']==out['over_0p1']
    trace=world['execution_tracking_trace'];assert trace['encoding']=='base64-zlib-little-endian-float64'
    trace_raw=zlib.decompress(base64.b64decode(trace['data'],validate=True));assert sha_bytes(trace_raw)==trace['raw_sha256']
    trace_data=array('d');trace_data.frombytes(trace_raw)
    if sys.byteorder!='little':trace_data.byteswap()
    assert trace['rows']==144000 and len(trace['columns'])==17 and len(trace_data)==144000*17
    assert all(math.isfinite(x) for x in trace_data)
    assert all(abs(trace_data[n*17]-(n+1)/240)<1e-9 for n in range(144000))
    out['tracking_error_trace']={'rows':144000,'columns':17,'raw_bytes':len(trace_raw),'raw_sha256':trace['raw_sha256'],
        'first_t':trace_data[0],'last_t':trace_data[-17],'scope':'error/z/interval samples, not full physical-state replay'}
    out['physics']={'simulated_seconds':world['simulated_seconds'],'wall_seconds':world['wall_seconds'],'physics_steps':world['physics_steps_executed'],
        'safety_audits':world['safety']['physics_audits'],'tracking_samples':world['execution_tracking_240hz']['physics_samples']}
    return out


def main():
    repo=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=repo/'paper/research/20261104-inspection';Q=R/'qualification/clock-localization-M2-L3.v1';F=Q/'freeze.v1';raw=repo/'logs/inspection-clock-localization-M2-L3.v1';out=Path('/tmp/inspection-M2-L3-clock-evidence-review')
    binding=read(Q/'actual-diagnostic-binding.v1.json');assert sha(Q/'actual-diagnostic-binding.v1.json')=='aff1ca1675c890427dd02e20b5da5947b28ebfd66323d37addf8e88e2240f2f6'
    freeze=read(F/'binding.json');assert sha(F/'binding.json')=='b6aa5715bf4855db70f5eff2fcc5f1b954951f5722ff27479b1b4d539e87e068'
    assert len(freeze['files'])==111 and len(freeze['live_source_hashes'])==107 and len(freeze['dependencies'])==61 and len(freeze['input_hashes'])==11
    for p,h in freeze['files'].items():assert sha(F/p)==h,p
    for group in ['live_source_hashes','dependencies','input_hashes']:
        for p,h in freeze[group].items():assert sha(Path(p))==h,p
    sources=freeze['live_source_hashes'];intent=read(R/freeze['intent']['path']);diag=read(raw/'diagnostic.json');receipt=read(Q/'usage-receipt.v1.json');op=repo/'logs/inspection-model-resource-operations.v1'/diag['expected_model_meter']['reservation'];archive=R/binding['archive']['path'];members=read(Q/'archive-members.v1.json')
    assert sha(archive)==receipt['archive_sha256']==binding['archive']['sha256']=='ce6f65637a989b01ce35eefa707f5fe009cc0af0209a41cab72a871c7b39e852'
    assert sha(raw/'diagnostic.json')==binding['envelope_sha256'];assert sha(Q/'usage-receipt.v1.json')==binding['usage_receipt_sha256'];assert sha(Q/'phase-recount.v1.json')==binding['phase_recount_sha256']
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names))==len(members)==106 and set(names)==set(members)
        for n in names:
            assert not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
            assert sha_bytes(z.read(n))==members[n]
            p=raw/n[len('diagnostic/'):] if n.startswith('diagnostic/') else op/n[len('model-operation/'):]
            assert sha(p)==members[n]
        current={'diagnostic/'+str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file()}|{'model-operation/'+str(p.relative_to(op)) for p in op.rglob('*') if p.is_file()}
        assert current==set(names)
    run=read(raw/'flight/run.json');world=read(raw/'flight/world.json');assert run['status']=='completed' and run['requested_seconds']==600 and run['realtime'] is True
    assert run['policy']=='B3-M2' and run['sampler_base_seed']==1096691202 and run['injected_delay_seconds']==0 and run['ablation'] is None
    assert all(run[k] is True for k in ['sources_unchanged','dependencies_unchanged','private_case_unchanged'])
    assert sha(raw/'flight/world.json')==run['world_sha256'];assert len(run['sources'])==23 and set(run['sources'])==set(run['source_hashes'])
    for p,text in run['sources'].items():assert sha_bytes(text.encode())==run['source_hashes'][p]==sources[p]
    assert run['dependency_hashes']==freeze['dependencies'];assert diag['extra_worker_variant_sources']==intent['extra_sources']
    assert sha(raw/'flight/private-case.json')==sha(R/intent['case']['path'])==intent['case']['sha256']==diag['case_file_sha256']==run['private_case_sha256']==world['private_case_file_sha256']
    assert diag['status']=='completed_engineering_not_qualification' and diag['errors']==[] and diag['returncode']==0 and diag['worker_variant_sources_unchanged'] and diag['original_case_unchanged'] and diag['no_SDK_or_pilot_slot_admission']
    sup=diag['supervision'];assert sup['root_pid']==diag['coordinator_pid'] and sup['status']=='closed' and sup['remaining_owned']==[] and sup['unclaimed_new_python_orphans']==[]
    assert diag['supervisor_inventory']=={'hot_native':5824,'hot_legacy':0,'fallback_reasons':{}} and diag['supervisor_observation_overflow']==0 and len(diag['supervisor_observations'])==5825
    clock=clock_recount(raw);assert clock['rows']==72000 and clock['gc']['pairs']==116 and clock['gc_overflow']==0 and clock['over_0p1']==32 and clock['max_lag']>.1
    assert diag['max_clock_lag_seconds']==clock['max_lag'] and diag['lag_threshold_exceeded'] is True
    reported=read(Q/'phase-recount.v1.json')
    for n,h in reported['inputs'].items():assert sha(raw/'diagnostic.json' if n=='diagnostic_envelope' else raw/'flight'/n)==h
    assert reported['statistics']=={'max_lag':clock['max_lag'],'p95_lag':clock['p95_lag'],'ticks_over_0p1':clock['over_0p1']} and reported['rows']==72000 and reported['gc_pairs']==116 and reported['gc_overflow']==0 and reported['peak_tick_index']==clock['peak_index_zero_based']
    meta=read(raw/'flight/clock-profile.json');a=doubles(raw/'flight/clock-ticks.f64',meta['byte_order']);width=len(meta['fields']);ix={k:i for i,k in enumerate(meta['fields'])};at=lambda n,k:a[n*width+ix[k]]
    rm=read(raw/'flight/clock-rusage-profile.json');ru=doubles(raw/'flight/clock-rusage.f64',rm['byte_order']);rw=len(rm['fields']);ri={k:i for i,k in enumerate(rm['fields'])};rat=lambda n,s,k:ru[n*rw+ri[s+'_'+k]]
    gcpairs=clock['gc']['all_pairs'];counters=['ru_utime','ru_stime','ru_minflt','ru_majflt','ru_nswap','ru_nvcsw','ru_nivcsw','ru_maxrss']
    def window(n,start,end,ru_start,ru_end):
        lo,hi=at(n,start+'_wall'),at(n,end+'_wall')
        overlaps=[g for g in gcpairs if g['start_event']['wall']<hi and g['stop_event']['wall']>lo]
        return {'tick_index':n,'sim_after':at(n,'sim_after'),'lag':at(n,'lag'),'sensor_tick':at(n,'sensor_tick'),'rpc_count':at(n,'rpc_count'),'sleep_requested':at(n,'sleep_requested'),'phase':start+'__'+end,'wall_start':lo,'wall_end':hi,'wall_seconds':hi-lo,'thread_cpu_seconds':at(n,end+'_thread_cpu')-at(n,start+'_thread_cpu'),'process_cpu_seconds':at(n,end+'_process_cpu')-at(n,start+'_process_cpu'),'all_recorded_gc_overlaps':overlaps,'rusage_snapshot_interval':ru_start+'__'+ru_end,'rusage_deltas_raw_unclipped':{k:rat(n,ru_end,k)-rat(n,ru_start,k) for k in counters},'rusage_brackets':{s:{k:rat(n,s,k) for k in ('wall_before','wall_after')} for s in [ru_start,ru_end]}}
    peak=clock['peak_index_zero_based'];assert peak==51300
    publication=window(peak-1,'stats_end','pub_end','wake','log_end');assert publication['sim_after']==427.5 and len(publication['all_recorded_gc_overlaps'])==1
    postpub_gc=[g for g in gcpairs if g['start_event']['wall']<at(peak-1,'log_end_wall') and g['stop_event']['wall']>at(peak-1,'pub_end_wall')]
    assert postpub_gc==reported['gc_overlapping_previous_pub_end_to_log_end']==[]
    gen2=publication['all_recorded_gc_overlaps'][0];assert gen2['generation']==2 and gen2['thread_id']==meta['main_thread_id'] and gen2['start_tick']==gen2['stop_tick']==peak-1
    gc_report=reported['longest_gc_intervals'][0];assert gen2['start_event']==gc_report['start'] and gen2['stop_event']==gc_report['end'] and gen2['wall_seconds']==gc_report['wall'] and gen2['thread_cpu_seconds']==gc_report['thread_cpu']
    long_sleep=sorted(range(72000),key=lambda n:at(n,'wake_wall')-at(n,'begin_wall'),reverse=True)[:2]
    sleeps=[window(n,'begin','wake','begin','wake') for n in long_sleep];assert long_sleep==[21562,54562] and all(not x['all_recorded_gc_overlaps'] for x in sleeps)
    for x,y in zip(sleeps,reported['longest_phase_intervals']['begin__wake'][:2]):assert x['tick_index']==y['tick_index'] and x['wall_seconds']==y['wall'] and x['thread_cpu_seconds']==y['thread_cpu']
    assert clock['rusage']['qos']['return_code']==0 and clock['rusage']['qos']['requested_qos_class']==33 and clock['rusage']['qos']['requested_relative_priority']==0 and clock['rusage']['qos']['no_scheduling_settings_changed'] is True
    header=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/sys/qos.h');text=header.read_text();assert 'QOS_CLASS_USER_INTERACTIVE' in text and '= 0x21,' in text
    expected=diag['expected_model_meter'];service_intent=read(op/'service-intent.json');completion=read(op/'service-completion.json')
    for k in ['reservation','owner_frozen_plan','reserved','model_receipt_sha256']:assert service_intent[k]==expected[k]
    assert service_intent['source_sha256']==expected['service_sha256']==sources[str(repo/'eval/inspection_model_service.py')];assert service_intent['request_budget']==expected['generation_profile']==run['model_generation_profile']==intent['request_budget']
    actor=[json.loads(x) for x in (raw/'flight/actor.jsonl').read_bytes().splitlines()];requests=[x for x in actor if x['kind']=='public_model_request'];responses=[x for x in actor if x['kind']=='actual_model_response'];dirs=sorted(op.glob('call-*'));assert len(requests)==len(responses)==len(dirs)==30
    assert [p.name for p in dirs]==[f'call-{i:04d}' for i in range(1,31)]
    calls=[];input_sum=generated_sum=0
    for i,(folder,reqrow,resrow,meter) in enumerate(zip(dirs,requests,responses,receipt['resource_recount']['calls']),1):
        req=read(folder/'request.json');res=read(folder/'response.json');tokens=[json.loads(x) for x in (folder/'tokens.jsonl').read_bytes().splitlines()]
        assert reqrow['request_ordinal']==i and reqrow['sampler_seed']==req['seed']==(1096691202+i-1)%2**32 and reqrow['messages']==req['messages'] and canonical(req['messages'])==reqrow['messages_sha256']
        assert resrow['response']['submitted_ordinal']==i and all(resrow['response'][k]==v for k,v in res.items())
        count=req['input_tokens_submitted'];generated=res['generated_tokens_observed'];assert type(count)is int and 0<count<=8192 and res['input_tokens_submitted']==count and generated==res['emitted_tokens_journaled']==len(tokens)<=1024
        assert req['max_output_tokens']==1024 and req['temperature']==.2 and req['top_p']==.95 and req['enable_thinking'] is False and res['status']=='completed'
        assert all(t['runtime_prompt_tokens']==count and t['runtime_generation_tokens']==j and type(t['token'])is int and t['token']>=0 for j,t in enumerate(tokens,1))
        assert res['usage']=={'calls':1,'tokens':count+generated}
        hashes={n:sha(folder/n) for n in ['request.json','response.json','tokens.jsonl']};assert meter['ordinal']==i and all(meter[k]==hashes[n] for k,n in [('request_sha256','request.json'),('response_sha256','response.json'),('tokens_sha256','tokens.jsonl')])
        input_sum+=count;generated_sum+=generated;calls.append({'ordinal':i,'seed':req['seed'],'input_tokens':count,'generated_tokens':generated,'journal_rows':len(tokens),'status':res['status'],'artifact_hashes':hashes})
    usage={'calls':len(calls),'tokens':input_sum+generated_sum};assert usage=={'calls':30,'tokens':165378}==receipt['usage']==receipt['resource_recount']['usage']==completion['usage']==diag['resource_recount']['usage']==run['model_operation']['usage']==binding['known_usage'];assert input_sum==159770 and generated_sum==5608
    assert run['model_operation']['sha256']==sha(op/'service-completion.json') and receipt['resource_recount']['intent_sha256']==sha(op/'service-intent.json') and receipt['resource_recount']['completion_sha256']==sha(op/'service-completion.json')
    packet=read(Q/'reservation-packet.v1.json');purpose=json.loads(packet['purpose']);ack=read(Q/'native-reservation.v1.json');settled=read(Q/'native-settlement.v1.json');settlepacket=read(Q/'settlement-packet.v1.json')
    assert packet['usage']==expected['reserved']=={'calls':32,'tokens':180000} and ack['status']=='reserved' and ack['reservation']==expected['reservation'] and purpose['phase']=='development' and purpose['no_failed_SDK_slot_retry'] is True
    assert purpose['sampler_seed']==1096691202 and purpose['case_file_sha256']==intent['case']['sha256'] and purpose['worker_variant_source_hashes']==intent['extra_sources'] and purpose['diagnostic_role']==intent['role']
    for key in ['diagnostic_intent','fixed_source_freeze','launch_review']:assert sha(R/purpose[key]['path'])==purpose[key]['sha256']
    assert settled['status']=='known' and settled['reservation']==expected['reservation'] and not settled['budget']['unresolved_usage'] and not settled['budget']['unknown_usage'] and settled['budget']['reserved']['calls']==settled['budget']['reserved']['tokens']==0
    assert settlepacket['status']=='known' and settlepacket['usage']==usage and settlepacket['receipt']==str((Q/'usage-receipt.v1.json').relative_to(R))
    owner=read(R/'.paper/control.json');entry=owner['usage_ledger'][expected['reservation']];assert entry['status']=='known' and entry['unreserved'] is False and entry['receipt_sha256']==sha(Q/'usage-receipt.v1.json') and entry['reported_usage']==usage
    assert len(owner['usage_ledger'])==23 and all(e['status']=='known' for e in owner['usage_ledger'].values());totals={k:sum(e['reported_usage'].get(k,0) for e in owner['usage_ledger'].values()) for k in ['calls','tokens']};assert totals=={'calls':504,'tokens':3169634}=={k:settled['budget']['known_usage'][k] for k in ['calls','tokens']}
    obs=diag['supervisor_observations'];supervisor={'count':len(obs),'wall':summary([x['wall_seconds'] for x in obs]),'self_process_cpu':summary([x['process_cpu_seconds'] for x in obs]),'inventory':diag['supervisor_inventory'],'overflow':0,'cleanup_status':sup['status'],'span_seconds':diag['finished_monotonic']-diag['started_monotonic'],'child_returncode':diag['returncode'],'supervisor_own_OS_exit':'not_observed_session_lost_after_host_interrupt'}
    inputs={str(p.relative_to(repo)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in [Q/'actual-diagnostic-binding.v1.json',Q/'archive-members.v1.json',archive,Q/'phase-recount.v1.json',Q/'usage-receipt.v1.json',F/'binding.json',raw/'diagnostic.json',raw/'flight/run.json',raw/'flight/world.json',raw/'flight/clock-ticks.f64',raw/'flight/clock-profile.json',raw/'flight/clock-gc-events.f64',raw/'flight/clock-rusage.f64',raw/'flight/clock-rusage-profile.json',Q/'reservation-packet.v1.json',Q/'native-reservation.v1.json',Q/'settlement-packet.v1.json',Q/'native-settlement.v1.json']}
    result={'role':'independent raw M2/L3 telemetry recount; engineering clock invalid, no unique cause/G4','script_sha256':sha(Path(__file__)),'input_bindings':inputs,'archive':{'members':106,'all_sha_and_actual_tree_match':True},'sources':{'frozen_files_verified':111,'current_postflight_sources_verified':107,'inputs_verified':11,'embedded_source_texts_verified':23,'dependencies_verified':61,'variant_six_match':True},'clock':clock,'localization':{'peak_previous_publication':publication,'two_longest_begin_to_wake':sleeps,'gc_pub_end_to_log_end_overlap':postpub_gc,'counter_scope':'RUSAGE_SELF all physics-process threads; publication counters use wider wake-to-log_end snapshot interval'},'supervisor':supervisor,'model':{'usage':usage,'input_tokens':input_sum,'generated_tokens':generated_sum,'journal_rows':generated_sum,'calls_all_completed':True,'native_settlement':'known','native_operations_total':23,'native_cumulative_usage':totals,'capital_energy_cost_unmetered':True},'qos_enum_primary_local_header':{'path':str(header),'sha256':sha(header),'constant':'QOS_CLASS_USER_INTERACTIVE=0x21=33','requested_only':True},'limitations':['not full physical-state replay','instrumented source/caffeinate conditions differ from failed pilot','stages include marker/array/GC/OS waits; counter cooccurrence cannot uniquely assign causes','RUSAGE_SELF counter negative deltas retained','childreturn0+completed envelope does not invent supervisor own OS exit witness','no scientific policy score, SDK slot completion, G4 or main/holdout permission']}
    (out/'recount.v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');(out/'model-calls.v1.json').write_text(json.dumps(calls,indent=2)+'\n')
    print(json.dumps({'clock':{k:clock[k] for k in ['rows','max_lag','p95_lag','over_0p1','peak_index_zero_based']},'gc':{k:v for k,v in clock['gc'].items() if k not in ['all_pairs','longest5']},'publication':publication,'sleep_windows':sleeps,'rusage':clock['rusage'],'model':result['model'],'supervisor':supervisor},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
