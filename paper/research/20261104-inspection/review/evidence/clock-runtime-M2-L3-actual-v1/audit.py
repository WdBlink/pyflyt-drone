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
    repo=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=repo/'paper/research/20261104-inspection';Q=R/'qualification/clock-runtime-M2-L3.v1';F=Q/'freeze.v1';raw=repo/'logs/inspection-clock-runtime-M2-L3.v1';oldraw=repo/'logs/inspection-clock-localization-M2-L3.v1';out=Path(__file__).resolve().parent
    observed=read(out/'current-postload-observation.v1.json');assert sha(out/'current-postload-observation.v1.json')=='4e94a7087b9e0c10f675a08a25c55647d64d0f71c4d9d259d416279c31a378e6'
    binding=read(Q/'actual-diagnostic-binding.v1.json');assert sha(Q/'actual-diagnostic-binding.v1.json')=='745f09383426d9d05ce07318c1249a990d6e0b64ddcd7aa3756d46ea011d552e'
    freeze=read(F/'binding.json');assert sha(F/'binding.json')=='0ed2471d31e2814e487b62a4b3fc3544130da2ec16ed32ca651f889b6bdfb3ce'
    assert len(freeze['files'])==113 and len(freeze['live_source_hashes'])==109 and len(freeze['dependencies'])==61 and len(freeze['input_hashes'])==11
    for p,h in freeze['files'].items():assert sha(F/p)==h,p
    for key in ['live_source_hashes','dependencies','input_hashes']:assert observed[key]==freeze[key]
    # Deliberate subsequent production integration is allowed after this bound observation.
    # Recheck the archived immutable bytes, never require the changed live workspace here.
    sources=freeze['live_source_hashes'];intent=read(F/'intent.json');diag=read(raw/'diagnostic.json');receipt=read(Q/'usage-receipt.v1.json');op=repo/'logs/inspection-model-resource-operations.v1'/diag['expected_model_meter']['reservation'];archive=R/binding['archive']['path'];members=read(Q/'archive-members.v1.json')
    assert sha(archive)==receipt['archive_sha256']==binding['archive']['sha256']=='75629ca2b694f8a763ea46fc3d330420c7cfcbf253e86746e27d7c39e5bf520b'
    assert sha(raw/'diagnostic.json')==binding['envelope_sha256'];assert sha(Q/'usage-receipt.v1.json')==binding['usage_receipt_sha256'];assert sha(Q/'phase-recount.v1.json')==binding['phase_recount_sha256']
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names))==len(members)==106 and set(names)==set(members)
        for n in names:
            assert not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
            assert sha_bytes(z.read(n))==members[n]
            p=raw/n[len('diagnostic/'):] if n.startswith('diagnostic/') else op/n[len('model-operation/'):]
            assert sha(p)==members[n]
        assert {'diagnostic/'+str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file()}|{'model-operation/'+str(p.relative_to(op)) for p in op.rglob('*') if p.is_file()}==set(names)
    run=read(raw/'flight/run.json');world=read(raw/'flight/world.json');assert run['status']=='completed' and run['requested_seconds']==600 and run['realtime'] is True
    assert run['policy']=='B3-M2' and run['sampler_base_seed']==1096691202 and run['injected_delay_seconds']==0 and run['ablation'] is None
    assert all(run[k] is True for k in ['sources_unchanged','dependencies_unchanged','private_case_unchanged']);assert sha(raw/'flight/world.json')==run['world_sha256'];assert len(run['sources'])==23 and set(run['sources'])==set(run['source_hashes'])
    for p,text in run['sources'].items():assert sha_bytes(text.encode())==run['source_hashes'][p]==sources[p]
    assert run['dependency_hashes']==freeze['dependencies'];assert diag['extra_worker_variant_sources']==intent['extra_sources'] and len(intent['extra_sources'])==8
    assert sha(raw/'flight/private-case.json')==intent['case']['sha256']==diag['case_file_sha256']==run['private_case_sha256']==world['private_case_file_sha256']
    assert diag['status']=='completed_engineering_not_qualification' and diag['errors']==[] and diag['returncode']==0 and diag['worker_variant_sources_unchanged'] and diag['original_case_unchanged'] and diag['no_SDK_or_pilot_slot_admission']
    sup=diag['supervision'];assert sup['root_pid']==diag['coordinator_pid'] and sup['status']=='closed' and not sup['remaining_owned'] and not sup['unclaimed_new_python_orphans'];assert diag['supervisor_inventory']=={'hot_native':5951,'hot_legacy':0,'fallback_reasons':{}} and diag['supervisor_observation_overflow']==0 and len(diag['supervisor_observations'])==5952
    clock=clock_recount(raw);previous=clock_recount(oldraw);assert clock['rows']==72000 and clock['gc']['pairs']==1 and clock['gc_overflow']==0 and clock['over_0p1']==0 and clock['max_lag']<=.1
    assert diag['max_clock_lag_seconds']==clock['max_lag'] and diag['lag_threshold_exceeded'] is False
    reported=read(Q/'phase-recount.v1.json')
    for n,h in reported['inputs'].items():assert sha(raw/'diagnostic.json' if n=='diagnostic_envelope' else raw/'flight'/n)==h
    assert reported['statistics']=={'max_lag':clock['max_lag'],'p95_lag':clock['p95_lag'],'ticks_over_0p1':clock['over_0p1']} and reported['rows']==72000 and reported['gc_pairs']==1 and reported['gc_overflow']==0 and reported['peak_tick_index']==clock['peak_index_zero_based']
    cm=read(raw/'flight/clock-profile.json');rm=read(raw/'flight/clock-rusage-profile.json');runtime=cm['runtime_policy'];assert runtime==world['clock_profile']['runtime_policy']==rm['runtime_policy']==receipt['runtime_policy']==binding['runtime_policy']
    assert runtime['errors']==[] and runtime['before']==runtime['restored']=={'values':[1],'default':False} and runtime['prior_rt']['default'] and runtime['rt_after_restore']['default']
    assert runtime['mach_timebase']=={'numer':125,'denom':3} and runtime['requested_ticks']==[200000,96000,192000,1];assert runtime['active_start']==runtime['active_end']=={'values':runtime['requested_ticks'],'default':False}
    assert runtime['gc_enabled_before'] is True and runtime['gc_enabled_at_epoch'] is False and runtime['gc_enabled_after'] is True and runtime['gc_threshold_before']==[700,10,10]
    assert cm['recorder_changes_gc_configuration'] is True and cm['gc_configuration_at_start']['enabled'] is False and cm['gc_configuration_at_start']['threshold']==[700,10,10] and cm['gc_configuration_at_start']['frozen_count']==0
    epoch=cm['anchor']['worker_monotonic_epoch'];g=clock['gc']['all_pairs'][0];assert g['generation']==2 and g['start_tick']==g['stop_tick']==-1 and not g['start_control_active'] and g['stop_event']['control_active']==0 and g['stop_event']['wall']<epoch and g['collected']==runtime['pre_epoch_collected']==32
    assert clock['gc']['control_active_at_start']==0 and runtime['post_loop_collected']==14 and 600<=runtime['window_wall_seconds']<601
    q=clock['rusage']['qos'];assert q['return_code']==0 and q['requested_qos_class']==33 and q['requested_relative_priority']==0 and 'before applying' in q['scope'] and q['no_scheduling_settings_changed']
    oldintent=read(R/'qualification/clock-localization-M2-L3.v1/freeze.v1/intent.json');oldrun=read(oldraw/'flight/run.json');assert intent['case']==oldintent['case'] and intent['sampler_seed']==oldintent['sampler_seed'] and run['model_generation_profile']==oldrun['model_generation_profile'] and run['model_receipt_sha256']==oldrun['model_receipt_sha256']
    expected=diag['expected_model_meter'];service_intent=read(op/'service-intent.json');completion=read(op/'service-completion.json')
    for k in ['reservation','owner_frozen_plan','reserved','model_receipt_sha256']:assert service_intent[k]==expected[k]
    assert service_intent['source_sha256']==expected['service_sha256']==sources[str(repo/'eval/inspection_model_service.py')];assert service_intent['request_budget']==expected['generation_profile']==run['model_generation_profile']==intent['request_budget']
    actor=[json.loads(x) for x in (raw/'flight/actor.jsonl').read_bytes().splitlines()];requests=[x for x in actor if x['kind']=='public_model_request'];responses=[x for x in actor if x['kind']=='actual_model_response'];dirs=sorted(op.glob('call-*'));assert len(requests)==len(responses)==len(dirs)==30 and [p.name for p in dirs]==[f'call-{i:04d}' for i in range(1,31)]
    calls=[];input_sum=generated_sum=0
    for i,(folder,ar,aresp,meter) in enumerate(zip(dirs,requests,responses,receipt['resource_recount']['calls']),1):
        req=read(folder/'request.json');res=read(folder/'response.json');tokens=[json.loads(x) for x in (folder/'tokens.jsonl').read_bytes().splitlines()]
        assert ar['request_ordinal']==i and ar['sampler_seed']==req['seed']==(1096691202+i-1)%2**32 and ar['messages']==req['messages'] and canonical(req['messages'])==ar['messages_sha256'];assert aresp['response']['submitted_ordinal']==i and all(aresp['response'][k]==v for k,v in res.items())
        count=req['input_tokens_submitted'];generated=res['generated_tokens_observed'];assert type(count)is int and 0<count<=8192 and res['input_tokens_submitted']==count and generated==res['emitted_tokens_journaled']==len(tokens)<=1024
        assert req['max_output_tokens']==1024 and req['temperature']==.2 and req['top_p']==.95 and req['enable_thinking'] is False and res['status']=='completed';assert all(t['runtime_prompt_tokens']==count and t['runtime_generation_tokens']==j and type(t['token'])is int and t['token']>=0 for j,t in enumerate(tokens,1));assert res['usage']=={'calls':1,'tokens':count+generated}
        hashes={n:sha(folder/n) for n in ['request.json','response.json','tokens.jsonl']};assert meter['ordinal']==i and all(meter[k]==hashes[n] for k,n in [('request_sha256','request.json'),('response_sha256','response.json'),('tokens_sha256','tokens.jsonl')]);input_sum+=count;generated_sum+=generated;calls.append({'ordinal':i,'seed':req['seed'],'input_tokens':count,'generated_tokens':generated,'journal_rows':len(tokens),'status':res['status'],'artifact_hashes':hashes})
    usage={'calls':len(calls),'tokens':input_sum+generated_sum};assert usage=={'calls':30,'tokens':165378}==receipt['usage']==receipt['resource_recount']['usage']==completion['usage']==diag['resource_recount']['usage']==run['model_operation']['usage']==binding['usage'];assert input_sum==159770 and generated_sum==5608 and run['model_operation']['sha256']==sha(op/'service-completion.json')
    assert receipt['resource_recount']['intent_sha256']==sha(op/'service-intent.json') and receipt['resource_recount']['completion_sha256']==sha(op/'service-completion.json')
    packet=read(Q/'reservation-packet.v1.json');purpose=json.loads(packet['purpose']);ack=read(Q/'native-reservation.v1.json');settled=read(Q/'native-settlement.v1.json');settlepacket=read(Q/'settlement-packet.v1.json')
    assert packet['usage']==expected['reserved']=={'calls':32,'tokens':180000} and ack['status']=='reserved' and ack['reservation']==expected['reservation'] and purpose['phase']=='development' and purpose['no_failed_SDK_slot_retry'] is True
    assert purpose['sampler_seed']==1096691202 and purpose['case_file_sha256']==intent['case']['sha256'] and purpose['worker_variant_source_hashes']==intent['extra_sources'] and purpose['diagnostic_role']==intent['role']
    for key in ['diagnostic_intent','fixed_source_freeze','launch_review']:assert sha(R/purpose[key]['path'])==purpose[key]['sha256']
    assert settled['status']=='known' and settled['reservation']==expected['reservation'] and not settled['budget']['unresolved_usage'] and not settled['budget']['unknown_usage'] and settled['budget']['reserved']['calls']==settled['budget']['reserved']['tokens']==0
    assert settlepacket['status']=='known' and settlepacket['usage']==usage and settlepacket['receipt']==str((Q/'usage-receipt.v1.json').relative_to(R))
    owner=read(R/'.paper/control.json');entry=owner['usage_ledger'][expected['reservation']];assert entry['status']=='known' and entry['unreserved'] is False and entry['receipt_sha256']==sha(Q/'usage-receipt.v1.json') and entry['reported_usage']==usage
    cumulative={k:settled['budget']['known_usage'][k] for k in ['calls','tokens']};assert cumulative=={'calls':534,'tokens':3335012}
    obs=diag['supervisor_observations'];supervisor={'count':len(obs),'wall':summary([x['wall_seconds'] for x in obs]),'self_process_cpu':summary([x['process_cpu_seconds'] for x in obs]),'inventory':diag['supervisor_inventory'],'overflow':0,'cleanup_status':sup['status'],'span_seconds':diag['finished_monotonic']-diag['started_monotonic'],'child_returncode':diag['returncode'],'supervisor_own_OS_exit':{'root_recorded_in_receipt':receipt['supervisor_exit_code'],'reviewer_direct_OS_session_observation':False}}
    assert receipt['supervisor_exit_code']==0
    inputs={str(p.relative_to(repo)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in [Q/'actual-diagnostic-binding.v1.json',Q/'archive-members.v1.json',archive,Q/'phase-recount.v1.json',Q/'usage-receipt.v1.json',F/'binding.json',raw/'diagnostic.json',raw/'flight/run.json',raw/'flight/world.json',raw/'flight/clock-ticks.f64',raw/'flight/clock-profile.json',raw/'flight/clock-gc-events.f64',raw/'flight/clock-rusage.f64',raw/'flight/clock-rusage-profile.json',Q/'reservation-packet.v1.json',Q/'native-reservation.v1.json',Q/'settlement-packet.v1.json',Q/'native-settlement.v1.json',oldraw/'flight/clock-profile.json',oldraw/'flight/clock-ticks.f64',oldraw/'flight/clock-rusage.f64']}
    result={'role':'independent actual single-load engineering recount; clock passed, not G4','script_sha256':sha(Path(__file__)),'input_bindings':inputs,'archive':{'members':106,'all_sha_and_actual_tree_match':True},'source_observation':{'sha256':sha(out/'current-postload-observation.v1.json'),'captured_at_utc':observed['captured_at_utc'],'frozen_files_verified':113,'postload_current_sources_verified':109,'inputs_verified':11,'dependencies_verified':61,'embedded_source_texts_verified':23,'variant_eight_match':True,'subsequent_source_changes_allowed_after_captured_observation':True},'runtime_policy':runtime,'clock':clock,'preceding_default_same_case_sampler':previous,'supervisor':supervisor,'model':{'usage':usage,'input_tokens':input_sum,'generated_tokens':generated_sum,'journal_rows':generated_sum,'calls_all_completed':True,'native_settlement':'known','native_operation_ordinal':24,'native_cumulative_usage_at_ack':cumulative,'capital_energy_cost_unmetered':True},'comparison_limits':['same case/model/profile/seed, but simultaneous GC+scheduling intervention, changed variant/factory/worker provenance; single load, not isolated causal AB','requested QoS33 sampled before Mach setting, not effective priority','ru_maxrss is process cumulative high-water mark, not instantaneous RSS/total host or model memory','counter negative deltas preserved and unchanging counters do not exclude events','policy readback only at start/end, not continuous guarantee','tracking error/z arrays not full physical-state replay','not fullpilot/G4/SDK slot admission']}
    (out/'recount.v1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');(out/'model-calls.v1.json').write_text(json.dumps(calls,indent=2)+'\n')
    print(json.dumps({'clock':{k:clock[k] for k in ['rows','max_lag','p95_lag','over_0p1','peak_index_zero_based']},'gc':{k:v for k,v in clock['gc'].items() if k not in ['all_pairs','longest5']},'runtime':runtime,'current_rusage':clock['rusage'],'previous_stats':{k:previous[k] for k in ['max_lag','p95_lag','over_0p1']},'previous_rusage':previous['rusage']['counter_changes_raw_unclipped'],'model':result['model'],'supervisor':supervisor},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
