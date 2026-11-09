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
            pairs.append({'generation':int(e['generation']),'thread_id':int(e['thread_id']),
                'wall_seconds':e['wall']-start['wall'],'thread_cpu_seconds':e['thread_cpu']-start['thread_cpu'],
                'process_cpu_seconds':e['process_cpu']-start['process_cpu'],'start_tick':int(start['tick_index']),
                'stop_tick':int(e['tick_index']),'start_stage':int(start['stage_index']),
                'start_control_active':bool(start['control_active']),'collected':e['collected'],'uncollectable':e['uncollectable']})
    assert not pending
    out['gc']={'events':len(events),'pairs':len(pairs),'unmatched':0,'generations':dict(Counter(x['generation'] for x in pairs)),
        'main_thread_pairs':sum(x['thread_id']==meta['main_thread_id'] for x in pairs),
        'control_active_at_start':sum(x['start_control_active'] for x in pairs),
        'wall_intervals':summary([x['wall_seconds'] for x in pairs]),'longest5':sorted(pairs,key=lambda x:x['wall_seconds'],reverse=True)[:5]}
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
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    repo=args.repo.resolve();R=repo/'paper/research/20261104-inspection';Q=R/'qualification/clock-localization-adaptive-L2-postinventory.v1';F=Q/'freeze.v1'
    raw=repo/'logs/inspection-clock-localization-adaptive-L2-postinventory.v1';old=repo/'logs/inspection-clock-localization-adaptive-L2.v1'
    inputs={};bind=lambda p:inputs.update({str(p.relative_to(repo)):{'sha256':sha(p),'bytes':p.stat().st_size}})
    freeze=read(F/'binding.json');assert sha(F/'binding.json')=='3d7f26115c117565bd475bafe14927567d04ef4191b056ba552cd12e1ac0f931'
    for name,h in freeze['files'].items():assert sha(R/name)==h
    plan=read(F/'intent.json');sources=plan['complete_frozen_source_hashes'];assert len(sources)==107 and all(sha(Path(n))==h for n,h in sources.items())
    old63=read(R/'planning/clock-localization-adaptive-L2.v1.json')['prior_scientific_sources']['hashes']
    changes={n:{'before':h,'after':sha(Path(n))} for n,h in old63.items() if sha(Path(n))!=h}
    assert changes==plan['scientific_source_changes_from_failed_pilot_v2']
    receipt=read(Q/'usage-receipt.v1.json');assert sha(Q/'usage-receipt.v1.json')=='18ecfbaaef8251cc12ef5a7ea6d4e29926f8b3a1b650b2fef613d985ccc5b41c'
    archive=R/receipt['archive'];assert sha(archive)==receipt['archive_sha256']=='f50c9eff10bdc1be30f09d89a884a921a15c73c850229bc1efb329d56b9f304b'
    members=read(Q/'archive-members.v1.json');diag=read(raw/'diagnostic.json');op=repo/'logs/inspection-model-resource-operations.v1'/diag['expected_model_meter']['reservation']
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert len(names)==len(set(names))==len(members)==86 and set(names)==set(members)
        local_count=0
        for n in names:
            assert not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
            assert sha_bytes(z.read(n))==members[n],n
            path=raw/n[len('diagnostic/'):] if n.startswith('diagnostic/') else op/n[len('model-operation/'):] if n.startswith('model-operation/') else None
            if path is not None:
                assert sha(path)==members[n],n;local_count+=1
        assert local_count==85
        actual_files={'diagnostic/'+str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file()}|{'model-operation/'+str(p.relative_to(op)) for p in op.rglob('*') if p.is_file()}
        assert actual_files==set(names)-{'supervisor.stdout-stderr.log'}
        supervisor=json.loads(z.read('supervisor.stdout-stderr.log').splitlines()[-1])
        assert supervisor['status']=='completed_engineering_not_qualification' and supervisor['usage']==receipt['usage']
    assert sha(raw/'diagnostic.json')==receipt['diagnostic_sha256']
    for n in ['diagnostic.json','flight/run.json','flight/world.json','flight/clock-profile.json','flight/clock-ticks.f64','flight/clock-gc-events.f64','flight/clock-rusage-profile.json','flight/clock-rusage.f64','flight/actor.jsonl']:
        bind(raw/n)
    for p in [F/'binding.json',F/'intent.json',Q/'usage-receipt.v1.json',Q/'archive-members.v1.json',archive,old/'diagnostic.json',old/'flight/clock-profile.json',old/'flight/clock-ticks.f64']:bind(p)
    run=read(raw/'flight/run.json');assert run['status']=='completed' and run['requested_seconds']==600 and run['realtime'] is True
    assert run['sampler_base_seed']==2026110616 and run['injected_delay_seconds']==0 and run['ablation'] is None
    assert all(run[k] is True for k in ['sources_unchanged','dependencies_unchanged','private_case_unchanged'])
    assert len(run['sources'])==23 and set(run['sources'])==set(run['source_hashes'])
    for n,text in run['sources'].items():assert sha_bytes(text.encode())==run['source_hashes'][n]==sources[n]==sha(Path(n))
    assert len(run['dependency_hashes'])==61 and all(sha(Path(n))==h for n,h in run['dependency_hashes'].items())
    assert len(diag['extra_worker_variant_sources'])==6 and diag['extra_worker_variant_sources']==plan['worker_variant_source_hashes']
    assert sha(raw/'flight/private-case.json')==sha(R/plan['case']['path'])==plan['case']['sha256']==diag['case_file_sha256']
    assert diag['sampler_seed']==plan['sampler_seed']==2026110616
    assert diag['returncode']==0 and diag['errors']==[] and diag['worker_variant_sources_unchanged'] and diag['original_case_unchanged']
    sup=diag['supervision'];assert sup['status']=='closed' and sup['remaining_owned']==[] and sup['unclaimed_new_python_orphans']==[]
    assert diag['supervisor_inventory']=={'hot_native':5775,'hot_legacy':0,'fallback_reasons':{}} and diag['supervisor_observation_overflow']==0
    current=clock_recount(raw);previous=clock_recount(old)
    def observe(d):
        rows=d['supervisor_observations'];return {'count':len(rows),'wall':summary([x['wall_seconds'] for x in rows]),'self_process_cpu':summary([x['process_cpu_seconds'] for x in rows]),
            'span_seconds':d['finished_monotonic']-d['started_monotonic'],'overflow':d['supervisor_observation_overflow'],'usage':d.get('resource_recount',{}).get('usage')}
    observers={'postrepair':observe(diag),'preceding':observe(read(old/'diagnostic.json'))}
    expected=diag['expected_model_meter'];intent=read(op/'service-intent.json');completion=read(op/'service-completion.json')
    for k in ['reservation','owner_frozen_plan','reserved','model_receipt_sha256']:assert intent[k]==expected[k]
    assert intent['source_sha256']==expected['service_sha256']==sources[str(repo/'eval/inspection_model_service.py')]
    assert intent['request_budget']==expected['generation_profile']==run['model_generation_profile']
    assert expected['reserved']=={'calls':32,'tokens':180000}
    actor=[json.loads(x) for x in (raw/'flight/actor.jsonl').read_bytes().splitlines()]
    requests=[x for x in actor if x['kind']=='public_model_request'];responses=[x for x in actor if x['kind']=='actual_model_response']
    assert len(requests)==len(responses)==23
    dirs=sorted(p for p in op.glob('call-*'));assert [p.name for p in dirs]==[f'call-{i:04d}' for i in range(1,24)]
    calls=[];total=0;input_sum=0;generated_sum=0
    meter_calls=receipt['resource_recount']['calls'];assert len(meter_calls)==23
    for ordinal,(folder,ar,aresp,meter) in enumerate(zip(dirs,requests,responses,meter_calls),1):
        req=read(folder/'request.json');res=read(folder/'response.json');tokens=[json.loads(x) for x in (folder/'tokens.jsonl').read_bytes().splitlines()]
        assert ar['request_ordinal']==ordinal and ar['sampler_seed']==req['seed']==2026110616+ordinal-1
        assert ar['messages']==req['messages'] and canonical(req['messages'])==ar['messages_sha256']
        assert aresp['response']['submitted_ordinal']==ordinal
        for k,v in res.items():assert aresp['response'][k]==v
        count=req['input_tokens_submitted'];generated=res['generated_tokens_observed']
        assert type(count)is int and 0<count<=8192 and req['max_output_tokens']==1024
        assert req['temperature']==.2 and req['top_p']==.95 and req['enable_thinking'] is False
        assert res['input_tokens_submitted']==count and generated==res['emitted_tokens_journaled']==len(tokens)<=1024
        assert all(t['runtime_prompt_tokens']==count and t['runtime_generation_tokens']==i and type(t['token'])is int and t['token']>=0 for i,t in enumerate(tokens,1))
        input_sum+=count;generated_sum+=generated;total+=count+generated
        assert res['usage']=={'calls':1,'tokens':count+generated}
        hashes={n:sha(folder/n) for n in ['request.json','response.json','tokens.jsonl']}
        assert meter['ordinal']==ordinal and all(meter[k]==hashes[n] for k,n in [('request_sha256','request.json'),('response_sha256','response.json'),('tokens_sha256','tokens.jsonl')])
        calls.append({'ordinal':ordinal,'seed':req['seed'],'input_tokens':count,'generated_tokens':generated,'journal_rows':len(tokens),'status':res['status'],
            'messages_sha256':ar['messages_sha256'],'artifact_hashes':hashes})
    usage={'calls':len(calls),'tokens':total};assert usage=={'calls':23,'tokens':163524}==completion['usage']==receipt['usage']==diag['resource_recount']['usage']
    assert input_sum+generated_sum==total and total<=180000 and len(calls)<=32
    ack=read(Q/'native-reservation.v1.json');packet=read(Q/'reservation-packet.v1.json');purpose=json.loads(packet['purpose'])
    assert ack['status']=='reserved' and ack['reservation']==expected['reservation'] and packet['usage']==expected['reserved']
    assert purpose['worker_variant_source_hashes']==plan['worker_variant_source_hashes']
    assert purpose['sampler_seed']==2026110616 and purpose['case_file_sha256']==plan['case']['sha256']
    assert purpose['phase']=='development' and purpose['no_failed_SDK_slot_retry'] is True
    for p in [op/'service-intent.json',op/'service-completion.json',Q/'native-reservation.v1.json',Q/'reservation-packet.v1.json']:bind(p)
    output={'role':'independent offline recount; no policy score or G4 acceptance','script_sha256':sha(Path(__file__)),
        'input_bindings':inputs,'archive':{'sha256':sha(archive),'members':86,'all_member_manifest_hashes_match':True,'live_files_matching_archive':85,'remaining_archive_only_member':'supervisor.stdout-stderr.log'},
        'sources':{'freeze_files_verified':111,'current_source_count_verified':107,'embedded_run_sources_verified':23,'dependencies_bytes_verified':61,'old63_delta':changes,'variant_six_exact':True},
        'postrepair':current,'preceding_same_case_sampler':previous,'supervisor_comparison':observers,
        'supervisor_backend':diag['supervisor_inventory'],'cleanup':{k:v for k,v in sup.items() if k!='owned_births'},
        'model':{'usage':usage,'input_tokens':input_sum,'generated_tokens':generated_sum,'journal_rows':generated_sum,'statuses':dict(Counter(x['status'] for x in calls)),
            'actor_requests':len(requests),'actor_responses':len(responses),'paid_api':receipt['paid_API_used'],'capital_energy_cost_unmetered':receipt['capital_energy_cost_unmetered']},
        'limits':['No actual ledger settlement status is certified.','Raw tracking errors and wallclock traces are not full physical-state replay.','Supervisor process CPU excludes ps child CPU; cumulative observer wall is not unique causal attribution.','23 vs18 real requests and adaptive feedback differ; no strict causal or strategy-performance claim.','Negative rusage deltas retained; counter names do not uniquely label waits.','One same-case unscored engineering clock pass does not qualify all G4 or repair old failed SDK slots.']}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'recount.v1.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (args.out/'model-calls.v1.json').write_text(json.dumps(calls,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'clock':{k:current[k] for k in ['rows','max_lag','p95_lag','over_0p1']},'gc':current['gc']['pairs'],'rusage_nivcsw':current['rusage']['counter_changes_raw_unclipped']['ru_nivcsw'],
        'usage':usage,'input':input_sum,'output':generated_sum,'observer_new':observers['postrepair'],'observer_old':observers['preceding']},indent=2))

if __name__=='__main__':main()
