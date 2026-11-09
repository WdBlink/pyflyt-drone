"""Read-only original flight/request/journal recount. No forward or physics."""
import base64
import collections
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import zlib

import numpy as np

ROOT = Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
R = ROOT/'paper/research/20261104-inspection'
HERE = Path(__file__).resolve().parent
B = R/'qualification/local-model-generation/M1-B3-L3-00-diagnostic.v1'
P = 'logs/inspection-real-model-dev-M1B3-L3-00.v1/'
O = 'logs/inspection-model-resource-operations.v1/dev-model-realtime-M1-B3-L3-00-v1-20261106/'
sha = lambda b: hashlib.sha256(b).hexdigest()
canon = lambda v: json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
errors = []
with zipfile.ZipFile(B/'complete-operation.zip') as z:
    blob = {n:z.read(n) for n in z.namelist()}
    manifest = json.loads(blob['archive-manifest.json'])
    if len(blob)!=len(z.namelist()) or set(blob)!=set(manifest)|{'archive-manifest.json'}:
        errors.append('archive set/duplicate mismatch')
    for name,expected in manifest.items():
        if sha(blob[name])!=expected: errors.append('archive hash:'+name)
    run = json.loads(blob[P+'run.json'])
    world = json.loads(blob[P+'world.json'])
    receipt = json.loads((B/'usage-receipt.json').read_text())
    actor = [json.loads(x) for x in blob[P+'actor.jsonl'].decode().splitlines()]
    requests = [x for x in actor if x['kind']=='public_model_request']
    responses = [x for x in actor if x['kind']=='actual_model_response']
    decoded = [x for x in actor if x['kind']=='decoded_reply']
    source_hashes = {k:sha(v.encode()) for k,v in run['sources'].items()}
    if source_hashes!=run['source_hashes']: errors.append('source text/hash mismatch')
    if sha(blob[P+'world.json'])!=run['world_sha256']: errors.append('world hash')
    if sha(blob[P+'private-case.json'])!=run['private_case_sha256'] or run['private_case_sha256']!=world['private_case_file_sha256']:
        errors.append('private case binding')
    count=tokens=generated=journaled=0
    timing=[]
    for i,(a,b,c) in enumerate(zip(requests,responses,decoded),1):
        q=json.loads(blob[O+f'call-{i:04d}/request.json'])
        p=json.loads(blob[O+f'call-{i:04d}/response.json'])
        rows=[json.loads(s) for s in blob[O+f'call-{i:04d}/tokens.jsonl'].decode().splitlines()]
        if a['request_ordinal']!=i or a['sampler_seed']!=(run['sampler_base_seed']+i-1)%2**32:
            errors.append(f'actor ordinal/seed:{i}')
        if q['messages']!=a['messages'] or q['seed']!=a['sampler_seed'] or sha(canon(a['messages']))!=a['messages_sha256']:
            errors.append(f'actor messages/seed/hash:{i}')
        # Pipe/client adds two parent-only timing/ordinal fields; raw service
        # response bytes otherwise remain identical to its durable response.
        transport={k:v for k,v in b['response'].items() if k not in ('parent_end_to_end_wall_seconds','submitted_ordinal')}
        if transport!=p or b['response']['submitted_ordinal']!=i:
            errors.append(f'actor/durable response:{i}')
        if c['audit']['request']['messages']!=a['messages']:
            errors.append(f'decoder request binding:{i}')
        n=q['input_tokens_submitted'];g=p['generated_tokens_observed']
        if (p['status']!='completed' or p['input_tokens_submitted']!=n
            or p['usage']!={'calls':1,'tokens':n+g} or len(rows)!=g
            or p['emitted_tokens_journaled']!=g
            or any(x['runtime_prompt_tokens']!=n or x['runtime_generation_tokens']!=k for k,x in enumerate(rows,1))):
            errors.append(f'token accounting:{i}')
        if ''.join(x['text'] for x in rows)!=p['text']:
            errors.append(f'journal text reconstruction:{i}')
        saved=receipt['resource_recount']['calls'][i-1]
        for label,field in [('request','request_sha256'),('response','response_sha256'),('tokens','tokens_sha256')]:
            suffix='tokens.jsonl' if label=='tokens' else label+'.json'
            if sha(blob[O+f'call-{i:04d}/'+suffix])!=saved[field]:errors.append(f'receipt {label} hash:{i}')
        before,after=b['worker_before_request'],b['worker_after_response']
        timing.append({'ordinal':i,'sim_advance_seconds':after[0]-before[0],
            'xy_displacement_m':float(np.linalg.norm(np.asarray(after[1:3])-before[1:3])),
            'service_wall_seconds':p['wall_seconds'],
            'parent_wall_seconds':b['response']['parent_end_to_end_wall_seconds']})
        count+=1;tokens+=n+g;generated+=g;journaled+=len(rows)
    if (len(requests),len(responses),len(decoded),count)!=(26,26,26,26):errors.append('request/call count')
    completion=json.loads(blob[O+'service-completion.json'])
    if completion['usage']!={'calls':count,'tokens':tokens} or receipt['usage']!=completion['usage']:
        errors.append('completion/usage receipt total')
    raw=base64.b64decode(world['execution_tracking_trace']['data'])
    raw=zlib.decompress(raw)
    if sha(raw)!=world['execution_tracking_trace']['raw_sha256']: errors.append('raw240 SHA')
    a=np.frombuffer(raw,dtype='<f8').reshape(-1,17)
    if len(a)!=144000 or not np.isfinite(a).all() or not np.allclose(a[:,0],np.arange(1,144001)/240,atol=1e-9,rtol=0):
        errors.append('raw240 time grid')
    xy=a[:,[1,5,9,13]];height=a[:,[2,6,10,14]];low=a[:,[3,7,11,15]];high=a[:,[4,8,12,16]]
    xybad=int(np.any(xy>10+1e-9,axis=1).sum());zbad=int(np.any((height<low-1e-9)|(height>high+1e-9),axis=1).sum())
    progress=[json.loads(x) for x in blob[P+'progress.jsonl'].decode().splitlines()]
    result={'archive_entries_verified':len(manifest),'source_texts_verified':len(source_hashes),
        'source_hashes':{Path(p).name:h for p,h in source_hashes.items()},
        'requests':len(requests),'durable_calls':count,'usage':{'calls':count,'tokens':tokens},
        'generated_tokens':generated,'journaled_tokens':journaled,
        'input_token_total':tokens-generated,
        'original_decoder_statuses':dict(collections.Counter(x['audit']['status'] for x in decoded)),
        'commands_enqueued_not_accepted':sum(len(x['commands_enqueued_not_accepted']) for x in decoded),
        'sim_advance_range_s':[min(x['sim_advance_seconds'] for x in timing),max(x['sim_advance_seconds'] for x in timing)],
        'service_wall_range_s':[min(x['service_wall_seconds'] for x in timing),max(x['service_wall_seconds'] for x in timing)],
        'all_26_requests_advanced_sim_and_position':all(x['sim_advance_seconds']>0 and x['xy_displacement_m']>0 for x in timing),
        'raw240':{'rows':len(a),'max_XY':float(xy.max()),'XY_breach_steps':xybad,'Z_breach_steps':zbad,
            'time_start':float(a[0,0]),'time_end':float(a[-1,0])},
        'clock':{k:world[k] for k in ('simulated_seconds','wall_seconds','max_clock_lag_seconds','clock_lag_p95_seconds','clock_lag_over_0p1_seconds_ticks')},
        'progress_rows':len(progress),'progress_max_wall_minus_sim':max(x['wall_elapsed']-x['t'] for x in progress),
        'score':{k:v for k,v in world['score'].items() if k not in ('static_entities','dynamic_entities')},
        'errors':errors,'call_timing':timing}
(HERE/'archive-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','call_timing')},ensure_ascii=False,indent=2))
