"""Fixed development input replay/roundtrip only. No physics/model/main draw."""
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO))
from eval.inspection_population import load_development_case, generate_development_population_case, PopulationCase
from eval.prepare_inspection_development_population import STRATA, configuration_digest, prior_identities
BASE=Path('/tmp/inspection-development-population-review-v1');RUN=REPO/'paper/research/20261104-inspection'
fresh=RUN/'qualification/dev-population.v1';prior=RUN/'qualification/dev-cases.v2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raw_config(v):
    v=json.loads(json.dumps(v))
    for k in ('schema','case_id','lineage_group','split','seed','domain'):v.pop(k,None)
    for e in v['entities']:e.pop('key')
    v['entities'].sort(key=lambda e:json.dumps(e,sort_keys=True))
    return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
assert not (fresh/'failure.json').exists()
m=read(fresh/'manifest.json');intent=read(fresh/'namespace-intent.json');pm=read(prior/'manifest.json')
assert len(m['records'])==5 and m['namespace_intent_sha256']==sha(fresh/'namespace-intent.json')
assert np.__version__=='1.26.4'
children=np.random.SeedSequence(intent['master_seed']).spawn(5)
roots=[int.from_bytes(c.generate_state(4).tobytes(),'little') for c in children]
checks={'final_manifest_five_dev_only':m['split']=='dev' and m['first_pilot'] is False and m['main_and_holdout_instantiated'] is False,
    'master_roots_recomputed':roots==intent['root_generation_seeds'],'strata_exact':intent['strata']==[list(x) for x in STRATA],
    'sources_unchanged':all(sha(p)==h for p,h in intent['source_hashes'].items()),
    'prior_manifest_exact':all(sha(x['manifest'])==x['sha256'] for x in intent['prior_development_manifests'])}
seeds,semantics,configs,bindings=prior_identities([prior])
old_cases=[load_development_case(read(prior/r['file'])) for r in pm['records']]
old_pilot=sorted(r['case_id'] for r in pm['records'] if r['first_pilot'])
checks['original_six_pilot_unchanged']=old_pilot==[f'dev-{l}-{i:02d}' for l in ('L1','L2','L3') for i in (0,1)]
checks['old24_reader_roundtrip']=len(old_cases)==24 and all(load_development_case(c.as_dict())==c for c in old_cases)
old_seed_set=seeds|{pm['master_seed']};all_new_seeds=[intent['master_seed'],*roots];rows=[];file_bindings={}
started=time.monotonic()
for index,row in enumerate(m['records']):
    p=fresh/row['file'];raw=read(p);case=load_development_case(raw);audit=read(fresh/row['generation_audit'])
    sensor=int.from_bytes(np.random.SeedSequence(roots[index]).spawn(5)[4].generate_state(4).tobytes(),'little')
    replica,ra=generate_development_population_case(case.case_id,case.domain,case.level,roots[index])
    cfg=configuration_digest(case)
    renamed=replace(case,case_id=case.case_id+'-clone',lineage_group=case.case_id+'-clone',seed=case.seed+1,
        entities=tuple(replace(e,key='renamed-'+e.key) for e in reversed(case.entities)))
    rc={'file_sha':sha(p)==row['file_sha256'],'case_digest':case.digest()==row['case_sha256'],
        'new_schema_roundtrip':isinstance(case,PopulationCase) and load_development_case(case.as_dict())==case,
        'sensor_child_stream':sensor==case.seed==audit['sensor_seed']==row['sensor_seed'],
        'root_stream':roots[index]==row['root_generation_seed']==audit['root_generation_seed'],
        'fixed_seed_joint_replay_case_and_every_attempt':replica==case and ra==audit,
        'audit_sha':sha(fresh/row['generation_audit'])==row['generation_audit_sha256'],
        'configuration_independent_hash':cfg==row['configuration_sha256']==raw_config(raw),
        'configuration_ignores_identity_seed_handles_order':cfg==configuration_digest(renamed),
        'metadata_stratum':(row['domain'],row['level'])==STRATA[index] and row['case_id']==case.case_id==case.lineage_group,
        'no_prior_seed_semantic_config_collision':case.seed not in old_seed_set and case.digest() not in semantics and cfg not in configs,
        'finite_joint_attempts':len(audit['attempts'])<=1000 and [x['candidate_index'] for x in audit['attempts']]==list(range(len(audit['attempts'])))
            and audit['attempts'][-1]['status']=='accepted' and all(x['status']=='rejected' for x in audit['attempts'][:-1])}
    all_new_seeds.append(sensor);semantics.add(case.digest());configs.add(cfg)
    rows.append({'case_id':case.case_id,'domain':case.domain,'level':case.level,'entities':len(case.entities),
        'candidate_attempts':len(audit['attempts']),'root_seed':roots[index],'sensor_seed':sensor,'checks':rc,'passed':all(rc.values())})
    file_bindings[str(p)]=sha(p);file_bindings[str(fresh/row['generation_audit'])]=sha(fresh/row['generation_audit'])
checks['all11_new_seeds_unique_no_prior_overlap']=len(set(all_new_seeds))==11 and not set(all_new_seeds)&old_seed_set
checks['new_cases_all_pass']=all(x['passed'] for x in rows)
old_source=RUN/'qualification/source/model-orchestrator-v2/inspection_population.py'
old_ast=ast.parse(old_source.read_text());new_ast=ast.parse((REPO/'eval/inspection_population.py').read_text())
def fn(tree,name):return ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)
checks['scientific_generator_and_validator_ast_unchanged']=all(fn(old_ast,k)==fn(new_ast,k) for k in ('generate_development_population_case','validate_population_case'))
faults=[]
sample=load_development_case(read(fresh/m['records'][2]['file']))
for name,changes in [('main',{'split':'main'}),('holdout',{'case_id':'holdout-X'}),('extra',{'unexpected':1}),('bool_seed',{'seed':True}),('nan_wind',{'wind_xy':[float('nan'),0]})]:
    try:load_development_case({**sample.as_dict(),**changes});faults.append({'name':name,'rejected':False})
    except (ValueError,TypeError):faults.append({'name':name,'rejected':True})
checks['negative_reader_boundaries']=all(x['rejected'] for x in faults)
for p in [fresh/'manifest.json',fresh/'namespace-intent.json',prior/'manifest.json',old_source]:file_bindings[str(p)]=sha(p)
result={'role':'CPU replay of fixed five dev inputs, not a new namespace/sample/physics or G4','numpy_version':np.__version__,
    'namespace_manifest_sha256':sha(fresh/'manifest.json'),'checks':checks,'rows':rows,'faults':faults,'passed':all(checks.values()),
    'cpu_seconds':time.monotonic()-started,'original_pilot_ids':old_pilot,'input_sha256':file_bindings}
(BASE/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','rows')},indent=2))
for row in rows:print(json.dumps(row))
