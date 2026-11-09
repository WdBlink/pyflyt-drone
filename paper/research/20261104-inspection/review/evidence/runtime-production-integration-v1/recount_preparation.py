"""Read existing prepare-only v4 bytes; no new schedule, RNG, dataset or effects."""
from pathlib import Path
import json,hashlib,zipfile,datetime
from collections import Counter
from eval.inspection_recording import CONDITION_REPEATS
def validate_episode_order(cases,order):
 rows=order["slots"];expected={(case,c,repeat) for case in cases for c,n in CONDITION_REPEATS.items() for repeat in range(n)}
 assert len(rows)==len(expected) and {(s["case_id"],s["condition"],s["repeat_ordinal"]) for s in rows}==expected
 assert all(type(s["order_index"])is int and s["order_index"]==i and type(s["repeat_ordinal"])is int for i,s in enumerate(rows))
 for i in range(len(cases)):
  part=rows[18*i:18*(i+1)];assert len({s["case_id"]for s in part})==1 and all(s["root_block_index"]==i for s in part)
 return rows
repo=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');r=repo/'paper/research/20261104-inspection';q=r/'qualification/pilot-source-preparation.v4';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();b=json.loads((q/'binding.json').read_text());assert h(q/'binding.json')=='59122e64c8dbeb812655ed1792859ca79cd8846f10a8048b91e5365f0c771317'
assert all(h(repo/n)==v for n,v in b['original_files'].items());folder=repo/'logs/inspection-eight-condition-pilot-preparation.v4';read=lambda name:json.loads((folder/name).read_text());sources=read('source-hashes.json');texts=read('source-files.json');deps=read('dependency-hashes.json');inputs=read('input-hashes.json');order=read('prepared-episode-order.json');seeds=read('sampler-seeds.json')
assert len(sources)==104 and len(deps)==61 and len(inputs)==10
assert set(sources)==set(texts) and all(hashlib.sha256(v.encode()).hexdigest()==sources[n] for n,v in texts.items())
assert all(h(Path(n))==v for group in (sources,deps,inputs) for n,v in group.items())
helper=str((repo/'eval/inspection_worker_runtime.py').resolve());assert sources[helper]=='bf4e0b38f1d7da98ccd559091b2b722c74ed0e6fdfd0088336ccaa441bdfc21c'
cases=[row['case_id'] for row in order['cases']];rows=validate_episode_order(cases,order);assert len(cases)==6 and len(rows)==108;assert order['seed']==2026110641 and seeds['master']==2026110642
assert rows[0]==b['first_slot'];assert len(seeds['by_order_index'])==90 and set(seeds['by_order_index'])=={str(s['order_index']) for s in rows if CONDITION_REPEATS[s['condition']]==3}
assert all(type(x)is int and 0<=x<2**32 for x in seeds['by_order_index'].values());counts=Counter(s['condition'] for s in rows);assert counts=={n:6*v for n,v in CONDITION_REPEATS.items()}
archive=r/b['archive']['path'];assert h(archive)==b['archive']['sha256'] and archive.stat().st_size==b['archive']['bytes']
with zipfile.ZipFile(archive) as z:assert len(z.infolist())==8 and z.testzip()is None
assert b['prepared']['original_classic_selection_sha256']=='75f57580f0f5371d91a741fd4f9573e52192d019e9730335123a7ae67897a32e' and b['prepared']['classic_best']=='B1'
assert b['prepared']['actual_model_generation_calls']==b['prepared']['actual_physical_flights']==0
assert not any(p.name=='results.json' or p.name=='terminal.json' for p in folder.rglob('*'))
diags=['inspection_diagnostic_worker.py','probe_inspection_clock.py','probe_inspection_clock_load.py','probe_inspection_adaptive_clock.py','inspection_clock_profile.py','inspection_clock_profile_rusage.py','profile_inspection_replay_gc.py'];diag_hashes={n:h(repo/'eval'/n) for n in diags}
assert all('PhysicsRuntimeWindow'not in(repo/'eval'/n).read_text() for n in diags)
result={'role':'existing preparation intent only, no actual callbacks/flight/model proof','binding_sha256':h(q/'binding.json'),'source_count':104,'dependency_count':61,'input_count':10,'source_texts_and_current_bytes_exact':True,'helper_bound':True,'case_roots':6,'slots':108,'classic_slots':18,'LLM_slots':90,'root_means':48,'condition_counts':dict(counts),'schedule_seed':2026110641,'sampler_master_seed':2026110642,'first_slot':rows[0],'sampler_exact_keyset_and_native_uint32':True,'old_B1_selection_sha256':b['prepared']['original_classic_selection_sha256'],'archive_SHA_CRC_8members_checked':True,'no_new_RNG_or_instances':True,'diagnostic_current_hashes':diag_hashes,'diagnostic_source_has_no_runtime_helper_symbol':True,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
Path('/tmp/runtime-production-integration-review-v1/preparation-recount.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='diagnostic_current_hashes'},indent=2))
