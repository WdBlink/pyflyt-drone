"""Bounded synthetic side-effect probes; all actual RNG remains forbidden."""
from pathlib import Path
import importlib.util,json,sys,os,stat
import pytest
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO));R=REPO/'paper/research/20261104-inspection';T=R/'planning/formal-admission-candidate.v3/tests'
spec=importlib.util.spec_from_file_location('independent_writer_fixtures_v1',T/'test_formal_population_writer.py');fixture=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fixture;spec.loader.exec_module(fixture);m=fixture.module
observations=[]
def test_native_applied_namespace_moved_as_evidence_does_not_block_new_entropy(tmp_path,monkeypatch):
 f=fixture.helper.DatasetFixture(tmp_path);seen1=fixture.literal_entropy(monkeypatch,f,first_root=10000)
 old=m.write_namespace_intent(f.run,f.project,f.acceptance,'main',verification=f.verification);body=f.read(old);f.seal_intent(body,old)
 oldfolder=f.run/'datasets/main.v1';backup=f.run/'retained-main-namespace-evidence';oldfolder.rename(backup)
 assert (backup/'namespace-intent.json').is_file() # Existing known native evidence retained, never deleted.
 seen2=fixture.literal_entropy(monkeypatch,f,first_root=20000)
 new=m.write_namespace_intent(f.run,f.project,f.acceptance,'main',verification=f.verification);newbody=f.read(new)
 assert len(seen2)==4 and old['sha256']!=new['sha256']
 assert set(body['root_generation_seeds'].values())==set(range(10000,10072))
 assert set(newbody['root_generation_seeds'].values())==set(range(20000,20072))
 with pytest.raises(ValueError):f.proof(new)
 observations.append({'probe':'native-applied existing namespace retained in backup; current writer redraws new canonical pending namespace','old_intent':old,'new_intent':new,'old_root_seed_range':[10000,10071],'new_root_seed_range':[20000,20071],'second_literal_entropy_events':seen2,'known_namespace_native_findings':sum(row['decision']['finding']['id']==m.NAMESPACE_FINDINGS['main'] for row in f.owner['rounds']),'consumer_new_proof_still_rejects':True,'no_real_RNG':True})

def test_claim_directory_fsync_failure_leaves_blocking_empty_file_without_entropy(tmp_path,monkeypatch):
 f=fixture.helper.DatasetFixture(tmp_path);events=[]
 monkeypatch.setattr(m.secrets,'randbits',fixture.forbid_random);monkeypatch.setattr(m.np.random,'SeedSequence',fixture.forbid_random)
 original=m._sync_directory;once={'failed':False}
 def fail_claim(path):
  claim=f.run/'datasets/main.v1/namespace-intent.json'
  if claim.exists() and not once['failed']:
   once['failed']=True;raise OSError('injected claim directory fsync failure')
  return original(path)
 monkeypatch.setattr(m,'_sync_directory',fail_claim)
 with pytest.raises(OSError,match='injected claim'):m.write_namespace_intent(f.run,f.project,f.acceptance,'main',verification=f.verification)
 folder=f.run/'datasets/main.v1';assert (folder/'namespace-intent.json').read_bytes()==b''
 assert json.loads((folder/'namespace-failure.json').read_text())['complete_intent_available'] is False
 with pytest.raises(ValueError):m.write_namespace_intent(f.run,f.project,f.acceptance,'main',verification=f.verification)
 observations.append({'probe':'claim directory fsync error','claim_retained_empty':True,'failure_retained':True,'retry_blocked':True,'actual_RNG_calls':0})

def test_fsync_order_before_first_entropy_is_directory_only(tmp_path,monkeypatch):
 f=fixture.helper.DatasetFixture(tmp_path);fixture.literal_entropy(monkeypatch,f);events=[];orig=m.os.fsync;bits=m.secrets.randbits
 def observe(fd):events.append({'kind':'directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'regular','size':os.fstat(fd).st_size});return orig(fd)
 def before_entropy(n):
  if n==128:
   assert events and not any(x['kind']=='regular' for x in events)
   observations.append({'probe':'literal entropy boundary fsync observation','before_first_entropy':list(events),'scope':'directory fsync before entropy; empty regular intent fd is first fsynced after complete JSON write; no power-loss guarantee inferred'})
  return bits(n)
 monkeypatch.setattr(m.os,'fsync',observe);monkeypatch.setattr(m.secrets,'randbits',before_entropy)
 m.write_namespace_intent(f.run,f.project,f.acceptance,'main',verification=f.verification)
 assert any(x['kind']=='regular' and x['size']>0 for x in events)

def test_geometry_regular_file_write_error_retains_partial_and_blocks_retry(tmp_path,monkeypatch):
 f,intent,seen=fixture.geometry_fixture(tmp_path,monkeypatch);orig=m.durable_once
 def injected(path,value):
  if path.parent.name=='cases':raise OSError('injected case write failure')
  return orig(path,value)
 monkeypatch.setattr(m,'durable_once',injected)
 with pytest.raises(OSError,match='injected case'):m.write_geometry(f.run,f.project,f.acceptance,intent,verification=f.verification)
 folder=f.run/'datasets/main.v1';assert len(list((folder/'generation').glob('*.json')))==1
 assert json.loads((folder/'generation-failure.json').read_text())['completed_root_records']==0
 assert not (folder/'manifest.json').exists()
 before=list(seen)
 with pytest.raises(ValueError):m.write_geometry(f.run,f.project,f.acceptance,intent,verification=f.verification)
 assert seen==before
 observations.append({'probe':'geometry case write error','audit_retained':True,'partial_failure_retained':True,'completed_root_records':0,'retry_blocked':True})

@pytest.mark.parametrize('mode',['missing_g4_sha','phase_mismatch','positive_literal'])
def test_cli_front_door(tmp_path,monkeypatch,capsys,mode):
 f=fixture.helper.DatasetFixture(tmp_path);seen=fixture.literal_entropy(monkeypatch,f)
 monkeypatch.setattr(m,'ROOT',f.project);monkeypatch.setattr(m,'context_from_environment',lambda *a:f.verification)
 args=['writer','--run',str(f.run),'--runtime','unused','--sdk','unused','--action','namespace','--phase','main','--g4',f.acceptance['path']]
 if mode!='missing_g4_sha':args+=['--g4-sha256',f.acceptance['sha256']]
 if mode=='phase_mismatch':
  args[args.index('namespace')]='geometry';args+=['--intent','datasets/holdout.v1/namespace-intent.json','--intent-sha256','0'*64]
 monkeypatch.setattr(sys,'argv',args)
 if mode=='positive_literal':
  m.main();out=json.loads(capsys.readouterr().out);assert out['status']=='artifact_written_pending_independent_native_binding';assert len(seen)==4
 else:
  with pytest.raises(SystemExit) as e:m.main()
  assert e.value.code==2 and not (f.run/'datasets').exists() and not seen
 observations.append({'probe':'CLI '+mode,'literal_only':True,'new_native_or_independent_review':False})

def teardown_module():Path('/tmp/formal-writer-independent-probes-v1.json').write_text(json.dumps(observations,indent=2)+'\n')
