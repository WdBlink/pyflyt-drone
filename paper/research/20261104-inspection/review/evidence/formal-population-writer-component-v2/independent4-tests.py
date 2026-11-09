"""Original W1 and fd durability narrow recheck, all /tmp literal entropy."""
from pathlib import Path
import importlib.util,json,sys,os,stat
import pytest
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO));R=REPO/'paper/research/20261104-inspection';T=R/'planning/formal-admission-candidate.v3/tests'
s=importlib.util.spec_from_file_location('independent_writer_repair_fixture_v1',T/'test_formal_population_writer.py');x=importlib.util.module_from_spec(s);sys.modules[s.name]=x;s.loader.exec_module(x);m=x.module
obs=[]
@pytest.mark.parametrize('revoked',[False,True])
def test_original_retained_native_namespace_requires_restore_before_entropy(tmp_path,monkeypatch,revoked):
 f=x.helper.DatasetFixture(tmp_path);x.literal_entropy(monkeypatch,f,first_root=10000)
 old=m.write_namespace_intent(f.run,f.project,f.acceptance,'main',verification=f.verification);f.seal_intent(f.read(old),old)
 if revoked:f.bind('synthetic-independent-revoke',[f.catalog],revises=m.NAMESPACE_FINDINGS['main'])
 backup=f.run/'retained-main-namespace-evidence';(f.run/'datasets/main.v1').rename(backup)
 before=(backup/'namespace-intent.json').read_bytes();seen=x.literal_entropy(monkeypatch,f,first_root=20000)
 with pytest.raises(ValueError,match='already issued') as error:m.write_namespace_intent(f.run,f.project,f.acceptance,'main',verification=f.verification)
 assert seen==[] and not(f.run/'datasets/main.v1').exists()
 assert (backup/'namespace-intent.json').read_bytes()==before
 obs.append({'probe':'original W1 retained-native-namespace replay','revoked':revoked,'old_intent':old,'new_entropy_events':seen,'canonical_not_recreated':True,'retained_bytes_unchanged':True,'reason':str(error.value)})

def test_empty_fd_then_directory_fsync_both_before_literal_entropy(tmp_path,monkeypatch):
 f=x.helper.DatasetFixture(tmp_path);x.literal_entropy(monkeypatch,f);real_sync=m.os.fsync;real_bits=m.secrets.randbits;events=[]
 def sync(fd):
  z=os.fstat(fd);events.append('empty_regular_fsync' if stat.S_ISREG(z.st_mode) and z.st_size==0 else 'regular_fsync' if stat.S_ISREG(z.st_mode) else 'directory_fsync');return real_sync(fd)
 def bits(n):
  if n==128:
   assert events[-2:]==['empty_regular_fsync','directory_fsync']
   obs.append({'probe':'actual fsync call ordering before literal entropy','events_before_entropy':list(events),'no_power_loss_claim':True})
  return real_bits(n)
 monkeypatch.setattr(m.os,'fsync',sync);monkeypatch.setattr(m.secrets,'randbits',bits)
 ref=m.write_namespace_intent(f.run,f.project,f.acceptance,'main',verification=f.verification)
 assert len(f.read(ref)['root_generation_seeds'])==72 and 'regular_fsync' in events

def test_empty_fd_fsync_failure_never_touches_entropy_and_keeps_blocked_claim(tmp_path,monkeypatch):
 f=x.helper.DatasetFixture(tmp_path);monkeypatch.setattr(m.secrets,'randbits',x.forbid_random);monkeypatch.setattr(m.np.random,'SeedSequence',x.forbid_random);real_sync=m.os.fsync;once=False
 def sync(fd):
  nonlocal once
  z=os.fstat(fd)
  if not once and stat.S_ISREG(z.st_mode) and z.st_size==0:
   once=True;raise OSError('independent injected empty-fd fsync error')
  return real_sync(fd)
 monkeypatch.setattr(m.os,'fsync',sync)
 with pytest.raises(OSError,match='empty-fd fsync'):m.write_namespace_intent(f.run,f.project,f.acceptance,'main',verification=f.verification)
 folder=f.run/'datasets/main.v1';assert (folder/'namespace-intent.json').read_bytes()==b''
 failure=json.loads((folder/'namespace-failure.json').read_text());assert failure['complete_intent_available'] is False
 with pytest.raises(ValueError):m.write_namespace_intent(f.run,f.project,f.acceptance,'main',verification=f.verification)
 obs.append({'probe':'empty-fd fsync injected failure','empty_claim_retained':True,'failure':failure,'retry_blocked_before_entropy':True})
def teardown_module():Path('/tmp/formal-writer-repair-independent-probes-v1.json').write_text(json.dumps(obs,indent=2)+'\n')
