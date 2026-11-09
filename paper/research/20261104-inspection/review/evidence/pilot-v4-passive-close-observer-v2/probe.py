from pathlib import Path
from types import SimpleNamespace
import sys,importlib.util,tempfile,json,hashlib,os,copy
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-030.v1/observer-freeze.v2';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'observe_cleanup.py')=='83ffb02d0582c243e33421ba0f831d754010c1cb99c70b3022b5a743d96c646c'
assert sha(Q/'binding.json')=='f2e539dcd6a5bd6ac369e03527962c2a7f5519efb184a9fef1dac8a966c4e55f'
assert sha(REPO/'eval/inspection_process_cleanup.py')=='ce210650067d2b4e6f49c47845dea36e034d2c7c2e1187899023ddf22f792a9d'
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent_passive_close_v2',Q/'observe_cleanup.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.cleanup;orig_close=c.SubprocessTree.close;orig_inventory=c._inventory;orig_observed=c._observed_inventory;orig_kill=c.os.kill
context={'case_id':'dev-L3-01','condition':'B4-M1','repeat_ordinal':1,'plan_sha256':'a'*64,'reservation_prefix':'dev-all8-pilot-v4-030'}
row=lambda parent:{'parent':parent,'group':10,'uid':os.getuid(),'status':'R','born':'Sat Nov 7 08:23:13 2026','command':'/opt/Python'}
table={99:row(1)};before=copy.deepcopy(table);calls=[]
def inventory():calls.append(1);return table
def tree():
 t=c.SubprocessTree.__new__(c.SubprocessTree);t.baseline={8:row(1)};t.owned={10:row(123)};t.process=SimpleNamespace(pid=10,wait=lambda timeout:0,poll=lambda:0);return t
def no_signal(*args):raise AssertionError('no signal may be sent')
def reject_check(observer,label):
 try:observer.check()
 except RuntimeError as exc:result[label]={'rejected':True,'reason':str(exc)}
 else:raise AssertionError(label+' incorrectly accepted')
result={'source_sha256':sha(Q/'observe_cleanup.py'),'scope':'same real original close/synthetic inventory/fake wait as v1; real isolated IO; no worker/model/native owner or signal'}
try:
 c._inventory=inventory;c._observed_inventory=lambda owned:c._inventory();c.os.kill=no_signal
 expected=orig_close(tree());expected_count=len(calls);calls.clear()
 with tempfile.TemporaryDirectory(prefix='passive-close-v2-') as tmp:
  base=Path(tmp).resolve();m.PROJECT=base/'project';m.PROJECT.mkdir()
  for kind in ['delete','status','external_symlink','hardlink']:
   m.OUTPUT=m.PROJECT/'logs'/kind
   calls.clear()
   with m.CloseObserver(context) as observer:
    write=observer._write;restored=[]
    def checked_write(*args):
     assert c._inventory is inventory;restored.append(True);return write(*args)
    observer._write=checked_write
    actual=c.SubprocessTree.close(tree());assert actual==expected and len(calls)==expected_count and table==before
    ref=observer.check()[0];path=m.PROJECT/ref['path'];raw=path.read_bytes();saved=json.loads(raw)
    assert saved['global_scans'][-1]['rows']['99']==before[99] and saved['baseline_python_rows']=={'8':row(1)} and restored
    assert hashlib.sha256(raw).hexdigest()==ref['sha256']
    # Ordinary returned dictionaries are copies, not writable baseline refs.
    changed_ref=observer.check();changed_ref[0]['sha256']='0'*64
    assert observer.check()[0]['sha256']==ref['sha256']
    if kind=='delete':path.unlink()
    elif kind=='status':
     changed=json.loads(raw);changed['original_result']['status']='closed';path.write_text(json.dumps(changed))
    elif kind=='external_symlink':
     external=base/'outside-project.json';external.write_bytes(raw);path.unlink();path.symlink_to(external)
    elif kind=='hardlink':os.link(path,base/'outside-hardlink.json')
    reject_check(observer,kind+'_sidecar')
    assert actual==expected
   assert c.SubprocessTree.close is orig_close and c._inventory is inventory
  result['normal_original_close']={'same_unknown_result':True,'original_inventory_calls':expected_count,'observed_inventory_calls_per_close':expected_count,'inventory_restored_before_write':True,'snapshot_values_preserved':True,'fixed_written_bytes_sha_matches':True,'returned_ref_mutation_does_not_rebind':True}
  m.OUTPUT=m.PROJECT/'logs/second-close'
  with m.CloseObserver(context) as observer:
   assert c.SubprocessTree.close(tree())==expected and len(observer.check())==1
   observer._write=lambda *args:None
   assert c.SubprocessTree.close(tree())==expected
   reject_check(observer,'missing_second_record')
  m.OUTPUT=m.PROJECT/'logs/hash-window'
  with m.CloseObserver(context) as observer:
   assert c.SubprocessTree.close(tree())==expected
   ref=observer.check()[0];payload=(m.PROJECT/ref['path']).read_bytes();original_sha=m.hashlib.sha256;external=base/'moved-output';moved=[]
   def window_sha(data):
    value=original_sha(data)
    if data==payload and not moved:
     m.OUTPUT.rename(external);m.OUTPUT.symlink_to(external,target_is_directory=True);moved.append(True)
    return value
   m.hashlib.sha256=window_sha
   try:reject_check(observer,'parent_alias_inserted_during_hash')
   finally:
    m.hashlib.sha256=original_sha
    if moved:m.OUTPUT.unlink();external.rename(m.OUTPUT)
   assert moved==[True]
  m.OUTPUT=m.PROJECT/'logs/output-is-file';m.OUTPUT.write_text('not a directory')
  with m.CloseObserver(context) as observer:
   assert c.SubprocessTree.close(tree())==expected
   reject_check(observer,'real_IO_failure')
  m.OUTPUT=m.PROJECT/'logs/exception-sidecars';error=OSError('original inventory failure')
  def fail_inventory():raise error
  c._inventory=fail_inventory
  with m.CloseObserver(context) as observer:
   try:c.SubprocessTree.close(tree())
   except OSError as caught:assert caught is error
   else:raise AssertionError('original exception lost')
   assert c._inventory is fail_inventory
   saved=json.loads((m.PROJECT/observer.check()[0]['path']).read_bytes());assert saved['original_exception']=={'type':'OSError'}
   result['original_exception_identity_and_restore_preserved']=True
finally:
 c.SubprocessTree.close=orig_close;c._inventory=orig_inventory;c._observed_inventory=orig_observed;c.os.kill=orig_kill
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
