from pathlib import Path
from types import SimpleNamespace
import sys,importlib.util,tempfile,json,hashlib,os,copy
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'recovery/pilot-v4-cleanup-030.v1/observer-freeze.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Q/'observe_cleanup.py')=='0a08e7e33d35dd504c6b4bbfc1d7bb5111bdafaeb2e8e0fa1c45d0722f8909df'
assert sha(Q/'binding.json')=='1cc05ab4e33989bea22109dd56318f2007668cd46f64905059b258edaeb3201e'
sys.path.insert(0,str(REPO));spec=importlib.util.spec_from_file_location('independent_passive_close',Q/'observe_cleanup.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c=m.cleanup;orig_close=c.SubprocessTree.close;orig_inventory=c._inventory;orig_observed=c._observed_inventory;orig_kill=c.os.kill
context={'case_id':'dev-L3-01','condition':'B4-M1','repeat_ordinal':1,'plan_sha256':'a'*64,'reservation_prefix':'dev-all8-pilot-v4-030'}
row=lambda parent:{'parent':parent,'group':10,'uid':os.getuid(),'status':'R','born':'Sat Nov 7 08:23:13 2026','command':'/opt/Python'}
table={99:row(1)};before=copy.deepcopy(table);calls=[]
def inventory():calls.append(1);return table
def tree():
 t=c.SubprocessTree.__new__(c.SubprocessTree);t.baseline={8:row(1)};t.owned={10:row(123)};t.process=SimpleNamespace(pid=10,wait=lambda timeout:0,poll=lambda:0);return t
def no_signal(*args):raise AssertionError('no signal may be sent')
result={'source_sha256':sha(Q/'observe_cleanup.py'),'scope':'real original close with synthetic inventory/fake wait; real isolated sidecar IO; no worker/model/native owner or signal'}
try:
 c._inventory=inventory;c._observed_inventory=lambda owned:c._inventory();c.os.kill=no_signal
 expected=orig_close(tree());expected_count=len(calls);calls.clear()
 with tempfile.TemporaryDirectory(prefix='passive-close-') as tmp:
  base=Path(tmp).resolve();m.PROJECT=base/'project';m.PROJECT.mkdir();m.OUTPUT=m.PROJECT/'logs/pilot/cleanup-observations'
  with m.CloseObserver(context) as observer:
   write=observer._write;restored=[]
   def checked_write(*args):
    assert c._inventory is inventory;restored.append(True);return write(*args)
   observer._write=checked_write
   actual=c.SubprocessTree.close(tree());assert actual==expected and actual['status']=='unknown' and actual['unclaimed_new_python_orphans']==[99]
   assert len(calls)==expected_count and table==before
   records=observer.check();path=m.PROJECT/records[0]['path'];raw=path.read_bytes();saved=json.loads(raw)
   assert saved['global_scans'][-1]['rows']['99']==before[99]
   assert saved['baseline_python_rows']=={'8':row(1)} and restored
   result['normal_real_original_close']={'same_unknown_result':True,'original_inventory_calls':expected_count,'observed_inventory_calls':len(calls),'snapshot_rows_unchanged':True,'inventory_restored_before_write':True,'check_record_count':len(records)}
   path.unlink();accepted=observer.check();assert len(accepted)==1
   result['deleted_sidecar_incorrectly_accepted']={'accepted':True,'exists_at_check':path.exists(),'returned_record_sha256':accepted[0]['sha256']}
   changed=json.loads(raw);changed['original_result']['status']='closed';path.write_text(json.dumps(changed))
   accepted=observer.check();assert sha(path)!=accepted[0]['sha256']
   result['changed_sidecar_incorrectly_accepted']={'accepted':True,'expected_sha256':accepted[0]['sha256'],'actual_sha256':sha(path),'mutated_status':changed['original_result']['status']}
   path.unlink();external=base/'outside-project.json';external.write_bytes(raw);path.symlink_to(external)
   accepted=observer.check();assert path.is_symlink()
   result['symlink_sidecar_incorrectly_accepted']={'accepted':True,'symlink':True,'target_outside_project':True,'sha_still_matches':sha(path)==accepted[0]['sha256']}
   path.unlink();path.write_bytes(raw)
   result['restored_normal_check_passes']=len(observer.check())==1
  assert c.SubprocessTree.close is orig_close and c._inventory is inventory
  # Actual IO failure: output is a regular file rather than a directory.
  m.OUTPUT=base/'project/logs/output-is-file';m.OUTPUT.write_text('not a directory')
  with m.CloseObserver(context) as observer:
   actual=c.SubprocessTree.close(tree());assert actual==expected
   try:observer.check()
   except RuntimeError:result['real_IO_failure_preserves_result_and_blocks_check']=True
   else:raise AssertionError('IO failure accepted')
  # Original inventory exception object must survive observer metadata/finally.
  m.OUTPUT=base/'project/logs/exception-sidecars';error=OSError('original inventory failure')
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
