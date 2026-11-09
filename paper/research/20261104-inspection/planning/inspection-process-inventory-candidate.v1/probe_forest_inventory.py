"""Idle four-owned-process microprobe; no model, simulation or science RNG."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import select
import statistics
import subprocess
import sys
import time
import traceback

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parents[4]))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('fresh result path required')
    load('eval.inspection_darwin_inventory',BASE/'source/eval/inspection_darwin_inventory.py')
    module=load('candidate_forest_cleanup',BASE/'source/eval/inspection_process_cleanup.py')
    watch=module.SubprocessTree();child=None
    result={'role':__doc__,'status':'intent_only','metrics':[],'comparisons':[],
        'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256s':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((BASE/'source/eval').glob('*.py'))}}
    try:
        code="import subprocess,sys,time,json; p=[subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']) for _ in range(3)]; print(json.dumps([x.pid for x in p]),flush=True); time.sleep(30)"
        child=subprocess.Popen([sys.executable,'-u','-c',code],stdout=subprocess.PIPE,text=True)
        watch.attach(child);assert select.select([child.stdout],[],[],5)[0]
        expected=set(json.loads(child.stdout.readline()))|{child.pid};watch.observe();assert set(watch.owned)==expected
        for iteration in range(20):
            for kind,fn in [('legacy_full',module._inventory),('native_forest',lambda:module._observed_inventory(watch.owned))]:
                cpu=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
                start=time.monotonic();rows=fn();after=resource.getrusage(resource.RUSAGE_SELF);afterchildren=resource.getrusage(resource.RUSAGE_CHILDREN)
                result['metrics'].append({'kind':kind,'iteration':iteration,'wall':time.monotonic()-start,
                    'self_cpu':after.ru_utime+after.ru_stime-cpu.ru_utime-cpu.ru_stime,
                    'child_cpu':afterchildren.ru_utime+afterchildren.ru_stime-children.ru_utime-children.ru_stime,
                    'returned_rows':len(rows)})
                if kind=='legacy_full':previous=rows
                else:
                    fields=('parent','group','uid','born')
                    check={'iteration':iteration,'complete_expected_forest':set(rows)==expected,
                        'identity_fields_equal':all(previous[p][k]==rows[p][k] for p in expected for k in fields),
                        'python_detection_equal':all(('python' in previous[p]['command'].lower())==('python' in rows[p]['command'].lower()) for p in expected)}
                    result['comparisons'].append(check)
                    assert all(check[k] for k in ('complete_expected_forest','identity_fields_equal','python_detection_equal'))
        result['status']='completed_microprobe_not_clock_qualification'
    except BaseException:result.update(status='failed_microprobe',error=traceback.format_exc())
    finally:
        if child is not None:
            try:
                result['cleanup']=watch.close()
                if result['cleanup']['status']!='closed':result['status']='cleanup_unknown'
            except BaseException:result.update(status='cleanup_unknown',cleanup_error=traceback.format_exc())
            child.stdout.close()
        result['backend_counts']=module.inventory_diagnostics()
        result['summary']={kind:{key:(statistics.median([x[key] for x in result['metrics'] if x['kind']==kind])
            if any(x['kind']==kind for x in result['metrics']) else None)
            for key in ('wall','self_cpu','child_cpu','returned_rows')} for kind in ('legacy_full','native_forest')}
        with args.out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':result['status'],'summary':result['summary'],'backend_counts':result['backend_counts']}))
    return 0 if result['status']=='completed_microprobe_not_clock_qualification' else 1


if __name__=='__main__':raise SystemExit(main())
