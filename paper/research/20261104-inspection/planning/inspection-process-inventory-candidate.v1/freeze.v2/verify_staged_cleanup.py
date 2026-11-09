"""Run existing cleanup/receipt tests against staged code; no flight/MLX."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import pytest
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[4]
sys.path.insert(0,str(ROOT))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module
load('eval.inspection_darwin_inventory',BASE/'source/eval/inspection_darwin_inventory.py')
module=load('eval.inspection_process_cleanup',BASE/'source/eval/inspection_process_cleanup.py')
import eval
eval.inspection_process_cleanup=module
if __name__=='__main__':
    raise SystemExit(pytest.main(['-q','-p','no:cacheprovider',str(ROOT/'tests/test_inspection_process_cleanup.py'),str(ROOT/'tests/test_inspection_llm_episode_receipts.py'),str(BASE/'tests/test_native_inventory.py'),'--basetemp='+tempfile.mkdtemp(prefix='inspection-native-compat-')]))
