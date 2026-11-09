"""Nested formal meter identity types; synthetic files only, no generation."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE=Path(__file__).resolve().parent;STAGE=HERE.parent/'source/eval';PROJECT=HERE.parents[5]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result);return result

module=load('formal_nested_meter_candidate',STAGE/'inspection_model_receipts.py')
helper=load('formal_nested_meter_original_fixture',PROJECT/'tests/test_inspection_model_receipts.py')


@pytest.mark.parametrize('field,value',[('repeat_ordinal',False),('order_index',False),('sampler_seed',70.)])
def test_model_intent_requires_the_same_typed_formal_identity(tmp_path,field,value):
    expected=helper.fixture(tmp_path)
    expected['formal_dispatch_binding']={'repeat_ordinal':0,'order_index':0,'sampler_seed':70}
    path=tmp_path/'service-intent.json';intent=json.loads(path.read_text())
    intent['formal_dispatch_binding']=dict(expected['formal_dispatch_binding']);path.write_text(json.dumps(intent))
    assert module.validate_model_operation(tmp_path,expected)['usage']=={'calls':1,'tokens':4}
    intent['formal_dispatch_binding'][field]=value;path.write_text(json.dumps(intent))
    with pytest.raises(ValueError,match='formal dispatch differs'):module.validate_model_operation(tmp_path,expected)
