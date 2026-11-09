"""Actual synthetic subprocess/cache faults, never model or physical flights."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest

HERE=Path(__file__).resolve().parent;STAGE=HERE.parent/'source/eval';PROJECT=HERE.parents[5]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result

module=load('readonly_llm_coordinator_candidate',STAGE/'run_inspection_llm_episode.py')
validator=load('readonly_llm_validator_candidate',STAGE/'inspection_llm_episode_receipts.py')
module.validate_llm_episode=validator.validate_llm_episode
original=load('original_synthetic_llm_receipt_helpers',PROJECT/'tests/test_inspection_llm_episode_receipts.py')
original.module=module


def snapshot(directory):
    return {str(path.relative_to(directory)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob('*') if path.is_file()}


def test_read_only_cached_validation_does_not_create_or_change_episode_files(tmp_path,monkeypatch):
    plan,directory,counter,effects,execute=original.fixture(tmp_path,monkeypatch)
    execute();before=snapshot(directory);count=counter.read_text();old_effects=list(effects)
    def forbidden(*args,**kwargs):raise AssertionError('read-only mode cannot produce owner or process effects')
    value,receipt=module.execute_development_llm_episode(directory,plan,forbidden,
        run=tmp_path/'owner',reservation_prefix='fixture-M1',call_owner=forbidden,read_only=True)
    assert value==0 and receipt['cache_reused'] is True
    assert snapshot(directory)==before and counter.read_text()==count and effects==old_effects


def test_read_only_validation_does_not_recreate_a_missing_lock(tmp_path,monkeypatch):
    plan,directory,_,_,execute=original.fixture(tmp_path,monkeypatch);execute();(directory/'.lock').unlink()
    before=snapshot(directory)
    def forbidden(*args,**kwargs):raise AssertionError('no effects')
    with pytest.raises(module.UnresolvedEpisode,match='existing plan and lock'):
        module.execute_development_llm_episode(directory,plan,forbidden,run=tmp_path/'owner',
            reservation_prefix='fixture-M1',call_owner=forbidden,read_only=True)
    assert snapshot(directory)==before and not (directory/'.lock').exists()


@pytest.mark.parametrize('extra',['attempt-01','attempt-02','attempt-wrong'])
def test_cached_completion_cannot_ignore_an_additional_or_malformed_attempt(tmp_path,monkeypatch,extra):
    plan,directory,counter,_,execute=original.fixture(tmp_path,monkeypatch);execute()
    (directory/extra).mkdir()
    (directory/extra/'terminal.json').write_text(json.dumps({'status':'completed','Y':.9,
        'role':'intentional fake later attempt, not a genuine receipt'}))
    with pytest.raises(module.UnresolvedEpisode,match='finite receipt|last and unique'):
        execute()
    assert counter.read_text()=='1'
