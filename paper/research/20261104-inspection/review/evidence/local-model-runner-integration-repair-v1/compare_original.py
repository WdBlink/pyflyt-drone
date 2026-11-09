"""Run the SAME strengthened fake fixture against the frozen original runner."""
import importlib.util
import json
from pathlib import Path

BASE=Path('/tmp/inspection-llm-integration-repair-review-v1')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
p=load('repair_probe',BASE/'probe.py')
old=load('frozen_original_runner',Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/review/evidence/local-model-runner-integration-v1/source/eval/run_inspection_llm_development.py'))
p.runner=old
rows=[p.main_probe(mode) for mode in ('valid_fake','private_deleted','source_changed','world_missing','bad_completion')]
(BASE/'original-v1-same-fixture.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(r['mode'],r['status'],r['return_code'],r['raised'])
