"""Pure frozen scoring, no mission construction/physics/model calls."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
r=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection')
closure=r/'experiment/evaluator-source.v2'; manifest=r/'experiment/evaluator-source-manifest.v2.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(manifest)=='ecc2069fc8795adb9096173ab2d859b6540a94309ecf99310755cb7b3177bf80'
m=json.loads(manifest.read_text())
assert {str(p.relative_to(closure)) for p in closure.rglob('*') if p.is_file()}==set(m['sources'])
assert not any(p.is_symlink() for p in closure.rglob('*'))
assert all(sha(closure/name)==digest for name,digest in m['sources'].items())
sys.path.insert(0,str(closure))
import numpy as np
from eval.inspection_cases import CaseEntity
from eval.inspection_common_calendar import _score
assert np.__version__==m['numpy_version']
clean=json.loads(Path(sys.argv[1]).read_text()); event=json.loads(Path(sys.argv[2]).read_text())
def case(w):
    p=w['private_case']
    assert hashlib.sha256(json.dumps(p,sort_keys=True,allow_nan=False).encode()).hexdigest()==w['case_sha256']
    return SimpleNamespace(level=p['level'],priority_at=p['priority_at'],entities=tuple(CaseEntity(**e) for e in p['entities']))
c,e=case(clean),case(event)
assert all(clean['private_case'][k]==event['private_case'][k] for k in ('case_id','lineage_group','split','level','seed','roi_polygon','wind_xy','entities','domain'))
assert _score(clean,c,c)==clean['score']
print(json.dumps({'role':'pure frozen common-demand score, not new rollout','score':_score(clean,c,e)},allow_nan=False))
