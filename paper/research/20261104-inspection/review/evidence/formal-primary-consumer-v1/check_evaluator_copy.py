"""Execute only frozen runner's file-copy statements against existing evaluator bytes."""
from pathlib import Path
import ast,json,hashlib,sys
HOME=Path('/tmp/formal-primary-consumer-review-v1');PROJECT=HOME/'project';sys.path.insert(0,str(PROJECT))
from eval.inspection_formal_admission import Artifact
repo=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');run=repo/'paper/research/20261104-inspection';out=HOME/'evaluator-copy-only';assert not out.exists()
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();design=json.loads((run/'experiment/design.ral.v3.json').read_text());evaluator=Artifact.parse(design['evaluator']);source_manifest=run/'experiment/evaluator-source-manifest.v2.json';manifest=json.loads(source_manifest.read_text());base=run/'experiment/evaluator-source.v2'
source_names=[source_manifest,*[base/n for n in manifest['sources']]]
closure={'sources':{str(p.relative_to(repo)):h(p) for p in source_names}}
source=(PROJECT/'eval/run_inspection_formal.py').read_text();tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_primary')
loop=next(n for n in ast.walk(fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [getattr(t,'id',None) for t in n.target.elts]==['name','sha'])
copy=next(n for n in ast.walk(fn) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='write_once' and len(n.value.args)==3 and isinstance(n.value.args[1],ast.Attribute) and isinstance(n.value.args[1].value,ast.Name) and n.value.args[1].value.id=='evaluator')
def write_once(root,name,raw):
 p=Path(root)/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(raw)
env={'root':run,'project':repo,'output':out,'evaluator':evaluator,'closure':closure,'source_prefix':str((run/'experiment').relative_to(repo))+'/','write_once':write_once,'Artifact':Artifact}
exec(compile(ast.Module(body=[copy,loop],type_ignores=[]),'<snapshot runner evaluator copy only>','exec'),env)
actual={str(p.relative_to(out/'experiment/evaluator-source.v2')) for p in (out/'experiment/evaluator-source.v2').rglob('*') if p.is_file()}
assert actual==set(manifest['sources']);assert all(h(out/'experiment/evaluator-source.v2'/n)==v for n,v in manifest['sources'].items());assert (out/evaluator.path).read_bytes()==evaluator.checked_path(run).read_bytes();assert (out/'experiment/evaluator-source-manifest.v2.json').read_bytes()==source_manifest.read_bytes()
(HOME/'evaluator-copy-only-result.json').write_text(json.dumps({'scope':'actual snapshot copy AST, actual existing frozen evaluator bytes, literal qualified-source-shaped map; no G4 assertion or evaluator execution','closure_source_count':len(actual),'file_set_exact':True,'all_original_member_SHA_match':True,'evaluator_sha256':h(out/evaluator.path),'manifest_sha256':h(out/'experiment/evaluator-source-manifest.v2.json'),'no_formal_data_or_scores_created':True},indent=2)+'\n')
print('actual frozen evaluator file-copy AST: complete',len(actual),'source members; no evaluator execution')
