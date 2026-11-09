"""AST comparison only; never invoke a generator."""
from pathlib import Path
import ast,copy,hashlib,json
R=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection');F=R/'planning/formal-admission-candidate.v3/population-component-freeze.v1'
old=ast.parse((F/'production-before-inspection_population.py').read_text());new=ast.parse((F/'inspection_population.py').read_text())
dump=lambda x:ast.dump(x,include_attributes=False)
defs=lambda x:{v.name:v for v in x.body if isinstance(v,(ast.FunctionDef,ast.ClassDef))}
o,n=defs(old),defs(new)
assert [dump(v) for v in o['validate_population_case'].body[1:]]==[dump(v) for v in n['_validate_population_contents'].body]
unchanged=[k for k in o if k not in ('validate_population_case','generate_development_population_case')]
assert all(dump(o[k])==dump(n[k]) for k in unchanged)
assert dump(o['validate_population_case'].body[0])==dump(n['validate_population_case'].body[0])
assert dump(o['generate_development_population_case'].body[0])==dump(n['generate_development_population_case'].body[0])
# Normalize only the explicitly intended formal identity/validation/audit changes.
class Normalize(ast.NodeTransformer):
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Name) and node.func.id=='PopulationCase':
   assert isinstance(node.args[3],ast.Name) and node.args[3].id=='phase';node.args[3]=ast.Constant('dev')
  return node
 def visit_Assign(self,node):
  if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
   name=node.targets[0].id
   if name=='validation':
    assert isinstance(node.value,ast.IfExp);node.value=node.value.body
   if name=='role':return None
  return self.generic_visit(node)
 def visit_Dict(self,node):
  self.generic_visit(node)
  for i,k in enumerate(node.keys):
   if isinstance(k,ast.Constant) and k.value=='role' and isinstance(node.values[i],ast.Name) and node.values[i].id=='role':node.values[i]=ast.Constant('development geometry generation, not test or flight')
  return node
old_body=o['generate_development_population_case'].body[1:]
new_body=n['_generate_population_case'].body[2:]
normalized=[]
for v in copy.deepcopy(new_body):
 v=Normalize().visit(v)
 if v is not None:normalized.append(v)
assert [dump(v) for v in old_body]==[dump(v) for v in normalized]
loops=[x for x in ast.walk(n['_generate_population_case']) if isinstance(x,ast.For) and isinstance(x.iter,ast.Call) and isinstance(x.iter.func,ast.Name) and x.iter.func.id=='range' and len(x.iter.args)==1 and isinstance(x.iter.args[0],ast.Constant) and x.iter.args[0].value==1000]
assert len(loops)==1
assert any(isinstance(v,ast.Raise) and isinstance(v.exc,ast.Call) and isinstance(v.exc.func,ast.Name) and v.exc.func.id=='PopulationGenerationExhausted' for v in n['_generate_population_case'].body)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=json.loads((F/'binding.json').read_text())
assert sha(F/'binding.json')=='a098c81c91a11c0120d9d519c7860c965cacc89bc41d2f33019165a7442b8f1e'
assert all(sha(R/k)==v for k,v in b['files'].items())
repo=R.parents[2];assert all(sha(repo/k)==v for k,v in b['working_sources'].items())
assert sha(repo/'eval/inspection_population.py')==b['production_before_sha256']
result={'binding_sha256':sha(F/'binding.json'),'all_frozen_and_working_files_verified':True,'production_unchanged':True,'content_validator_body_AST_identical':True,'legacy_dev_identity_guards_AST_identical':True,'other_original_definitions_AST_identical':unchanged,'generation_body_AST_identical_after_only_named_transformations':['PopulationCase split dev→phase','conditional dev/formal identity validator','dev/formal audit role'],'shared_five_stream_law_preserved_by_AST':True,'single_finite_outer_range_1000_and_same_exhaustion_class':True,'runtime_forced_1000_exhaustion_not_attempted':True,'datasets_absent':not(R/'datasets').exists(),'source_hashes':{'production_population':sha(repo/'eval/inspection_population.py'),'candidate_population':sha(F/'inspection_population.py'),'candidate_formal_wrapper':sha(F/'inspection_formal_population.py'),'inspection_cases':sha(repo/'eval/inspection_cases.py'),'scenario':sha(repo/'envs/inspection/scenario.py')},'five_existing_replay_inputs':{str(p.relative_to(R)):sha(p) for p in [R/'qualification/dev-population.v1/manifest.json',*sorted((R/'qualification/dev-population.v1').glob('devext-*.json'))]}}
assert result['datasets_absent']
print(json.dumps(result,indent=2)+'\n')
