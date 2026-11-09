from pathlib import Path
import json,hashlib,ast,difflib
ROOT=Path.cwd().resolve();R=ROOT/'paper/research/20261104-inspection';C=R/'qualification/public-contract-clean-candidate.v1';O=R/'review/public-task-contract-sufficiency-v1-evidence/source';T=Path('/tmp/inspection-public-contract-repair-review')
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();binding=json.loads((C/'source-binding.json').read_text());declared=binding['sources'];actual={str(p.relative_to(C/'source')) for p in (C/'source').rglob('*') if p.is_file()};assert actual==set(declared);assert all(H(C/'source'/p)==v for p,v in declared.items())
semantic=('envs/inspection/protocol.py','envs/inspection/scoring.py','envs/inspection/mission.py','envs/inspection/execution.py','envs/inspection/reference.py','eval/inspection_classic.py')
assert all(H(C/'source'/p)==H(O/p) for p in semantic)
checks={}
for name in ('eval/inspection_llm_actor.py','eval/inspection_llm_protocol.py'):
 old=ast.parse((O/name).read_text());new=ast.parse((C/'source'/name).read_text())
 def funcs(tree):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 assert funcs(old)==funcs(new);checks[name]={'all_function_asts_unchanged':True,'functions':len(funcs(old))}
 (T/(Path(name).stem+'.diff')).write_text(''.join(difflib.unified_diff((O/name).read_text().splitlines(True),(C/'source'/name).read_text().splitlines(True),fromfile='original',tofile='frozen-candidate')))
# The only scenario change is removal of one literal metadata key, preserving all rules.
name='envs/inspection/scenario.py';old=ast.parse((O/name).read_text());new=ast.parse((C/'source'/name).read_text());removed=[]
for node in ast.walk(old):
 if isinstance(node,ast.Dict):
  keep=[i for i,k in enumerate(node.keys) if not (isinstance(k,ast.Constant) and k.value=='qualification_status')]
  removed += [node.values[i].value for i in range(len(node.keys)) if i not in keep]
  node.keys=[node.keys[i] for i in keep];node.values=[node.values[i] for i in keep]
assert removed==['development_unqualified_new_modes'];assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
(T/'scenario.diff').write_text(''.join(difflib.unified_diff((O/name).read_text().splitlines(True),(C/'source'/name).read_text().splitlines(True),fromfile='original',tofile='frozen-candidate')))
# Parse only; do not import or execute candidate modules.
p=ast.parse((C/'source/eval/inspection_llm_protocol.py').read_text());contract=next(n.value.value for n in p.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PUBLIC_TOOL_CONTRACT' for t in n.targets));assert isinstance(contract,str)
a=ast.parse((C/'source/eval/inspection_llm_actor.py').read_text());s=next(n for n in a.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SYSTEM' for t in n.targets));assert any(isinstance(n,ast.Name) and n.id=='PUBLIC_TOOL_CONTRACT' for n in ast.walk(s.value))
result={'reviewer':'/root/control_review','source_only_no_candidate_execution':True,'candidate_binding_sha256':H(C/'source-binding.json'),'all_13_declared_hashes_and_exact_file_set_match':True,'reviewed_semantic_sources_byte_identical_to_original':list(semantic),'ast_checks':checks,'scenario_only_removed_key':removed,'public_contract_is_string_constant':True,'actor_system_statically_appends_public_contract':True,'ablation_function_ast_unchanged':True,'unreviewed_but_hash_verified':['eval/inspection_population.py','eval/inspection_cases.py','eval/inspection_scenarios.py','tests/test_inspection_population.py']}
(T/'source-check-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))
