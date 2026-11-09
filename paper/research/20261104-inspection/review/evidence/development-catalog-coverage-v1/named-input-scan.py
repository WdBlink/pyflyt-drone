"""Additional bounded basename/schema scan; do not read owner state or run generators."""
from pathlib import Path
import json,zipfile,hashlib
repo=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');r=repo/'paper/research/20261104-inspection';cat=json.loads((r/'qualification/development-identity-catalog.v1/results.v2/catalog.json').read_text());known={x['case']['sha256'] for x in cat['case_records']};schemas={'inspection-dev-case-v1','inspection-population-case-v2','inspection-event-clean-sibling-v1'}
found=[];unknown=[]
for root in [repo/'logs',r/'qualification']:
 for p in sorted(set(root.rglob('*case*.json'))|set(root.rglob('dev*.json'))):
  if p.name=='private-case.json':continue
  try:v=json.loads(p.read_text())
  except (ValueError,OSError):continue
  if isinstance(v,dict) and v.get('schema') in schemas:
   h=hashlib.sha256(p.read_bytes()).hexdigest();found.append(str(p.relative_to(repo)))
   if h not in known:unknown.append(str(p.relative_to(repo)))
zip_candidates=0;zip_cases=[];zip_unmapped=[]
for p in sorted((r/'qualification').rglob('*.zip')):
 with zipfile.ZipFile(p) as z:
  for info in z.infolist():
   base=Path(info.filename).name
   if base=='private-case.json' or not base.endswith('.json') or not ('case' in base or base.startswith(('dev-','devext-'))):continue
   zip_candidates+=1;raw=z.read(info)
   try:v=json.loads(raw)
   except ValueError:continue
   if isinstance(v,dict) and v.get('schema') in schemas:
    name=str(p.relative_to(r))+'!'+info.filename;zip_cases.append(name)
    if hashlib.sha256(raw).hexdigest() not in known:zip_unmapped.append(name)
assert not unknown and not zip_unmapped
print(json.dumps({'scope':'additional case/dev basenames and exact known CaseSpec schemas; not arbitrary JSON or tree-external history','other_named_schema_files':len(found),'other_named_schema_files_all_known':True,'other_named_schema_file_paths':found,'other_named_ZIP_JSON_candidates_read':zip_candidates,'other_named_ZIP_CaseSpec_members':zip_cases,'unmapped_case_schema_files':unknown,'unmapped_case_schema_ZIP_members':zip_unmapped},indent=2))
