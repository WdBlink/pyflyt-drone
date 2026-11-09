from pathlib import Path
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,re,subprocess,sys,tempfile,urllib.request,difflib
R=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection')
E=R/'review/evidence/smart-gradnav-mechanism-citation-v1'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def out(n,d): (E/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
class Meta(HTMLParser):
 def __init__(self): super().__init__();self.v={}
 def handle_starttag(self,t,a):
  d=dict(a);n=d.get('name','')
  if t=='meta' and n.startswith('citation_'):self.v.setdefault(n,[]).append(d.get('content'))
files=['literature/packet.mechanism-neighbors.v1.json','literature/citation-check.mechanism-neighbors.v1.json','literature/catalog.mechanism-neighbors.v1.json','literature/citation-plan.mechanism-neighbors.v1.jsonl','literature/citation-verification.mechanism-neighbors.v1.json','literature/references.mechanism-neighbors.v1.bib','literature/references.mechanism-neighbors.v2.bib','literature/acquisition/mechanism-comparator-text-conversion.v1.json','literature/acquisition/bibliography-mechanism-eprint-output-repair.v1.json','literature/acquisition/bibliography-mechanism-neighbors.v1.json','literature/mechanism-boundaries.v2.md']
conv=json.loads((R/files[7]).read_text())['records']; ids=[]
for key,c in zip(['SMART2025','GRaDNav2025'],conv):
 p=f'literature/verification/{key}.v1.json';v=json.loads((R/p).read_text());m=Meta();raw=(R/v['metadata_path']).read_text();m.feed(raw);txt=(R/v['read_path']).read_text();pdf=R/c['source']; ident=v['identity']['identifier'][6:]
 norm=lambda x:re.sub(r'[.\s]','',x).casefold()
 assertions={'metadata_sha':h(R/v['metadata_path'])==v['metadata_sha256'],'text_sha':h(R/v['read_path'])==v['read_sha256']==c['text_sha256'],'pdf_sha':h(pdf)==c['source_sha256'],'title':m.v['citation_title'][0]==v['identity']['title'],'authors_in_order':[norm(x) for x in m.v['citation_author']]==[norm(x) for x in v['identity']['authors']],'first_date':m.v['citation_date']==[v['first_publication_date']],'update_date':m.v['citation_online_date']==[v['version_update_date']],'version_in_pdf_text':f'arXiv:{ident}' in txt,'version_in_metadata':ident in raw,'quotes_exist':all(s['quote'] in txt for s in v['evidence_spans']),'year_is_first_publication':v['identity']['year']==int(v['first_publication_date'][:4])}
 with tempfile.TemporaryDirectory(prefix='mechanism-citation-extract-') as td:
  dst=Path(td)/'text.txt'; subprocess.run(['/opt/homebrew/bin/pdftotext','-layout',str(pdf),str(dst)],check=True);assertions['independent_pdftotext_byte_exact']=dst.read_bytes()==(R/v['read_path']).read_bytes()
 assert all(assertions.values()),(key,assertions)
 ids.append({'key':key,'identity':v['identity'],'dates':{k:m.v[k] for k in ['citation_date','citation_online_date']},'doi_additional_only':v['metadata_declared_doi'],'url':v['url'],'pdf':c['source'],'pdf_sha256':h(pdf),'assertions':assertions,'quote_lf_lines':[[i+1 for i,l in enumerate(txt.split('\n')) if s['quote'] in l] for s in v['evidence_spans']]})
 files.extend([p,v['metadata_path'],v['read_path'],c['source']])
old=[json.loads(x) for x in (R/'literature/citation-plan.first-neighbors.v2.jsonl').read_text().splitlines()]
new=[json.loads(x) for x in (R/'literature/citation-plan.mechanism-neighbors.v1.jsonl').read_text().splitlines()]
oldv=json.loads((R/'literature/citation-verification.first-neighbors.v2.json').read_text())['records'];newv=json.loads((R/'literature/citation-verification.mechanism-neighbors.v1.json').read_text())['records']
assert new[:7]==old and newv[:7]==oldv
repair=json.loads((R/'literature/acquisition/bibliography-mechanism-eprint-output-repair.v1.json').read_text());a=(R/repair['source']).read_bytes();b=(R/repair['target']).read_bytes();lines=b.splitlines(keepends=True);extra=[x for x in lines if re.fullmatch(rb'  eprint = \{[^\n]+\},\n',x)];inverse=b''.join(x for x in lines if x not in extra);assert len(extra)==9 and inverse==a and h(R/repair['source'])==repair['source_sha256'] and h(R/repair['target'])==repair['target_sha256']
assert [x.decode().strip().split('{',1)[1][:-2] for x in extra]==[x['identifier'][6:] for x in new]
lib=R/'runtime/autoresearch-state-reader-v1/lib';sys.path.insert(0,str(lib.parent));from lib import production
payload=json.loads((R/'literature/acquisition/bibliography-mechanism-neighbors.v1.json').read_text());payload['target']='native.bib'
with tempfile.TemporaryDirectory(prefix='mechanism-citation-bib-') as td:
 production.bibliography(payload,Path(td));assert (Path(td)/'native.bib').read_bytes()==a
out('identity-and-conversion-checks.json',ids)
out('bibliography-and-inheritance-checks.json',{'first7_plan_records_exactly_unchanged':True,'first7_verification_records_exactly_unchanged':True,'new_eprint_lines':len(extra),'inverse_byte_exact':True,'eprint_ids_exact_in_plan_order':True,'native_helper_rerender_byte_exact':True,'native_sha256':hashlib.sha256(a).hexdigest(),'repaired_sha256':hashlib.sha256(b).hexdigest()})
# Separate, bounded official PDF byte comparison; failures remain explicit instead of becoming identity claims.
def fetch(d):
 url='https://arxiv.org/pdf/'+d['identity']['identifier'][6:];q=urllib.request.Request(url,headers={'User-Agent':'citation-record-review/1.0'})
 try:
  digest=hashlib.sha256();n=0
  with urllib.request.urlopen(q,timeout=25) as f:
   final=f.url
   while True:
    chunk=f.read(1048576)
    if not chunk:break
    digest.update(chunk);n+=len(chunk)
  return {'key':d['key'],'url':url,'resolved_url':final,'bytes':n,'sha256':digest.hexdigest(),'matches_existing_pdf':digest.hexdigest()==d['pdf_sha256']}
 except Exception as e:return {'key':d['key'],'url':url,'error':type(e).__name__+': '+str(e)}
with ThreadPoolExecutor(max_workers=2) as ex: live=list(ex.map(fetch,ids))
out('official-pdf-byte-checks.json',live)
files.extend(['runtime/autoresearch-state-reader-v1/lib/production.py','runtime/autoresearch-state-reader-v1/lib/science.py','literature/citation-plan.first-neighbors.v2.jsonl','literature/citation-verification.first-neighbors.v2.json'])
out('input-bindings.json',{p:h(R/p) for p in files})
out('reading-log.json',{'line_convention':'1-based LF','SMART2025':{'title_authors_version':[1,48],'task_definition':[128,148],'methods_III_B_C':[278,317],'extent':'Targeted PDF text; no scalability reproduction, implementation or visual figure audit'},'GRaDNav2025':{'title_authors_version':[1,44],'III_B1':[197,218],'III_D':[239,266],'extent':'Targeted PDF text; reports flight profile as paper statement, not independent hardware measurement'},'official_abs_pages':['https://arxiv.org/abs/2503.04798v3','https://arxiv.org/abs/2506.14009v2'],'online_scope':'Official abs title/authors/history and pinned PDF title/version independently opened; no journal edition substitution.'})
print(json.dumps({'identity_records':len(ids),'pdftotext_exact':True,'first7_unchanged':True,'eprint_repair9_inverse_exact':True,'native_render_exact':True,'official_pdf_checks':live},ensure_ascii=False))
