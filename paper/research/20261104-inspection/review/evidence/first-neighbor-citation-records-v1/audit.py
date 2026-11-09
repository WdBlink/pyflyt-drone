from pathlib import Path
import json, hashlib,re,sys,tempfile,difflib
from html.parser import HTMLParser
R=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection'); E=R/'review/evidence/first-neighbor-citation-records-v1'; E.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def out(n,x): (E/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
plan=[json.loads(x) for x in (R/'literature/citation-plan.first-neighbors.v2.jsonl').read_text().splitlines()]; V=json.loads((R/'literature/citation-verification.first-neighbors.v2.json').read_text())['records']
class Meta(HTMLParser):
 def __init__(self): super().__init__(); self.v={}
 def handle_starttag(self,t,a):
  d=dict(a); n=d.get('name','')
  if t=='meta' and n.startswith('citation_'): self.v.setdefault(n,[]).append(d.get('content'))
bind={}; checks=[]; reads=[]
for row in V:
 v=json.loads((R/row['evidence'][0]).read_text()); m=Meta(); raw=(R/v['metadata_path']).read_text(); m.feed(raw); t=(R/v['read_path']).read_text(); title=m.v['citation_title'][0].replace('$\\alpha^3$','α³'); norm=lambda s:re.sub(r'[.\s]','',s).casefold()
 assertions={'metadata_hash':sha(R/v['metadata_path'])==v['metadata_sha256'],'text_hash':sha(R/v['read_path'])==v['read_sha256'],'title':title==v['identity']['title'],'authors_order':list(map(norm,m.v['citation_author']))==list(map(norm,v['identity']['authors'])),'version_in_metadata':v['identity']['identifier'][6:] in raw,'version_in_text':v['identity']['identifier'][6:] in t,'quote_literal':all(s['quote'] in t for s in v['evidence_spans'])}
 assert all(assertions.values()), (row['key'],assertions)
 checks.append({'key':row['key'],'url':v['url'],'identity':v['identity'],'primary_metadata':m.v,'assertions':assertions,'online_check':'Official arXiv abs and HTML independently opened; authors/title/year/version and cited method context matched. Not full reproduction.'})
 ls=t.split('\n'); reads.append({'key':row['key'],'path':v['read_path'],'sha256':sha(R/v['read_path']),'quote_lf_lines':[i+1 for i,l in enumerate(ls) if any(s['quote'] in l for s in v['evidence_spans'])],'extent':'Targeted abstract/task/interface/method and surrounding context; supplementary clock/replay/tool/metric paragraphs. No full reproduction.'})
 for p in [row['evidence'][0],v['metadata_path'],v['read_path']]: bind[p]=sha(R/p)
repairs=[]
for ver in [1,2]:
 p=R/f'literature/acquisition/bibliography-eprint-output-repair.v{ver}.json'; d=json.loads(p.read_text()); src=R/d['source']; dst=R/d['target']; a=src.read_bytes(); b=dst.read_bytes(); extra=[x for x in b.splitlines(keepends=True) if re.match(rb'  eprint = \{[^\n]+\},\n$',x)]; inverse=b''.join(x for x in b.splitlines(keepends=True) if x not in extra)
 assert sha(src)==d['source_sha256'] and sha(dst)==d['target_sha256'] and len(extra)==7 and inverse==a
 diff=list(difflib.unified_diff(a.decode().splitlines(),b.decode().splitlines())); adds=[x[1:] for x in diff if x.startswith('+') and not x.startswith('+++')]; removes=[x for x in diff if x.startswith('-') and not x.startswith('---')]
 assert len(adds)==7 and not removes
 repairs.append({'record':str(p.relative_to(R)),'source_sha256':sha(src),'target_sha256':sha(dst),'additions':adds,'removals':removes,'inverse_byte_exact':True})
 for x in [p,src,dst]: bind[str(x.relative_to(R))]=sha(x)
lib=R/'runtime/autoresearch-state-reader-v1/lib'; sys.path.insert(0,str(lib.parent)); from lib import production
entries=[{'key':x['key'],'kind':'misc','authors':x['authors'],'title':x['title'],'year':x['year'],'howpublished':x['venue'],'url':x['url']} for x in plan]
with tempfile.TemporaryDirectory(prefix='citation-review-render-') as td:
 production.bibliography({'entries':entries,'target':'native.bib'},Path(td)); got=(Path(td)/'native.bib').read_bytes(); assert got==(R/'literature/references.first-neighbors.v4.bib').read_bytes()
repairs.append({'pure_native_helper_executed':'lib.production.bibliography only, temporary directory; no controller/stage mutation','native_v4_byte_exact':True,'render_sha256':hashlib.sha256(got).hexdigest()})
for p in [R/'literature/packet.first-neighbors.v7.json',R/'literature/citation-check.first-neighbors.v7.json',R/'literature/catalog.first-neighbors.v5.json',R/'literature/citation-plan.first-neighbors.v2.jsonl',R/'literature/citation-verification.first-neighbors.v2.json',R/'literature/references.first-neighbors.v5.bib',lib/'production.py',lib/'science.py']:
 bind[str(p.relative_to(R))]=sha(p)
out('input-bindings.json',bind);out('identity-checks.json',checks);out('output-repair-checks.json',repairs);out('reading-log.json',{'line_convention':'1-based LF only; Python splitlines also splits Unicode separators','records':reads,'extra_anchors':{'retriever-v2.txt':[644,1847],'physai-v1.txt':[284],'aerial-v1.txt':[297]},'read_scope':'Targeted method context. Earlier recorded MultiUAV six-file code audit not redone.'})
print(json.dumps({'identities':len(checks),'input_bindings':len(bind),'repairs':'both seven additions only; inverse exact','native_render_sha256':hashlib.sha256(got).hexdigest(),'packet_sha256':bind['literature/packet.first-neighbors.v7.json'],'catalog_sha256':bind['literature/catalog.first-neighbors.v5.json'],'production_sha256':sha(lib/'production.py'),'science_sha256':sha(lib/'science.py')}))
