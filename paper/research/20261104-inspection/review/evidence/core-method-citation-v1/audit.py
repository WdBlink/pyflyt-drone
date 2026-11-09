from pathlib import Path
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,re,subprocess,sys,tempfile,urllib.request
R=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection');E=R/'review/evidence/core-method-citation-v1'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def out(n,d): (E/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
class Meta(HTMLParser):
 def __init__(self):super().__init__();self.v={}
 def handle_starttag(self,t,a):
  d=dict(a);n=d.get('name','')
  if t=='meta' and n.startswith('citation_'):self.v.setdefault(n,[]).append(d.get('content'))
files=['literature/packet.core-and-mechanisms.v1.json','literature/citation-check.core-and-mechanisms.v1.json','literature/catalog.core-and-mechanisms.v1.json','literature/citation-plan.core-and-mechanisms.v1.jsonl','literature/citation-verification.core-and-mechanisms.v1.json','literature/references.core-and-mechanisms.v1.bib','literature/references.core-and-mechanisms.v2.bib','literature/acquisition/core-methods-partial-fetch-and-conversion.v1.json','literature/acquisition/bibliography-core-eprint-output-repair.v1.json','literature/acquisition/bibliography-core-and-mechanisms.v1.json','literature/core-method-adaptation-notes.v1.md','literature/acquisition/source-fetch-core-methods.v1.json']
partial=json.loads((R/files[7]).read_text());ids=[]
P=[json.loads(x) for x in (R/files[3]).read_text().splitlines()];V=json.loads((R/files[4]).read_text())['records'];B=json.loads((R/files[9]).read_text())['entries']
for key,c in zip(['PyFlyt2023','ReAct2022'],partial['conversion']):
 vp=f'literature/verification/{key}.v1.json';v=json.loads((R/vp).read_text());m=Meta();raw=(R/v['metadata_path']).read_text();m.feed(raw);txt=(R/v['read_path']).read_text();pdf=R/c['pdf'];prov=json.loads(Path(str(pdf)+'.source.json').read_text());ident=v['identity']['identifier'][6:];p=next(x for x in P if x['key']==key);vr=next(x for x in V if x['key']==key);b=next(x for x in B if x['key']==key)
 norm=lambda x:re.sub(r'[.\s]','',x).casefold()
 assertions={'metadata_sha':h(R/v['metadata_path'])==v['metadata_sha256'],'text_sha':h(R/v['read_path'])==v['read_sha256']==c['text_sha256'],'pdf_provenance_sha':h(pdf)==prov['sha256'],'title_metadata':m.v['citation_title'][0]==v['identity']['title'],'authors_metadata_order':[norm(x) for x in m.v['citation_author']]==[norm(x) for x in v['identity']['authors']],'first_date':m.v['citation_date']==[v['first_publication_date']],'update_date':m.v['citation_online_date']==[v['version_update_date']],'version_in_pdf_text':f'arXiv:{ident}' in txt,'version_in_metadata':ident in raw,'quotes_exist':all(s['quote'] in txt for s in v['evidence_spans']),'year_is_first_publication':v['identity']['year']==int(v['first_publication_date'][:4]),'plan_identity_exact':all(p[k]==v['identity'][k] for k in ['title','authors','year','venue','identifier']),'plan_claim_exact':p['claims'][0]['text']==v['claim'],'merged_verification_identity_exact':vr['identity']==v['identity'],'native_bib_packet_identity_exact':all(b[k]==p[k] for k in ['title','authors','year','url']) and b['howpublished']==p['venue']}
 with tempfile.TemporaryDirectory(prefix='core-citation-extract-') as td:
  target=Path(td)/'text.txt';subprocess.run(['/opt/homebrew/bin/pdftotext','-layout',str(pdf),str(target)],check=True);assertions['independent_pdftotext_byte_exact']=target.read_bytes()==(R/v['read_path']).read_bytes()
 assert all(assertions.values()),(key,assertions)
 ids.append({'key':key,'identity':v['identity'],'primary_metadata_authors':m.v['citation_author'],'first_publication':v['first_publication_date'],'version_updated':v['version_update_date'],'assertions':assertions,'pdf':c['pdf'],'pdf_sha256':h(pdf),'quotes_lf_lines':[[i+1 for i,l in enumerate(txt.split('\n')) if s['quote'] in l] for s in v['evidence_spans']],'source_name_variant':{'metadata':'Mauro Innocente','pdf':'Mauro S. Innocente','treatment':'Retain both as explicit same-position source variants; do not invent an added author or infer a new expanded name.'} if key=='PyFlyt2023' else None})
 files.extend([vp,v['metadata_path'],v['read_path'],c['pdf'],str(Path(c['pdf']+'.source.json')),v['metadata_path']+'.source.json'])
oldP=[json.loads(x) for x in (R/'literature/citation-plan.mechanism-neighbors.v1.jsonl').read_text().splitlines()];oldV=json.loads((R/'literature/citation-verification.mechanism-neighbors.v1.json').read_text())['records'];assert P[:9]==oldP and V[:9]==oldV
oldbib=(R/'literature/references.mechanism-neighbors.v2.bib').read_bytes();newbib=(R/files[6]).read_bytes();assert newbib.startswith(oldbib+b'\n@misc{PyFlyt2023,')
repair=json.loads((R/files[8]).read_text());a=(R/repair['source']).read_bytes();b=(R/repair['target']).read_bytes();lines=b.splitlines(keepends=True);extra=[x for x in lines if re.fullmatch(rb'  eprint = \{[^\n]+\},\n',x)];assert len(extra)==11 and b''.join(x for x in lines if x not in extra)==a and h(R/repair['source'])==repair['source_sha256'] and h(R/repair['target'])==repair['target_sha256'];assert [x.decode().strip().split('{',1)[1][:-2] for x in extra]==[x['identifier'][6:] for x in P]
lib=R/'runtime/autoresearch-state-reader-v1/lib';sys.path.insert(0,str(lib.parent));from lib import production
payload=json.loads((R/files[9]).read_text());payload['target']='native.bib'
with tempfile.TemporaryDirectory(prefix='core-citation-bib-') as td:
 production.bibliography(payload,Path(td));assert (Path(td)/'native.bib').read_bytes()==a
out('identity-and-conversion-checks.json',ids);out('bibliography-and-inheritance-checks.json',{'first9_plan_objects_exactly_unchanged':True,'first9_verification_objects_exactly_unchanged':True,'first9_repaired_bib_exact_prefix':True,'eprint_added_lines':11,'eprint_identifiers_exact_and_ordered':True,'inverse_byte_exact':True,'native_helper_rerender_byte_exact':True,'native_sha256':hashlib.sha256(a).hexdigest(),'repaired_sha256':hashlib.sha256(b).hexdigest()})
def fetch(d):
 url='https://arxiv.org/pdf/'+d['identity']['identifier'][6:]
 try:
  digest=hashlib.sha256();n=0
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'citation-record-review/1.0'}),timeout=25) as f:
   final=f.url
   while True:
    chunk=f.read(1048576)
    if not chunk:break
    digest.update(chunk);n+=len(chunk)
  return {'key':d['key'],'url':url,'resolved_url':final,'bytes':n,'sha256':digest.hexdigest(),'matches_existing_pdf':digest.hexdigest()==d['pdf_sha256']}
 except Exception as e:return {'key':d['key'],'url':url,'error':type(e).__name__+': '+str(e)}
with ThreadPoolExecutor(max_workers=2) as ex:live=list(ex.map(fetch,ids))
out('official-pdf-byte-checks.json',live)
actor=R.parents[2]/'eval/inspection_llm_actor.py';assert h(actor)=='cd1bf019c637d447946cba30e1e2d8572c4ff90594de4b5b2d90f49cd65ccc7b';fetchlog=Path('/tmp/inspection-literature-core-methods-fetch.v1.log');failure=json.loads(fetchlog.read_text());assert failure=={'status':'refused','reason':'HTTP Error 403: Forbidden'}
out('scope-and-source-variants.json',{'actor_sha256':h(actor),'actor_static_lines_read':[[1,43],[60,105],[110,121],[154,212]],'actor_scope':'Static bounds/prompt/history inspection only, no actor import or execution, no trace or runner certification. recent_exchanges is at most two and may be further trimmed by byte allowance. B3 cadence is due when >=20 simulated seconds, not a promise of exact on-time dispatch.','openreview_provided_fetch_log_sha256':h(fetchlog),'openreview_provided_fetch_log':failure,'openreview_scope':'Existing 403 refusal verified; earlier browser challenge retained as owner provenance, not independently replayed here; no conference decision artifact and no claim of nonacceptance.','reading_ranges_LF':{'pyflyt-v1.txt':[[1,75],[180,229]],'react-v3.txt':[[1,35],[201,239]]},'pdf_visual_audit':False,'full_reproduction':False})
for p in ['runtime/autoresearch-state-reader-v1/lib/production.py','runtime/autoresearch-state-reader-v1/lib/science.py','literature/citation-plan.mechanism-neighbors.v1.jsonl','literature/citation-verification.mechanism-neighbors.v1.json','literature/references.mechanism-neighbors.v2.bib']:files.append(p)
bind={p:h(R/p) for p in files};bind[str(actor)]=h(actor);bind[str(fetchlog)]=h(fetchlog);out('input-bindings.json',bind)
print(json.dumps({'new_identities':2,'pdftotext_exact':True,'first9_unchanged':True,'eprint11_inverse_exact':True,'native_render_exact':True,'official_pdf_checks':live,'input_count':len(bind)},ensure_ascii=False))
