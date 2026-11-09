"""Explicit synthetic checks for candidate root inference; no research results."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eval.inspection_recording import CONDITION_REPEATS
from eval.inspection_statistics import RootTable,analyze_roots


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('choose a new evidence directory')
    args.out.mkdir(parents=True)
    paths=[Path(__file__),ROOT/'eval/inspection_statistics.py',ROOT/'eval/inspection_recording.py']
    sources={str(p.relative_to(ROOT)):p.read_text() for p in paths}
    metadata=[];means={}
    for domain,levels,count in (('ID',(1,2,3),16),('OOD',(2,3),12)):
        for level in levels:
            for i in range(count):
                case=f'synthetic-{domain}-L{level}-{i:02d}'
                metadata.append(dict(case_id=case,lineage_group=case,domain=domain,level=level))
                fraction=i/(count-1)
                base=.3+.05*level+.02*fraction
                means[case]={c:base for c in CONDITION_REPEATS}
                means[case].update({'B1':base+.03,'B2':base+.02,'B4-M1':base+.1+.02*fraction,
                                   'B4-M2':base-.04+.01*fraction})
    table=RootTable.parse(metadata,means)
    analysis=analyze_roots(table,'B1',draws=9999)
    for domain in ('ID','OOD'):
        contrasts=analysis['domains'][domain]['contrasts']
        for name,expected in (('architecture-M1',.11),('architecture-M2',-.035),('classic-M1',.08)):
            if abs(contrasts[name]['effect']-expected)>1e-12:raise AssertionError('wrong macro paired estimator')
    reordered=RootTable.parse(list(reversed(metadata)),{k:dict(reversed(list(v.items()))) for k,v in reversed(list(means.items()))})
    if analyze_roots(reordered,'B1',draws=9999,batch_size=173)!=analysis:
        raise AssertionError('input order or batching changed the fixed bootstrap draws')
    zero={k:{c:.5 for c in CONDITION_REPEATS} for k in means}
    constant=analyze_roots(RootTable.parse(metadata,zero),'B1',draws=999)
    if any(c['p_raw']!=1 or c['effect']!=0 or c['ci95_bootstrap_pointwise_nominal']!=[0.,0.]
           for d in constant['domains'].values() for c in d['contrasts'].values()):
        raise AssertionError('degenerate null produced a false claim')
    for name,value in (('case-metadata.json',metadata),('root-means.json',means),('analysis.json',analysis),
                       ('constant-null.json',constant),('source-snapshot.json',sources)):
        (args.out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    if not all((ROOT/p).read_text()==s for p,s in sources.items()):raise AssertionError('source changed during preflight')
    report=dict(role='synthetic_numeric_preflight_not_method_or_test_evidence',passed=True,roots=72,
        main_shape_validated=True,hand_macro_effects=[.11,-.035,.08],input_and_batch_order_invariant=True,
        degenerate_null_conservative=True,source_sha256={p:hashlib.sha256(s.encode()).hexdigest() for p,s in sources.items()},
        primary_test='bounded independent-root Hoeffding p with globalHolm6',
        bootstrap_role='nominal descriptive interval and diagnostic p; not confirmatory',
        limitations=['no real episode/split/lineage/model/cost verification','revised statistics await independent implementation review',
            'finite-sample skew/rare-failure stress tests and SR/censoring/event adapter still pending',
            'bounded guarantee requires independent roots and fixed complete weights; low power at the planned n'])
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
