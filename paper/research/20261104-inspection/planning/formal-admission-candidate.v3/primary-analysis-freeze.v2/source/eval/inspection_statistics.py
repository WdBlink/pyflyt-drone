"""Root-paired bounded inference; this module does not certify input evidence.

Confirmatory p-values use independent bounded root differences and fixed
weights. Bootstrap intervals/reference p-values are explicitly descriptive.
"""

from dataclasses import dataclass
import math

import numpy as np

from eval.inspection_recording import CONDITION_REPEATS


CONDITIONS=tuple(CONDITION_REPEATS)
SEEDS={"ID":2026110401,"OOD":2026110402}
MAIN_SHAPE={"ID":{1:16,2:16,3:16},"OOD":{2:12,3:12}}
SE_EPS=1e-12


@dataclass(frozen=True)
class RootTable:
    roots: tuple[str,...]
    groups: tuple[tuple[str,int],...]
    values: np.ndarray

    @classmethod
    def parse(cls,metadata,means,*,require_main_shape=True):
        ids=[r["case_id"] for r in metadata]
        lineage=[r["lineage_group"] for r in metadata]
        if (not ids or len(set(ids))!=len(ids) or len(set(lineage))!=len(lineage)
                or any(not isinstance(x,str) or not x for x in ids+lineage) or set(means)!=set(ids)):
            raise ValueError("complete unique independent-root identities required; siblings are not new roots")
        ordered=sorted(metadata,key=lambda r:r['case_id'])
        groups=[]
        rows=[]
        for row in ordered:
            domain,level=row["domain"],row["level"]
            if domain not in MAIN_SHAPE or type(level) is not int or level not in MAIN_SHAPE[domain]:
                raise ValueError("invalid domain/level")
            values=means[row["case_id"]]
            if set(values)!=set(CONDITIONS):raise ValueError("all eight root-condition means required")
            if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.number))
                   or not math.isfinite(v) or not 0<=v<=1 for v in values.values()):
                raise ValueError("root means must be finite Y in[0,1]")
            groups.append((domain,level))
            rows.append([float(values[c]) for c in CONDITIONS])
        counts={g:{l:groups.count((g,l)) for l in levels} for g,levels in MAIN_SHAPE.items()}
        if require_main_shape and counts!=MAIN_SHAPE:raise ValueError("main analysis requires the complete48ID+24OOD roots")
        if any(groups.count(key)<2 for key in set(groups)):raise ValueError("at least two roots per resampled stratum")
        array=np.asarray(rows,dtype=np.float64);array.flags.writeable=False
        return cls(tuple(r["case_id"] for r in ordered),tuple(groups),array)


def macro_difference_and_se(stratum_differences):
    arrays=[np.asarray(a,dtype=np.float64) for a in stratum_differences]
    if not arrays or any(a.ndim!=2 or len(a)<2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("finite root-by-contrast strata with at least two roots required")
    k=len(arrays)
    delta=sum(a.mean(axis=0) for a in arrays)/k
    variance=sum(a.var(axis=0,ddof=1)/(k*k*len(a)) for a in arrays)
    return delta,np.sqrt(variance)


def studentized_reference(delta,se,draw_deltas,draw_se):
    delta,se,draw_deltas,draw_se=map(np.asarray,(delta,se,draw_deltas,draw_se))
    if (draw_deltas.shape!=draw_se.shape or draw_deltas.ndim!=2 or draw_deltas.shape[1]!=len(delta)
            or se.shape!=delta.shape or not len(draw_deltas)
            or any(not np.isfinite(a).all() for a in (delta,se,draw_deltas,draw_se))
            or np.any(se<0) or np.any(draw_se<0)):
        raise ValueError("finite compatible nonnegative-SE bootstrap arrays required")
    result=[]
    for c in range(len(delta)):
        degenerate=draw_se[:,c]<=SE_EPS
        count=int(degenerate.sum())
        if se[c]<=SE_EPS:
            result.append(dict(p_raw=1.,status="degenerate_or_near_degenerate",T_obs=None,
                               exceedances=None,degenerate_draws=count))
            continue
        observed=abs(float(delta[c]/se[c]))
        usable=~degenerate
        exceed=count+int((np.abs((draw_deltas[usable,c]-delta[c])/draw_se[usable,c])>=observed).sum())
        unstable=count/len(draw_deltas)>.01
        result.append(dict(p_raw=1. if unstable else (1+exceed)/(len(draw_deltas)+1),
            status="bootstrap_reference_unstable" if unstable else "nominal_asymptotic",
            T_obs=float(delta[c]/se[c]),exceedances=exceed,degenerate_draws=count))
    return result


def holm(p_values):
    if not p_values or any(isinstance(v,bool) or not math.isfinite(v) or not 0<=v<=1 for v in p_values.values()):
        raise ValueError("complete valid p-value family required")
    ordered=sorted(p_values,key=lambda key:(p_values[key],key))
    adjusted={};last=0.
    for rank,key in enumerate(ordered):
        last=max(last,(len(ordered)-rank)*p_values[key])
        adjusted[key]=min(1.,last)
    return adjusted


def bounded_reference(delta, weight_square_sum):
    """Hoeffding two-sided p and conservative clipped confidence intervals.

Each independent root difference is in[-1,1], with deterministic normalized
weights. Thus P(|estimate-E estimate|>=t)<=2exp(-t²/(2*sum(weights²))).
The simultaneous interval uses Bonferroni, not inversion of Holm.
"""
    if (isinstance(delta, (bool,np.bool_)) or not math.isfinite(delta) or not -1<=delta<=1
            or isinstance(weight_square_sum, (bool,np.bool_))
            or not math.isfinite(weight_square_sum) or not 0<weight_square_sum<=1):
        raise ValueError("bounded effect and valid frozen weights/family required")
    point = math.sqrt(2*weight_square_sum*math.log(40))
    simultaneous = math.sqrt(2*weight_square_sum*math.log(240))
    interval = lambda radius: [max(-1., float(delta)-radius), min(1., float(delta)+radius)]
    return {"p_raw": min(1., 2*math.exp(-float(delta)**2/(2*weight_square_sum))),
            "root_weight_square_sum": weight_square_sum,
            "ci95_bounded_pointwise": interval(point),
            "ci95_bounded_simultaneous_global6_bonferroni": interval(simultaneous)}


def analyze_roots(table,classic_best,*,draws=99999,batch_size=256):
    if classic_best not in ("B1","B2") or type(draws) is not int or draws<2 or type(batch_size) is not int or batch_size<1:
        raise ValueError("frozen comparator and finite bootstrap parameters required")
    if set(g for g,l in table.groups)!={"ID","OOD"}:raise ValueError("both domains required for the fixed six-test family")
    pairs=(("architecture-M1","B4-M1","B3-M1"),("architecture-M2","B4-M2","B3-M2"),
           ("classic-M1","B4-M1",classic_best))
    left=[CONDITIONS.index(a) for _,a,b in pairs];right=[CONDITIONS.index(b) for _,a,b in pairs]
    results={};p_values={};bootstrap_p_values={}
    for domain in ("ID","OOD"):
        levels=sorted({l for g,l in table.groups if g==domain})
        if levels!=sorted(MAIN_SHAPE[domain]):raise ValueError("all frozen domain/level strata required")
        cells=[table.values[[i for i,g in enumerate(table.groups) if g==(domain,l)]] for l in levels]
        if any(len(a)<2 for a in cells):raise ValueError("incomplete variance strata")
        difference=[a[:,left]-a[:,right] for a in cells]
        delta,se=macro_difference_and_se(difference)
        means=sum(a.mean(axis=0) for a in cells)/len(cells)
        streams=[np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEEDS[domain],l]))) for l in levels]
        sampled=np.zeros((draws,len(CONDITIONS)),dtype=np.float64)
        sampled_se=np.zeros((draws,len(pairs)),dtype=np.float64)
        for start in range(0,draws,batch_size):
            end=min(draws,start+batch_size);size=end-start
            for values,rng in zip(cells,streams):
                indices=rng.integers(0,len(values),size=(size,len(values)))
                block=values[indices]
                sampled[start:end]+=block.mean(axis=1)/len(cells)
                d=block[:,:,left]-block[:,:,right]
                sampled_se[start:end]+=d.var(axis=1,ddof=1)/(len(cells)**2*len(values))
        sampled_se=np.sqrt(sampled_se)
        difference_draws=sampled[:,left]-sampled[:,right]
        references=studentized_reference(delta,se,difference_draws,sampled_se)
        contrasts={}
        width=sum(1/(len(cells)**2*len(a)) for a in cells)
        for i,(name,a,b) in enumerate(pairs):
            key=f"{domain}.{name}"
            bound=bounded_reference(float(delta[i]),width)
            p_values[key]=bound["p_raw"]
            bootstrap_p_values[key]=references[i]["p_raw"]
            root_differences=[{"case_id":table.roots[j],"level":level,"difference":float(table.values[j,left[i]]-table.values[j,right[i]])}
                for j,(g,level) in enumerate(table.groups) if g==domain]
            contrasts[name]={"minuend":a,"subtrahend":b,"effect":float(delta[i]),"root_se":float(se[i]),
                "ci95_bootstrap_pointwise_nominal":np.quantile(difference_draws[:,i],[.025,.975],method="linear").tolist(),
                "root_differences":root_differences,"root_difference_median_unweighted":float(np.median([r['difference'] for r in root_differences])),
                "bootstrap_reference_diagnostic":references[i],**bound}
        results[domain]={"roots":sum(len(a) for a in cells),"stratum_counts":dict(zip(map(str,levels),map(len,cells))),
            "level_macro_mean":dict(zip(CONDITIONS,map(float,means))),"contrasts":contrasts}
    adjusted=holm(p_values)
    bootstrap_adjusted=holm(bootstrap_p_values)
    for domain,result in results.items():
        for name,contrast in result['contrasts'].items():
            contrast['p_holm_global6']=adjusted[f'{domain}.{name}']
            contrast['reject_primary_alpha05']=contrast['p_holm_global6']<=.05
            contrast['bootstrap_reference_diagnostic']['p_holm_global6_not_confirmatory']=bootstrap_adjusted[f'{domain}.{name}']
    return {"scope":"numeric_root_inference_not_input_evidence_or_scientific_acceptance",
        "spec":{"draws":draws,"rng":"NumPy PCG64 SeedSequence([domain_seed,level])","domain_seeds":SEEDS,
            "numpy":np.__version__,"quantile":"linear","SE_EPS":SE_EPS,"unstable_draw_fraction_limit":.01,
            "multiplicity":"globalHolm6; three prespecified contrasts times two domains","alpha":.05,
            "primary_test":"Hoeffding bounded independent-root p; fixed d in[-1,1], weight1/(K*n_level)",
            "primary_intervals":"bounded pointwise95 and Bonferroni-global6 simultaneous95; not Holm inversion",
            "bootstrap_role":"nominal/descriptive pointwise interval and reference p diagnostic only",
            "design_revision":"pre-formal-data synthetic rare-loss falsification; original candidate retained in independent review"},
        "domains":results,"limitations":["requires iid independent roots within frozen strata",
            "bounded p/interval guarantees are conditional on independent roots, deterministic weights and no optional selection",
            "bounded intervals are wide at12/16 roots per stratum; no promise of detecting a.05 effect or evidence of equivalence",
            "bootstrap intervals/reference p may fail under rare losses; they are not confirmatory evidence",
            "real trace, split, model, cost, schedule and classic selection provenance are checked outside this numeric routine"]}
