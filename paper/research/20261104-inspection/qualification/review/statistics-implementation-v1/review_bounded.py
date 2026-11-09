"""Independent bounded-mean algebra checks and two fixed high-draw diagnostics."""
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

from review_statistics import (
    REPO, SOURCE_PATHS, RootTable, analyze_roots, digest, generate_null,
    manual_holm, metadata, write,
)


def bounded(delta, counts, alpha=.05):
    k = len(counts)
    width = math.fsum(1/(k*k*n) for n in counts)
    p = min(1., 2*math.exp(-delta*delta/(2*width)))
    radius = math.sqrt(2*width*math.log(2/alpha))
    simultaneous_radius = math.sqrt(2*width*math.log(12/alpha))
    return dict(W=width, p=p,
                pointwise95=[max(-1.,delta-radius),min(1.,delta+radius)],
                pointwise_radius=radius,
                bonferroni_global6_simultaneous95=[max(-1.,delta-simultaneous_radius),min(1.,delta+simultaneous_radius)],
                simultaneous_radius=simultaneous_radius)


def main():
    out = Path(__file__).parent / "results.v1"
    frozen = json.loads((out/"source-sha256.json").read_text())
    assert frozen == {p:digest(REPO/p) for p in SOURCE_PATHS}
    exact = []
    for alpha in (.01,.05,.1,.2,.5):
        reject_probability = 0.; total = 0.; coverage = 0.
        # Five independent +/-1 root differences. Stratum means +.5 and
        # -.5 cancel under equal level weights, despite unequal n=2,3.
        for signs in product((-1.,1.), repeat=5):
            probability = math.prod((.75 if s==1 else .25) for s in signs[:2])
            probability *= math.prod((.25 if s==1 else .75) for s in signs[2:])
            delta = (math.fsum(signs[:2])/2 + math.fsum(signs[2:])/3)/2
            width = 1/8 + 1/12
            p = min(1.,2*math.exp(-delta*delta/(2*width)))
            radius = math.sqrt(2*width*math.log(2/alpha))
            reject_probability += probability*(p<=alpha)
            coverage += probability*(delta-radius<=0<=delta+radius)
            total += probability
        assert math.isclose(total,1.) and reject_probability <= alpha+1e-14
        assert coverage >= 1-alpha-1e-14
        exact.append(dict(alpha=alpha,total_probability=total,
                          exact_rejection_probability=reject_probability,exact_coverage=coverage))
    samples = []
    meta = metadata()
    for replication in (0,1):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261105,8101,2,replication])))
        means, diagnostic = generate_null("rare_root_loss_null",rng,meta)
        result = analyze_roots(RootTable.parse(meta,means),"B1",draws=99999,batch_size=256)
        tests={};bounded_p={}
        for domain,counts in (("ID",[16,16,16]),("OOD",[12,12])):
            for name,c in result["domains"][domain]["contrasts"].items():
                key=domain+"."+name
                b=bounded(c["effect"],counts)
                assert math.isclose(b["p"],c["hoeffding_p_robustness_only"],abs_tol=1e-15)
                assert b["pointwise95"][0]<=0<=b["pointwise95"][1]
                assert b["bonferroni_global6_simultaneous95"][0]<=0<=b["bonferroni_global6_simultaneous95"][1]
                bounded_p[key]=b["p"]
                tests[key]={"bootstrap":{k:c[k] for k in ("effect","root_se","p_raw","p_holm_global6","status","exceedances","degenerate_draws","ci95_pointwise")},"bounded":b}
        samples.append(dict(replication=replication,draws=99999,diagnostics=diagnostic,
                            tests=tests,bounded_global6_holm=manual_holm(bounded_p)))
    p_id_no_rare=.98**48;p_ood_no_rare=.98**24
    data=dict(role="independent_algebra_and_fixed_replication_diagnostics_not_research_results",
              counts_weights={"ID":bounded(0.,[16,16,16]),"OOD":bounded(0.,[12,12])},
              exact_finite_support_null=exact,
              no_rare_loss_probability=dict(ID_one_B4=p_id_no_rare,OOD_one_B4=p_ood_no_rare,
                  any_of_two_B4_by_two_domains=1-(1-p_id_no_rare)**2*(1-p_ood_no_rare)**2),
              fixed_high_draw_rechecks=samples,
              source_sha256=frozen,source_unchanged=frozen=={p:digest(REPO/p) for p in SOURCE_PATHS},
              review_script_sha256=digest(Path(__file__)))
    assert data["source_unchanged"]
    write(out/"bounded-and-high-draw-checks.json",data)
    print(json.dumps({"bounded_widths":data["counts_weights"],"no_rare_loss_probability":data["no_rare_loss_probability"],
                      "high_draw_rejected":[[k for k,v in x["tests"].items() if v["bootstrap"]["p_holm_global6"]<=.05] for x in samples]},indent=2))


if __name__=="__main__":main()
