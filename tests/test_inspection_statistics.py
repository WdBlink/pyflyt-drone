import numpy as np
import pytest

from eval.inspection_recording import CONDITION_REPEATS
from eval.inspection_statistics import RootTable,analyze_roots,bounded_reference,holm,macro_difference_and_se,studentized_reference


def fixture():
    metadata=[];means={}
    for domain,levels in (("ID",(1,2,3)),("OOD",(2,3))):
        for level in levels:
            for i in range(3):
                case=f'{domain}-{level}-{i}'
                metadata.append(dict(case_id=case,lineage_group=case,domain=domain,level=level))
                means[case]={c:.2+.01*level+.02*i+(.05+.01*i if c=='B4-M1' else 0) for c in CONDITION_REPEATS}
    return metadata,means


def test_stratum_macro_and_paired_se_match_independent_hand_calculation():
    delta,se=macro_difference_and_se([np.array([[0.],[.2]]),np.array([[.4],[.6],[.8],[.2]])])
    assert delta[0]==pytest.approx(.3)
    assert se[0]==pytest.approx(np.sqrt(1/150))
    assert delta[0]!=pytest.approx(.3666666666667)  # pooled roots would give another estimand


def test_holm_preserves_six_tests_including_ties_and_conservative_ones():
    p=dict(zip('abcdef',[.001,.01,.03,.04,.2,1.]))
    assert holm(p)==pytest.approx(dict(zip('abcdef',[.006,.05,.12,.12,.4,1.])))
    assert holm(dict(reversed(list(p.items()))))==holm(p)


def test_degenerate_reference_cannot_manufacture_significance():
    for effect in (0.,.25):
        result=studentized_reference([effect],[0.],np.full((100,1),effect),np.zeros((100,1)))[0]
        assert result['p_raw']==1 and result['T_obs'] is None
    # A degenerate resample is retained as a tail draw, never silently dropped.
    se=np.ones((100,1));se[:2]=0
    result=studentized_reference([.3],[.1],np.full((100,1),.3),se)[0]
    assert result['exceedances']==2 and result['degenerate_draws']==2
    assert result['status']=='bootstrap_reference_unstable' and result['p_raw']==1


def test_numeric_results_ignore_input_order_and_bootstrap_batch_size():
    metadata,means=fixture()
    a=RootTable.parse(metadata,means,require_main_shape=False)
    b=RootTable.parse(list(reversed(metadata)),{k:dict(reversed(list(v.items()))) for k,v in reversed(list(means.items()))},require_main_shape=False)
    first=analyze_roots(a,'B1',draws=999,batch_size=17)
    assert analyze_roots(b,'B1',draws=999,batch_size=31)==first
    with pytest.raises(ValueError):RootTable.parse(metadata,means)
    means.pop(next(iter(means)))
    with pytest.raises(ValueError):RootTable.parse(metadata,means,require_main_shape=False)


def test_bounded_primary_matches_independent_widths_and_does_not_use_zero_se_rule():
    import math
    result=bounded_reference(.5,1/48)
    assert result['p_raw']==pytest.approx(2*math.exp(-6.))
    assert result['ci95_bounded_pointwise']==pytest.approx([.5-.39205014,.5+.39205014],abs=1e-8)
    assert result['ci95_bounded_simultaneous_global6_bonferroni']==pytest.approx([.5-.47787023,.5+.47787023],abs=1e-8)
    ood=bounded_reference(0.,1/24)
    assert ood['ci95_bounded_pointwise']==pytest.approx([-.55444262,.55444262],abs=1e-8)
    assert bounded_reference(.9,1/48)['p_raw']<.05
    assert bounded_reference(0.,1/48)['p_raw']==1.


@pytest.mark.parametrize("delta,weight", [(np.bool_(True),1/48),(.2,np.bool_(True)),(True,1/48),(.2,True)])
def test_bounded_helper_rejects_boolean_effect_or_weight(delta,weight):
    with pytest.raises(ValueError):bounded_reference(delta,weight)


def test_small_exact_independent_null_enumeration_respects_bounded_p():
    from itertools import product
    from math import prod
    # Two strata have means+.5 and-.5; their equal-weight macro mean is0.
    # Enumerate all32 possible independent root vectors, with unequal counts.
    mass={alpha:0. for alpha in (.01,.05,.1,.2,.5)}
    for signs in product((-1.,1.),repeat=5):
        probability=prod((.75 if sign==1 else .25) if i<2 else (.25 if sign==1 else .75)
                         for i,sign in enumerate(signs))
        effect=sum(signs[:2])/4+sum(signs[2:])/6
        p=bounded_reference(effect,5/24)['p_raw']
        for alpha in mass:
            if p<=alpha:mass[alpha]+=probability
    assert all(probability<=alpha+1e-12 for alpha,probability in mass.items())


def test_small_unseen_loss_gain_is_not_confirmed_by_narrow_bootstrap_interval():
    metadata=[];means={}
    for domain,levels,n in (('ID',(1,2,3),16),('OOD',(2,3),12)):
        for level in levels:
            for i in range(n):
                case=f'{domain}-{level}-{i}'
                metadata.append(dict(case_id=case,lineage_group=case,domain=domain,level=level))
                means[case]={condition:.95 for condition in CONDITION_REPEATS}
                means[case]['B4-M1']=.969+.0001*i/n
    result=analyze_roots(RootTable.parse(metadata,means),'B1',draws=999)
    for domain in result['domains'].values():
        comparison=domain['contrasts']['architecture-M1']
        assert comparison['ci95_bootstrap_pointwise_nominal'][0]>0
        assert comparison['bootstrap_reference_diagnostic']['p_holm_global6_not_confirmatory']<.05
        assert comparison['p_raw']==1 and not comparison['reject_primary_alpha05']
        assert comparison['ci95_bounded_simultaneous_global6_bonferroni'][0]<0
