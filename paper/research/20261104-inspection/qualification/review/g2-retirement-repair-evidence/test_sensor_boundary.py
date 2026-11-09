"""Review-only sampler boundary fixtures: synthetic time/geometry, zero physics."""
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import pytest
from envs.inspection.mission import InspectionMission

@pytest.mark.parametrize('narrow', [False, True], ids=['survey','inspect'])
@pytest.mark.parametrize('retired', [False, True], ids=['active','retired'])
def test_retirement_sensor_boundary(monkeypatch,narrow,retired):
    m=InspectionMission(0,reference=True,start_planner=False)
    try:
        clock={'t':359.5}
        monkeypatch.setattr(InspectionMission,'t',property(lambda self:clock['t']))
        m.flight.exit_active[:]=[True,True,True,False]
        m.cells={'controlled-cell':(0.,0.)}
        m.entities=[SimpleNamespace(key='controlled-target',true_class='normal',object_type='static',position=lambda t:(0.,0.))]
        m.entity_bodies={'controlled-target':-1}
        job=SimpleNamespace(params={'candidate_id':'controlled-candidate','object_type':'static'},kind='inspect')
        monkeypatch.setattr(m,'_active',lambda i:job if narrow else None)
        monkeypatch.setattr(m,'camera_origin',lambda i:np.array([0.,0.,96.]))
        monkeypatch.setattr(m,'_visible',lambda *a,**kw:True)
        monkeypatch.setattr(m,'_image_quality',lambda *a,**kw:.95)
        monkeypatch.setattr(m.executor,'aim',lambda i,t:(0.,0.) if narrow else None)
        monkeypatch.setattr(m.executor,'camera_narrow',lambda i,t:narrow)
        monkeypatch.setattr(m.executor,'track_age',lambda i,t:0.)
        local=[]
        monkeypatch.setattr(m.executor,'local_measurement',lambda obs,t:local.append(obs))
        m._sample()
        assert len(m.sensor_audit)==len(m.sensor_opportunities)==len(m.pending)==len(m.pending_coverage)==len(local)==1
        assert m.sensor_audit[0]['mode']==('inspect' if narrow else 'survey')
        assert len(m.background_trials)==(0 if narrow else 1)
        old=asdict(m.pending[0][1].observation)
        counts=lambda:tuple(map(len,(m.sensor_audit,m.sensor_opportunities,m.background_trials,m.pending_coverage,m.pending,local)))
        before=counts()
        clock['t']=360.
        if retired:
            m._events()
            assert 3 in m.retired and not m.flight.exit_active[3]
        m._sample()
        after=counts()
        if retired:
            assert after==before
        else:
            assert after[0]==before[0]+1 and after[1]==before[1]+1 and after[3]==before[3]+1 and after[4]==before[4]+1 and after[5]==before[5]+1
            assert after[2]==before[2]+(0 if narrow else 1)
        m._deliver()
        delivered=next(asdict(o) for o in m.delivered if o.evidence_id==old['evidence_id'])
        assert delivered==old and delivered['measured_at']==359.5 and delivered['arrived_at']==360.
        assert m.flight.world.physics_steps==0
    finally:
        m.close()
