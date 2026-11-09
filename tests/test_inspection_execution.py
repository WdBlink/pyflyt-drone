from dataclasses import replace

import numpy as np
import pytest

from envs.inspection.execution import FlightExecutor, program_hash
from envs.inspection.mission import InspectionMission
from envs.inspection.planning import ObservedTrack, PlanProposal, PlanRequest, initial_program, plan
from envs.inspection.protocol import Observation
from envs.inspection.reference import ReferencePolicy


HOMES = (-300.,-100.,100.,300.)


@pytest.fixture
def executor():
    value = FlightExecutor(HOMES,(72.,80.,88.,96.),(0.,0.),start_planner=False)
    try:
        yield value
    finally:
        value.close()


def sample(i,candidate,stamp,xy=(0.,0.),aircraft=(0.,0.)):
    return Observation(f"e-{i}-{stamp}",candidate,i,stamp,stamp+.5,xy,"inspection_vehicle","dynamic",
                       .95,(1.,1.),"inspect",aircraft)


def test_observed_seed_must_have_causal_consumer_scope(executor):
    with pytest.raises(ValueError):
        ObservedTrack(0,2.,1.,(0.,0.),(0.,0.),("e",))
    with pytest.raises(ValueError):
        ObservedTrack(True,0.,0.,(0.,0.),(0.,0.),("e",))
    request = PlanRequest(1,0,0.,2.,1,((0,None),)*4,"observe",30.,None,
                          ObservedTrack(0,0.,0.,(-214.,-90.),(1.5,0.),("e",)),
                          executor.programs,HOMES,(0.,0.))
    with pytest.raises(ValueError,match="consumer-scoped"):
        plan(request)
    with pytest.raises(ValueError,match="consumer-scoped"):
        plan(replace(request,track=ObservedTrack(1,0.,1.,(-214.,-90.),(1.5,0.),("e",))))


def test_finite_observation_plan_has_complete_shared_exit_continuation(executor):
    request = PlanRequest(1,0,0.,2.,1,((0,None),)*4,"observe",30.,None,
                          ObservedTrack(1,0.,0.,(-214.,-90.),(1.5,0.),("e",)),
                          executor.programs,HOMES,(0.,0.))
    proposal = plan(request)
    assert proposal.program is not None
    assert proposal.geometry["continuous_lower_bound_horizontal_m"] >= 27
    assert proposal.program.prefix.duration < 565
    for stamp in np.linspace(0,2,30):
        assert np.allclose(proposal.program.at(float(stamp))[:2],executor.programs[1].at(float(stamp))[:2])
    # A stopped coordinator/planner leaves a closed target-independent tail.
    p = proposal.program
    assert np.allclose(p.at(600)[:2],p.at(600+p.tail.duration)[:2])


def test_cancel_invalidates_ready_proposal_without_releasing_physical_program(executor):
    response = executor.submit(0,"survey",{"region_id":"stripe-1","duration":20},0,0.)
    job_id = response["job_id"]
    before = tuple(map(program_hash,executor.programs))
    executor.ready = (PlanProposal(1,0,2.,0,executor.tokens(),executor.programs[0],2.,22.,{},"fixture"),job_id)
    assert executor.cancel(job_id,1,1.)["status"] == "accepted"
    nav = [np.array([[0.,0.,0.],[0.,0.,0.],[18.,0.,0.],[-250.,h-80.,a]])
           for h,a in zip(HOMES,(72.,80.,88.,96.))]
    executor.advance(2.,nav,[0.]*4)
    assert executor.ready is None
    assert tuple(map(program_hash,executor.programs)) == before
    assert executor.book.jobs[job_id].status == "cancelled"
    assert executor.audit[-1]["type"] == "proposal_invalidated"


def test_local_knowledge_does_not_cross_aircraft_without_delivered_seed(executor):
    obs = sample(0,"c",1.,xy=(20.,10.))
    executor.local_measurement(obs,1.)
    assert "c" in executor.local[0]
    assert all("c" not in executor.local[i] for i in (1,2,3))
    with pytest.raises(ValueError):
        executor.submit(1,"observe",{"candidate_id":"c","duration":30},0,1.,seed=obs)
    assert executor.book.versions[1] == 0 and not executor.local[1]
    ack = executor.submit(1,"observe",{"candidate_id":"c","duration":30},0,1.5,seed=obs)
    assert ack["status"] == "accepted"
    assert np.array_equal(executor.local[1]["c"].velocity,[0.,0.])
    assert executor.support[1]["c"] == (obs.evidence_id,1.5)


def test_physics_advances_with_no_planner_reply():
    mission = InspectionMission(0,reference=True,start_planner=False)
    try:
        before = mission.flight.world.state(0)[3].copy()
        ack = mission.tool("assign_survey",{"drone_id":0,"region_id":"stripe-1","duration":100,
            "expected_version":0,"rule_id":"static-0","task_card_version":1})
        assert ack["status"] == "accepted"
        for _ in range(120):
            mission.step()
        assert mission.flight.physics_audits == 240
        assert np.linalg.norm(mission.flight.world.state(0)[3]-before) > 15
        assert mission.jobs.jobs[ack["job_id"]].status == "queued"
        assert not mission.flight.violations
    finally:
        mission.close()


def test_handoff_cannot_cancel_a_successor_source_version(executor):
    seed = sample(0,"c",0.)
    source_ack = executor.submit(0,"observe",{"candidate_id":"c","duration":30},0,.5,seed=seed)
    executor.local_measurement(sample(0,"c",1.),1.)
    source = executor.book.jobs[source_ack["job_id"]]
    receiver_ack = executor.submit(1,"observe",{"candidate_id":"c","duration":30},0,1.5,seed=seed)
    receiver = executor.book.jobs[receiver_ack["job_id"]]
    executor.mark_handoff(receiver,source)
    executor.cancel(source.job_id,1,1.6)
    new_ack = executor.submit(0,"observe",{"candidate_id":"c","duration":30},2,1.7,seed=seed)
    executor.local_measurement(sample(0,"c",2.),2.)
    executor.local_measurement(sample(1,"c",2.5),2.5)
    assert executor.book.jobs[new_ack["job_id"]].status == "running"
    assert source.status == "cancelled"
    assert "handoff_effective_at" not in receiver.params
    assert receiver.params["handoff_source_stale_at"] == 2.5


def test_old_or_duplicate_local_samples_do_not_acquire_new_jobs(executor):
    seed = sample(0,"c",0.)
    ack = executor.submit(0,"observe",{"candidate_id":"c","duration":30},0,.5,seed=seed)
    job = executor.book.jobs[ack["job_id"]]
    executor.local_measurement(sample(0,"c",.25),.75)
    assert job.status == "queued"
    obs = sample(0,"c",1.)
    executor.local_measurement(obs,1.)
    executor.local_measurement(obs,2.)
    executor.local_measurement(sample(0,"c",.9),2.)
    assert executor.service_samples[job.job_id] == [1.]
    assert executor.support[0]["c"] == (obs.evidence_id,1.)


def test_capture_checks_height_and_no_reply_cannot_hold_pending_forever(executor):
    import time
    ack = executor.submit(0,"survey",{"region_id":"stripe-1","duration":20},0,0.)
    job_id = ack["job_id"]
    nav = []
    for p,a in zip(executor.programs,(72.,80.,88.,96.)):
        x,y,h,speed,bank = p.at(2.)
        nav.append(np.array([[0.,0.,0.],[bank,0.,h],[speed,0.,0.],[x,y,a]]))
    nav[0][3,2] = 92
    executor.ready = (PlanProposal(1,0,2.,0,executor.tokens(),executor.programs[0],2.,22.,{},"fixture"),job_id)
    executor.advance(2.,nav,[0.]*4)
    assert executor.revision == 0
    assert executor.audit[-1]["type"] == "capture_envelope_rejected"
    request = PlanRequest(2,0,2.,4.,0,executor.tokens(),"survey",20, -100.,None,executor.programs,HOMES,(0.,0.))
    executor.pending = (request,job_id)
    executor.pending_wall_deadline = time.monotonic()-1
    executor.advance(91.,nav,[0.]*4)
    assert executor.pending is None
    assert executor.infrastructure_errors[-1]["type"] == "planner_no_reply_timeout"
    # Finite complete owned continuations remain available after the fault.
    assert all(np.isfinite(p.at(600)).all() for p in executor.programs)


def test_report_dedup_covers_same_batch_and_rolls_back_rejected_aliases():
    policy = ReferencePolicy()
    observations = []
    for i in (0,1):
        for stamp,aircraft in ((1.,(0.,0.)),(2.,(21.,0.))):
            obs = sample(i,f"c{i}",stamp,xy=(100.,100.),aircraft=aircraft)
            observations.append(replace(obs,object_type="static",estimated_class="road_obstruction").public())
    state = {"observations":observations,"jobs":[],"aircraft":[],"sim_time":3.,"report_slots_used":0,
             "task_card":{"report_budget":10}}
    commands = policy.decide(state)
    assert len(commands) == 1 and commands[0][0] == "submit_or_revise_report"
    name,args = commands[0]
    assert "c1" not in policy.reports
    policy.result(name,args,{"status":"rejected"})
    assert not policy.pending_reports and not policy.report_aliases and not policy.reports
    commands = policy.decide({**state,"observations":[]})
    assert len(commands) == 1
    name,args = commands[0]
    policy.result(name,args,{"status":"accepted"})
    assert policy.reports["c0"] == policy.reports["c1"]


def test_observe_can_renew_during_acquisition_without_waiting_for_ideal_arrival(executor,monkeypatch):
    seed=sample(0,"c",0.)
    ack=executor.submit(0,"observe",{"candidate_id":"c","duration":180},0,.5,seed=seed)
    executor.local_measurement(sample(0,"c",1.),1.)
    job=executor.book.jobs[ack["job_id"]]
    executor.refreshed_at[job.job_id]=2.
    job.params["planned_service_from"]=40.
    class Alive:
        exitcode=None
    executor.process=Alive()
    requests=[]
    monkeypatch.setattr(executor,"_request",lambda job,now:requests.append((job.job_id,now)))
    nav=[np.array([[0.,0.,0.],[0.,0.,0.],[18.,0.,0.],[-250.,h-80.,a]]) for h,a in zip(HOMES,(72.,80.,88.,96.))]
    try:
        executor.advance(5.,nav,[0.]*4)
        assert requests==[(job.job_id,5.)]
        requests.clear()
        executor.advance(40.,nav,[0.]*4)
        assert requests==[]  # navigation cannot renew an indefinitely stale anchor
        executor.local_measurement(sample(0,"c",40.5),40.5)
        executor.advance(41.,nav,[0.]*4)
        assert requests==[(job.job_id,41.)]
    finally:
        executor.process=None


def test_stale_camera_reacquires_without_creating_a_predicted_measurement(executor):
    seed=sample(0,"c",0.)
    ack=executor.submit(0,"observe",{"candidate_id":"c","duration":180},0,.5,seed=seed)
    assert executor.camera_narrow(0,1.)
    assert not executor.camera_narrow(0,4.)
    assert executor.aim(0,4.) is not None
    assert ack["job_id"] not in executor.service_samples
    executor.local_measurement(sample(0,"c",4.5),4.5)
    assert executor.camera_narrow(0,4.5)
    assert executor.service_samples[ack["job_id"]]==[4.5]


def test_capture_rejects_wrong_world_course_even_with_matching_speed_yaw(executor):
    from envs.inspection.planning import OwnState
    nav=[]
    snapshots=[]
    for p,a in zip(executor.programs,(72.,80.,88.,96.)):
        x,y,h,speed,bank=p.at(2.)
        nav.append(np.array([[0.,0.,0.],[bank,0.,h],[0.,speed,0.],[x,y,a]]))
        snapshots.append(OwnState(2.,(x,y,a),(bank,0.,h),(0.,0.,0.),(0.,speed,0.),18.,(0.,0.)))
    executor.own_states=tuple(snapshots)
    accepted,metrics=executor._capture(1,2.,nav)
    assert not accepted
    assert metrics["course_error_degrees"]==pytest.approx(90.)


@pytest.mark.parametrize("kind",["survey","inspect","observe","exit"])
def test_every_job_kind_requires_the_enlarged_complete_fleet_clearance(monkeypatch,kind):
    import math
    import envs.inspection.planning as planning
    from envs.inspection.paths import Pose,Primitive,RouteProgram,TimedPath
    homes=(0.,31.,120.,220.)
    programs=[]
    for y in homes:
        circle=TimedPath((Primitive(Pose(0.,y-80.,0.),2*math.pi*80.,1/80.),),np.zeros(2),17.)
        programs.append(RouteProgram(circle,circle))
    start=Pose(*programs[0].at(2.)[:3])
    work=TimedPath((Primitive(start,2*math.pi*80.,1/80.),),np.zeros(2),17.)
    def candidate(*args):yield work,work,2.,20.,()
    monkeypatch.setattr(planning,'candidate_work',candidate)
    request=PlanRequest(1,0,0.,2.,0,((0,None),)*4,kind,20.,None,None,tuple(programs),homes,(0.,0.))
    proposal=planning.plan(request)
    #31m ideal spacing passed the old27m screen, but cannot protect15m with
    # two10m physical errors. The same rejection applies to all job modes.
    assert proposal.program is None and 'traffic occupation' in proposal.reason


def test_capture_uses_declared10m_boundary_without_changing_other_domains(executor):
    from envs.inspection.planning import OwnState
    nav=[];snapshots=[]
    for p,a in zip(executor.programs,(72.,80.,88.,96.)):
        x,y,h,speed,bank=p.at(2.)
        nav.append(np.array([[0.,0.,0.],[bank,0.,h],[speed,0.,0.],[x,y,a]]))
        snapshots.append(OwnState(2.,(x,y,a),(bank,0.,h),(0.,0.,0.),(speed,0.,0.),17.,(0.,0.)))
    for offset,expected in ((9.,True),(10.001,False)):
        sample=executor.programs[0].at(2.)
        nav[0][3,0]=sample[0]+offset
        snapshots[0]=replace(snapshots[0],position=tuple(nav[0][3]))
        executor.own_states=tuple(snapshots)
        assert bool(executor._capture(0,2.,nav)[0]) is expected
