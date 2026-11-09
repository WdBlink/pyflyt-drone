import math

import numpy as np
import pytest

from envs.inspection.control import air_heading_for_course, wrap_angle
from envs.inspection.guidance import LocalTrack, moving_orbit_target
from envs.inspection.paths import Pose, Primitive, TimedPath, RouteProgram, csc_connectors, stripe_transfer
from envs.inspection.paths import fleet_clearance


def test_local_track_does_not_rewind_or_alias_measurements():
    sample = np.array([0., 0.])
    track = LocalTrack.first(sample, 0)
    sample[:] = 999
    assert np.array_equal(track.position, [0, 0])
    assert track.update([1, 0], 1)
    before = (track.position.copy(), track.velocity.copy(), track.measured_at)
    assert not track.update([900, 900], 1)
    assert not track.update([900, 900], .5)
    assert all(np.array_equal(a, b) for a, b in zip(before, (track.position, track.velocity, track.measured_at)))
    with pytest.raises(ValueError):
        track.predict(.5)


def test_moving_orbit_air_heading_and_bank_match_differentiated_motion():
    wind, velocity = np.array([-2., 0.]), np.array([1.5, 0.])
    for theta in np.linspace(-math.pi, math.pi, 30):
        radial = np.array([math.cos(theta), math.sin(theta)])
        tangent = np.array([-radial[1], radial[0]])
        target = moving_orbit_target(90*radial, np.zeros(2), velocity, wind)
        air_heading = air_heading_for_course(target.heading, wind, target.airspeed)
        ground = 18*np.array([math.cos(air_heading), math.sin(air_heading)])+wind
        assert abs((ground-velocity)@radial) < 1e-10
        dt = 1e-5
        next_theta = theta+float((ground-velocity)@tangent)/90*dt
        next_target = moving_orbit_target(90*np.array([math.cos(next_theta), math.sin(next_theta)]), np.zeros(2), velocity, wind)
        rate = wrap_angle(air_heading_for_course(next_target.heading, wind, 18)-air_heading)/dt
        assert target.bank_feedforward == pytest.approx(-math.atan(18*rate/9.81), abs=1e-6)


@pytest.mark.parametrize("start,finish", [(Pose(-250,-380,0),Pose(0,220,0)),
                                        (Pose(45,70,1.2),Pose(-20,-300,-2.0)),
                                        (Pose(0,0,0),Pose(80,0,0))])
def test_all_csc_tangents_reach_the_requested_pose(start, finish):
    paths = csc_connectors(start, finish)
    assert paths
    for path in paths:
        assert math.dist((path[-1].end.x, path[-1].end.y), (finish.x, finish.y)) < 1e-6
        assert abs(wrap_angle(path[-1].end.heading-finish.heading)) < 1e-8
        for a, b in zip(path, path[1:]):
            assert a.end == b.start


def test_all_twelve_stripe_transfers_have_correct_port_and_heading():
    for source in (-300,-100,100,300):
        for destination in (-300,-100,100,300):
            if source == destination:
                continue
            path = stripe_transfer(Pose(-250,source-80,0), destination-80)
            assert path[-1].end.x == pytest.approx(-50)
            assert path[-1].end.y == pytest.approx(destination-80)
            assert path[-1].end.heading == pytest.approx(0)


def test_program_tail_is_continuous_and_repeats_after_long_wait():
    circle = TimedPath((Primitive(Pose(0,-80,0),2*math.pi*80,1/80),), wind=(-2.,0.))
    program = RouteProgram(circle, circle)
    assert np.allclose(program.at(circle.duration+3), program.at(101*circle.duration+3), atol=1e-8)
    assert np.allclose(program.at(circle.duration-1e-6)[:2], program.at(circle.duration+1e-6)[:2], atol=1e-4)
    open_line = TimedPath((Primitive(Pose(0,-80,0),20,0),))
    with pytest.raises(ValueError):
        RouteProgram(circle, open_line)


def test_discontinuous_primitives_cannot_bypass_clearance_speed_bound():
    with pytest.raises(ValueError,match="continuous"):
        TimedPath((Primitive(Pose(0,0,0),.1,0),Primitive(Pose(200,0,0),.1,0)))
    with pytest.raises(ValueError):
        TimedPath((Primitive(Pose(0,0,0),0,0),))
    with pytest.raises(ValueError):
        TimedPath((Primitive(Pose(0,0,0),1,0),),dt=math.nan)
    # An arbitrary sampled loop may be continuous but impossibly fast. The
    # screen must refuse the fixed27m/s premise, even with equal tail periods.
    fast = TimedPath.from_samples([[0,0,0,0,20,0],[.01,100,0,0,20,0],[.02,0,0,0,20,0]],(0,0))
    program = RouteProgram(fast,fast)
    with pytest.raises(ValueError,match="Lipschitz"):
        fleet_clearance([program]*4)


def test_equivalent_wrapped_tangents_do_not_interpolate_a_reverse_heading():
    path = TimedPath((Primitive(Pose(0,0,2*math.pi),1,0),Primitive(Pose(1,0,0),1,0)))
    assert abs(wrap_angle(path.at(.025)[2])) < 1e-8
    assert abs(wrap_angle(path.target(.025,[.5,0],80).heading)) < 1e-8


def test_ramp_matches_old_curvature_and_has_no_position_or_bank_jump():
    from envs.inspection.planning import initial_program,reference_curvature,ramped_work
    program=initial_program(0.,np.array([-2.,0.]))
    sample=program.at(2.)
    work=ramped_work(sample,0.,1/80.,8.,np.array([-2.,0.]))
    assert np.allclose(work.at(0),sample,atol=1e-6)
    assert work.samples[:,0].max()==pytest.approx(8.)
    assert np.max(np.abs(np.diff(work.samples[:,5])))<.025
    continuation=ramped_work(work.at(8.),1/80.,1/80.,8.,np.array([-2.,0.]))
    assert np.allclose(continuation.at(0),work.at(8.))
    assert np.isfinite(reference_curvature(program,2.))


def test_vectorized_traffic_geometry_matches_scalar_on_prefix_and_tail():
    from envs.inspection.planning import initial_program
    p=initial_program(-100.,np.array([-2.,0.]))
    stamps=np.r_[0.,2.,p.prefix.duration-1e-6,p.prefix.duration,p.prefix.duration+10.,600.,
                 np.random.default_rng(17).uniform(0,1500,100)]
    assert np.allclose(p.xy_at(stamps),[p.at(float(t))[:2] for t in stamps],atol=1e-10)


@pytest.mark.parametrize("wind",[(0.,0.),(-2.,0.),(2.,0.),(0.,2.)])
def test_seam_gate_uses_current_bank_not_a_cross_seam_curvature_average(wind):
    from envs.inspection.planning import reference_curvature,ramped_work
    start=Pose(-250.,-180.,0.)
    raw=TimedPath((Primitive(start,500.,0.),Primitive(Pose(250.,-180.,0.),math.pi*80.,1/80.)),wind,18.)
    tail=TimedPath((Primitive(raw.path[-1].end,2*math.pi*80.,1/80.),),wind,18.)
    program=RouteProgram(raw,tail)
    seam=raw.samples[np.flatnonzero(np.abs(np.diff(raw.samples[:,5]))>.04)[0]+1,0]
    for offset in (-.01,0.,.01):
        gate=seam+offset
        sample=program.at(gate)
        work=ramped_work(sample,reference_curvature(program,gate),0.,2.,np.asarray(wind))
        assert work.at(0)[4]==pytest.approx(sample[4],abs=1e-10)


@pytest.mark.parametrize("wind",[(0.,0.),(-2.,0.),(2.,0.),(0.,2.),(0.,-2.),(-math.sqrt(2),-math.sqrt(2))])
def test_complete_initial_route_and_home_tail_have_bounded_bank_steps(wind):
    from envs.inspection.planning import initial_program
    program=initial_program(-100.,np.asarray(wind))
    assert np.max(np.abs(np.diff(program.prefix.samples[:,5])))<math.radians(2.)
    assert abs(program.prefix.at(program.prefix.duration)[4]-program.tail.at(0)[4])<1e-9
    assert max(abs(program.prefix.samples[:,5]))<=math.radians(30.)
    assert np.min(np.diff(program.prefix.samples[:,0]))>1e-8
    slope=np.linalg.norm(np.diff(program.prefix.samples[:,1:3],axis=0),axis=1)/np.diff(program.prefix.samples[:,0])
    assert slope.max()<=27.


def test_lateral_correction_does_not_accelerate_and_bank_uses_measured_airdata():
    from envs.inspection.paths import tracking_target,course_kinematics
    wind=np.zeros(2)
    straight=tracking_target([0.,0.,0.,17.,0.],[0.,6.],96.,wind,measured_airspeed=17.)
    assert straight.airspeed==pytest.approx(17.)
    speed,bank=course_kinematics(0.,1/70.,wind,17.)
    turn=tracking_target([0.,0.,0.,speed,bank],[0.,0.],96.,wind,measured_airspeed=18.)
    assert turn.bank_feedforward==pytest.approx(course_kinematics(0.,1/70.,wind,18.)[1])


def test_phase_catchup_respects_turn_budget_in_every_wind_quadrant():
    from envs.inspection.paths import tracking_target,course_kinematics
    for angle in np.linspace(0.,2*math.pi,17)[:-1]:
        wind=2*np.array([math.cos(angle),math.sin(angle)])
        for curvature in (-1/70.,1/70.,0.):
            speed,bank=course_kinematics(0.,curvature,wind,17.)
            target=tracking_target([0.,0.,0.,speed,bank],[-6.,0.],96.,wind,measured_airspeed=17.)
            assert 16<=target.airspeed<=18.5
            assert abs(course_kinematics(0.,curvature,wind,target.airspeed)[1])<=math.radians(29)+1e-9


def test_recovery_search_retains_distinct_complete_alternatives():
    from envs.inspection.planning import recovery_options
    options=list(recovery_options(Pose(-250.,-180.,0.),-100.,np.zeros(2)))
    assert 2<=len(options)<=4
    assert len({round(p.duration,6) for p,t in options})>=2
    for prefix,tail in options:
        RouteProgram(prefix,tail)
        assert np.max(np.abs(prefix.samples[:,1:3]))<=544.
