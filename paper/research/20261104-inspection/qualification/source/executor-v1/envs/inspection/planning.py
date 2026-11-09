"""Pure bounded flight-program construction for an isolated planning process.

Inputs are own navigation programs and one consumer-scoped observed estimate.
There is no mission, simulator, entity, seed, future truth or grading access.
All returned work programs include a physical target-independent home tail.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from envs.inspection.control import wrap_angle
from envs.inspection.guidance import moving_orbit_target
from envs.inspection.paths import (Pose, Primitive, RouteProgram, TimedPath, capsule_from_bottom,
                                   csc_connectors, fleet_clearance)


AIRSPEED = 18.


def fenced(path, bound=538.):
    return max(float(np.abs(p.samples[:,1:3]).max()) for p in path) <= bound


def recovery(start, home_y, wind):
    """Finite CSC family with tangent runouts before/after curvature changes."""
    first = TimedPath((Primitive(start,40.,0.),),wind,AIRSPEED)
    start = first.path[-1].end
    finish = Pose(-40.,home_y-80.,0.)
    for geometry in csc_connectors(start,finish,100.):
        middle = TimedPath(geometry,wind,AIRSPEED)
        last = TimedPath((Primitive(geometry[-1].end,40.,0.),),wind,AIRSPEED)
        if fenced((first,middle,last)):
            prefix = TimedPath.concatenate((first,middle,last))
            tail = TimedPath((Primitive(Pose(0.,home_y-80.,0.),2*math.pi*80.,1/80.),),wind,AIRSPEED)
            return prefix,tail
    raise ValueError("no fenced analytic recovery candidate")


def initial_program(home_y, wind):
    start = Pose(-250.,home_y-80.,0.)
    survey = TimedPath(capsule_from_bottom(start,home_y,laps=5),wind,AIRSPEED)
    returning,tail = recovery(survey.path[-1].end,home_y,wind)
    return RouteProgram(TimedPath.concatenate((survey,returning)),tail)


def history_prefix(program, until):
    if until <= program.prefix.duration:
        return program.prefix.slice(0.,until)
    rows = [np.r_[t,program.at(float(t))] for t in np.r_[np.arange(0.,until,.05),until]]
    return TimedPath.from_samples(rows,program.prefix.wind,AIRSPEED)


def moving_circle(center, velocity, angle, duration, wind, radius=90.):
    center,velocity,wind = [np.asarray(x,dtype=float) for x in (center,velocity,wind)]
    if duration <= 0 or duration > 200 or not math.isfinite(duration):
        raise ValueError("bounded positive circle duration required")
    rows,stamp = [],0.
    while True:
        radial = np.array([math.cos(angle),math.sin(angle)])
        tangent = np.array([-radial[1],radial[0]])
        drift = velocity-wind
        a = float(drift@tangent)
        discriminant = a*a+AIRSPEED*AIRSPEED-float(drift@drift)
        if discriminant <= 0:
            raise ValueError("estimated moving circle infeasible")
        speed = -a+math.sqrt(discriminant)
        position = center+velocity*stamp+radius*radial
        target = moving_orbit_target(position,center+velocity*stamp,velocity,wind,radius=radius)
        ground_velocity = velocity+speed*tangent
        rows.append(np.r_[stamp,position,target.heading,float(np.linalg.norm(ground_velocity)),target.bank_feedforward])
        if stamp >= duration-1e-9:
            break
        step = min(.05,duration-stamp)
        mid = angle+speed/radius*step/2
        mid_tangent = np.array([-math.sin(mid),math.cos(mid)])
        mid_a = float(drift@mid_tangent)
        mid_speed = -mid_a+math.sqrt(mid_a*mid_a+AIRSPEED*AIRSPEED-float(drift@drift))
        angle += mid_speed/radius*step
        stamp += step
    result = TimedPath.from_samples(rows,wind,AIRSPEED)
    if max(abs(result.samples[:,5])) > math.radians(28.5):
        raise ValueError("estimated moving circle exceeds development feedforward envelope")
    return result


def bridge(start_sample, finish_sample, seconds, wind):
    """One finite C1 quintic capture candidate; refuse excessive speed/bank.

    Zero endpoint acceleration is conservative in shape but not C2 with the
    circle. Actual roll/capture still requires physical qualification.
    """
    p0,p1 = np.asarray(start_sample[:2]),np.asarray(finish_sample[:2])
    v0 = start_sample[3]*np.array([math.cos(start_sample[2]),math.sin(start_sample[2])])
    v1 = finish_sample[3]*np.array([math.cos(finish_sample[2]),math.sin(finish_sample[2])])
    t = seconds
    matrix = np.array([[t**3,t**4,t**5],[3*t*t,4*t**3,5*t**4],[6*t,12*t*t,20*t**3]])
    coefficients = np.linalg.solve(matrix,np.vstack((p1-p0-v0*t,v1-v0,np.zeros(2))))
    rows = []
    for stamp in np.r_[np.arange(0.,t,.05),t]:
        c3,c4,c5 = coefficients
        position = p0+v0*stamp+c3*stamp**3+c4*stamp**4+c5*stamp**5
        velocity = v0+3*c3*stamp**2+4*c4*stamp**3+5*c5*stamp**4
        acceleration = 6*c3*stamp+12*c4*stamp**2+20*c5*stamp**3
        air_velocity = velocity-wind
        speed = float(np.linalg.norm(air_velocity))
        if not 16 <= speed <= 22:
            raise ValueError("capture speed outside development envelope")
        rate = float(air_velocity[0]*acceleration[1]-air_velocity[1]*acceleration[0])/(speed*speed)
        bank = -math.atan(speed*rate/9.81)
        if abs(bank) > math.radians(28.5):
            raise ValueError("capture feedforward outside development envelope")
        rows.append(np.r_[stamp,position,math.atan2(velocity[1],velocity[0]),np.linalg.norm(velocity),bank])
    return TimedPath.from_samples(rows,wind,AIRSPEED)


@dataclass(frozen=True)
class ObservedTrack:
    consumer_drone: int
    measured_at: float
    available_at: float
    position: tuple[float,float]
    velocity: tuple[float,float]
    evidence_ids: tuple[str,...]

    def __post_init__(self):
        values = (*self.position,*self.velocity,self.measured_at,self.available_at)
        if (type(self.consumer_drone) is not int or not 0 <= self.consumer_drone < 4
                or len(self.position) != 2 or len(self.velocity) != 2 or not all(math.isfinite(x) for x in values)
                or not 0 <= self.measured_at <= self.available_at or not 1 <= len(self.evidence_ids) <= 8
                or any(not isinstance(x,str) or not x for x in self.evidence_ids)):
            raise ValueError("finite consumer-scoped measurement provenance required")

    def predict(self, stamp):
        if stamp < self.measured_at:
            raise ValueError("causal prediction required")
        return np.asarray(self.position)+np.asarray(self.velocity)*(stamp-self.measured_at)


@dataclass(frozen=True)
class PlanRequest:
    serial: int
    revision: int
    now: float
    commit_at: float
    drone: int
    tokens: tuple
    kind: str
    duration: float
    region_y: float | None
    track: ObservedTrack | None
    programs: tuple[RouteProgram,...]
    homes: tuple[float,...]
    wind: tuple[float,float]


@dataclass(frozen=True)
class PlanProposal:
    serial: int
    revision: int
    commit_at: float
    drone: int
    tokens: tuple
    program: RouteProgram | None
    service_start: float | None
    service_end: float | None
    geometry: dict | None
    reason: str


def candidate_work(request, old):
    sample = old.at(request.commit_at)
    start = Pose(*sample[:3])
    wind = np.asarray(request.wind)
    if request.kind == "exit":
        returning,tail = recovery(start,request.homes[request.drone],wind)
        yield returning,tail,request.commit_at,request.commit_at,request.homes[request.drone]
        return
    if request.kind == "survey":
        goal = Pose(-250.,request.region_y-80.,0.)
        for geometry in csc_connectors(start,goal,100.):
            approach = TimedPath(geometry,wind,AIRSPEED)
            if not fenced((approach,)):
                continue
            survey = TimedPath(capsule_from_bottom(geometry[-1].end,request.region_y,
                                laps=max(1,math.ceil(request.duration/(1502/AIRSPEED)))) ,wind,AIRSPEED)
            returning,tail = recovery(survey.path[-1].end,request.homes[request.drone],wind)
            yield (TimedPath.concatenate((approach,survey,returning)),tail,
                   request.commit_at+approach.duration,request.commit_at+approach.duration+request.duration,request.region_y)
        return
    track = request.track
    if (track is None or track.consumer_drone != request.drone or track.available_at > request.now
            or track.measured_at > track.available_at or not track.evidence_ids):
        raise ValueError("consumer-scoped causally available observed seed required")
    # Receding finite work: every accepted plan already has a complete exit.
    # Failure to refresh it cannot freeze the physics or create future evidence.
    work_seconds = min(20.,max(5.,request.duration))
    relative = sample[:2]-track.predict(request.commit_at)
    near_angle = math.atan2(relative[1],relative[0])
    angles = [near_angle,0.,math.pi/2,math.pi,3*math.pi/2]
    for angle in angles:
        for capture_seconds in (2.,4.,6.):
            # Trial phase advance uses only the old owned program and the
            # received estimate. The ring, not hidden truth, is the own path.
            preview = moving_circle(track.predict(request.commit_at),track.velocity,
                                    angle,capture_seconds+work_seconds,wind)
            ring = preview.slice(capture_seconds,preview.duration)
            try:
                approach = bridge(sample,ring.at(0),capture_seconds,wind)
            except ValueError:
                continue
            finish = Pose(*ring.at(ring.duration)[:3])
            try:
                returning,tail = recovery(finish,request.homes[request.drone],wind)
            except ValueError:
                continue
            yield (TimedPath.concatenate((approach,ring,returning)),tail,
                   request.commit_at+approach.duration,request.commit_at+approach.duration+work_seconds,None)
        # General finite CSC approach for targets that cannot be captured by
        # the short smooth bridge. Recompute lead with causal constant velocity.
        eta = 25.
        geometry = None
        for _ in range(4):
            ring = moving_circle(track.predict(request.commit_at+eta),track.velocity,angle,work_seconds,wind)
            options = csc_connectors(start,Pose(*ring.at(0)[:3]),100.)
            if not options:
                break
            geometry = options[0]
            eta = TimedPath(geometry,wind,AIRSPEED).duration
        if geometry is None:
            continue
        approach = TimedPath(geometry,wind,AIRSPEED)
        # Preserve continuity exactly; remaining CV lead residual is diagnostic
        # acquisition error, not a warp of own pose or a future measurement.
        finish = geometry[-1].end
        radial = np.array([math.cos(angle),math.sin(angle)])
        center = np.array([finish.x,finish.y])-90*radial
        ring = moving_circle(center,track.velocity,angle,work_seconds,wind)
        if abs(wrap_angle(approach.at(approach.duration)[2]-ring.at(0)[2])) > 1e-5:
            continue
        try:
            returning,tail = recovery(Pose(*ring.at(ring.duration)[:3]),request.homes[request.drone],wind)
            work = TimedPath.concatenate((approach,ring,returning))
        except ValueError:
            continue
        yield work,tail,request.commit_at+approach.duration,request.commit_at+approach.duration+work_seconds,None


def plan(request):
    if len(request.programs) != 4 or len(request.tokens) != 4 or not request.now < request.commit_at:
        raise ValueError("four-aircraft snapshot and future capture required")
    old = request.programs[request.drone]
    prefix = history_prefix(old,request.commit_at)
    last_reason = "no finite candidate"
    for work,tail,start,finish,_ in candidate_work(request,old):
        combined = TimedPath.concatenate((prefix,work))
        if combined.duration > 565 or not fenced((combined,tail)):
            last_reason = "deadline/range/fence screen"
            continue
        candidate = RouteProgram(combined,tail)
        programs = list(request.programs)
        programs[request.drone] = candidate
        try:
            geometry = fleet_clearance(programs,sample_dt=.1)
        except ValueError as error:
            last_reason = str(error)
            continue
        if geometry["continuous_lower_bound_horizontal_m"] < 27:
            last_reason = "traffic occupation conflicts with committed continuations"
            continue
        return PlanProposal(request.serial,request.revision,request.commit_at,request.drone,request.tokens,
                            candidate,start,finish,geometry,"candidate admitted by development geometry screen")
    return PlanProposal(request.serial,request.revision,request.commit_at,request.drone,request.tokens,
                        None,None,None,None,last_reason)
