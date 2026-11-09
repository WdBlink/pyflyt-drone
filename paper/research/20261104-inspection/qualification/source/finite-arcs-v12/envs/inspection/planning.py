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
from envs.inspection.paths import (Pose, Primitive, RouteProgram, TimedPath, capsule_from_bottom,
                                   course_kinematics, csc_connectors, fleet_clearance)


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


def reference_curvature(program, stamp):
    a,b = program.at(max(0.,stamp-.025)),program.at(stamp+.025)
    return wrap_angle(b[2]-a[2])/((.05 if stamp >= .025 else stamp+.025)*max(1.,(a[3]+b[3])/2))


def ramped_work(sample, first_k, last_k, seconds, wind):
    """Finite, explicitly integrated curvature ramp then fixed arc or line.

    This is a kinematic reference. Qualification must measure the actual
    controller/aircraft, including the entire later return and tail seam.
    """
    if not (0 < seconds <= 12 and all(math.isfinite(v) for v in (first_k,last_k,seconds))
            and max(abs(first_k),abs(last_k)) <= 1/75.):
        raise ValueError("bounded work geometry required")
    xy,heading = np.array(sample[:2]),float(sample[2])
    ramp = min(2.,seconds)
    rows,stamp = [],0.
    def k_at(t):
        u = min(1.,t/ramp)
        return first_k+(last_k-first_k)*(3*u*u-2*u*u*u)
    while stamp < seconds-1e-9:
        step = min(.05,seconds-stamp)
        speed,bank = course_kinematics(heading,k_at(stamp),wind,AIRSPEED)
        rows.append(np.r_[stamp,xy,heading,speed,bank])
        mid_h = heading+speed*k_at(stamp+step/2)*step/2
        mid_speed,_ = course_kinematics(mid_h,k_at(stamp+step/2),wind,AIRSPEED)
        xy += mid_speed*step*np.array([math.cos(mid_h),math.sin(mid_h)])
        heading += mid_speed*k_at(stamp+step/2)*step
        stamp += step
    speed,bank = course_kinematics(heading,last_k,wind,AIRSPEED)
    rows.append(np.r_[stamp,xy,heading,speed,bank])
    return TimedPath.from_samples(rows,wind,AIRSPEED)


def optical_rank(work, track, starts_at, old_k, new_k):
    rows = work.samples[::5]
    target = np.asarray(track.position)+np.outer(starts_at+rows[:,0]-track.measured_at,track.velocity)
    distance = np.linalg.norm(rows[:,1:3]-target,axis=1)
    age = starts_at+rows[:,0]-track.measured_at
    # A heuristic uncertainty penalty, not Gaussian support or a visibility
    # guarantee. Keep the actual measurement mode/covariance/innovation visible.
    uncertainty = 2.448*math.sqrt(max(track.covariance_xy))+.5*track.innovation_m+age*.5
    deficit = np.maximum(0.,distance+6.+.3+uncertainty-(96.-2.-.9)*math.tan(math.radians(50)))
    blind = deficit > 0
    longest = current = 0
    for value in blind:
        current = current+1 if value else 0
        longest = max(longest,current)
    # Preserve present views, then reduce the worst forecast gap and geometric
    # offset. Small curvature changes are a tie-break, not a centering demand.
    return (float(deficit[:5].sum()),longest,float(deficit.sum()),
            float(np.mean(distance**2))+.05*abs(new_k-old_k)*80)


@dataclass(frozen=True)
class ObservedTrack:
    consumer_drone: int
    measured_at: float
    available_at: float
    position: tuple[float,float]
    velocity: tuple[float,float]
    evidence_ids: tuple[str,...]
    covariance_xy: tuple[float,float] = (16.,16.)
    sensor_mode: str = "survey"
    innovation_m: float = 0.

    def __post_init__(self):
        values = (*self.position,*self.velocity,self.measured_at,self.available_at,*self.covariance_xy,self.innovation_m)
        if (type(self.consumer_drone) is not int or not 0 <= self.consumer_drone < 4
                or len(self.position) != 2 or len(self.velocity) != 2 or not all(math.isfinite(x) for x in values)
                or not 0 <= self.measured_at <= self.available_at or not 1 <= len(self.evidence_ids) <= 8
                or any(not isinstance(x,str) or not x for x in self.evidence_ids)
                or len(self.covariance_xy)!=2 or min(self.covariance_xy)<0 or self.innovation_m<0
                or self.sensor_mode not in ("survey","inspect")):
            raise ValueError("finite consumer-scoped measurement provenance required")

    def predict(self, stamp):
        if stamp < self.measured_at:
            raise ValueError("causal prediction required")
        return np.asarray(self.position)+np.asarray(self.velocity)*(stamp-self.measured_at)


@dataclass(frozen=True)
class OwnState:
    at: float
    position: tuple[float,float,float]
    attitude: tuple[float,float,float]
    body_rates: tuple[float,float,float]
    world_velocity: tuple[float,float,float]
    airspeed: float
    controller_integrals: tuple[float,float]

    def __post_init__(self):
        groups = (self.position,self.attitude,self.body_rates,self.world_velocity)
        if (any(len(v)!=3 for v in groups) or len(self.controller_integrals)!=2
                or not all(math.isfinite(v) for v in (self.at,self.airspeed,*self.controller_integrals,
                                                       *(x for g in groups for x in g)))
                or self.at<0 or self.airspeed<0):
            raise ValueError("finite read-only own navigation/controller snapshot required")


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
    own_states: tuple[OwnState,...] = ()
    remaining_range_m: tuple = ()


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
    # Fixed geometry from the old future gate, never an exact target-circle
    # acquisition. Forecasts rank navigation; actual frames alone serve jobs.
    seconds = min(8.,max(2.,request.duration))
    start_k = reference_curvature(old,request.commit_at)
    choices = [start_k,1/80.,-1/80.,1/90.,-1/90.,0.]
    candidates = []
    for curvature in dict.fromkeys(round(k,9) for k in choices):
        work = ramped_work(sample,start_k,curvature,seconds,wind)
        forecast = optical_rank(work,track,request.commit_at,start_k,curvature)
        candidates.append((forecast,work,curvature))
    # Only two complete recovery/fleet checks are budgeted. A bounded-library
    # refusal is recorded; it is not a claim that every physical route fails.
    for rank,work,curvature in sorted(candidates,key=lambda row:row[0])[:2]:
        runout = ramped_work(work.at(work.duration),curvature,0.,2.,wind)
        try:
            returning,tail = recovery(Pose(*runout.at(runout.duration)[:3]),request.homes[request.drone],wind)
            complete = TimedPath.concatenate((work,runout,returning))
        except ValueError:
            continue
        yield complete,tail,request.commit_at,request.commit_at+seconds,None


def plan(request):
    if len(request.programs) != 4 or len(request.tokens) != 4 or not request.now < request.commit_at:
        raise ValueError("four-aircraft snapshot and future capture required")
    if request.own_states and (len(request.own_states)!=4 or any(s.at!=request.now for s in request.own_states)):
        raise ValueError("four synchronous own snapshots required")
    if request.remaining_range_m and (len(request.remaining_range_m)!=4
            or not all(math.isfinite(v) and v>=0 for v in request.remaining_range_m)):
        raise ValueError("four finite measured remaining ranges required")
    old = request.programs[request.drone]
    prefix = history_prefix(old,request.commit_at)
    last_reason = "no finite candidate"
    for work,tail,start,finish,_ in candidate_work(request,old):
        combined = TimedPath.concatenate((prefix,work))
        if combined.duration > 565 or not fenced((combined,tail)):
            last_reason = "deadline/range/fence screen"
            continue
        if request.remaining_range_m and (len(request.remaining_range_m)!=4 or
                request.remaining_range_m[request.drone] < (combined.duration-request.now+20.)*27.+1000.):
            last_reason = "insufficient measured remaining range for complete return and reserve"
            continue
        candidate = RouteProgram(combined,tail)
        programs = list(request.programs)
        programs[request.drone] = candidate
        try:
            geometry = fleet_clearance(programs,sample_dt=.1)
        except ValueError as error:
            last_reason = str(error)
            continue
        if (geometry["continuous_lower_bound_horizontal_m"] < 27
                or geometry["continuous_abs_xy_upper_bound_m"] > 544):
            last_reason = "traffic occupation conflicts with committed continuations"
            continue
        return PlanProposal(request.serial,request.revision,request.commit_at,request.drone,request.tokens,
                            candidate,start,finish,geometry,"candidate admitted by development geometry screen")
    return PlanProposal(request.serial,request.revision,request.commit_at,request.drone,request.tokens,
                        None,None,None,None,last_reason)
