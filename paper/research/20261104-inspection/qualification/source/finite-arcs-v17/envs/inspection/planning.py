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


AIRSPEED = 17.


def fenced(path, bound=538.):
    return max(float(np.abs(p.samples[:,1:3]).max()) for p in path) <= bound


def recovery_options(start, home_y, wind):
    """All fenced finite CSC recoveries, each already smoothed into its tail."""
    first = TimedPath((Primitive(start,40.,0.),),wind,AIRSPEED)
    start = first.path[-1].end
    finish = Pose(-40.,home_y-80.,0.)
    for geometry in csc_connectors(start,finish,100.):
        middle = TimedPath(geometry,wind,AIRSPEED)
        last = TimedPath((Primitive(geometry[-1].end,40.,0.),),wind,AIRSPEED)
        if fenced((first,middle,last)):
            prefix = TimedPath.concatenate((first,middle,last))
            tail = TimedPath((Primitive(Pose(0.,home_y-80.,0.),2*math.pi*80.,1/80.),),wind,AIRSPEED)
            # Include the first2s of the fixed home circle in the finite return
            # so its straight->circle join can be smoothed before the tail.
            try:
                joined = smooth_own_joins(TimedPath.concatenate((prefix,tail.slice(0.,2.))))
                rotated = TimedPath.concatenate((tail.slice(2.,tail.duration),tail.slice(0.,2.)))
            except ValueError:
                continue
            if fenced((joined,rotated),544.):
                yield joined,rotated


def recovery(start,home_y,wind):
    result=next(recovery_options(start,home_y,wind),None)
    if result is None:raise ValueError("no fenced analytic recovery candidate")
    return result


def initial_program(home_y, wind):
    start = Pose(-250.,home_y-80.,0.)
    survey = smooth_own_joins(TimedPath(capsule_from_bottom(start,home_y,laps=5),wind,AIRSPEED))
    returning,tail = recovery(Pose(*survey.at(survey.duration)[:3]),home_y,wind)
    prefix = smooth_own_joins(TimedPath.concatenate((survey,returning)))
    return RouteProgram(prefix,tail,((survey.duration,"recovery_start"),(prefix.duration,"tail_start")))


def history_prefix(program, until):
    if until <= program.prefix.duration:
        return program.prefix.slice(0.,until)
    rows = [np.r_[t,program.at(float(t))] for t in np.r_[np.arange(0.,until,.05),until]]
    return TimedPath.from_samples(rows,program.prefix.wind,AIRSPEED)


def reference_curvature(program, stamp):
    sample = program.at(stamp)
    wind = program.prefix.wind
    _,unit_bank = course_kinematics(sample[2],1.,wind,AIRSPEED)
    return math.tan(sample[4])/math.tan(unit_bank)


def sample_acceleration(sample, wind):
    course,speed,bank = sample[2:]
    tangent = np.array([math.cos(course),math.sin(course)])
    normal = np.array([-tangent[1],tangent[0]])
    _,unit_bank = course_kinematics(course,1.,wind,AIRSPEED)
    curvature = math.tan(bank)/math.tan(unit_bank)
    parallel,cross = float(wind@tangent),float(wind@normal)
    q = math.sqrt(AIRSPEED**2-cross**2)
    return cross*(1+parallel/q)*curvature*speed*tangent+speed**2*curvature*normal


def smooth_own_joins(path):
    """Replace finite own-route curvature jumps by bounded C2 geometry.

    Endpoints, velocity and acceleration are preserved. This does not acquire a
    target circle or alter real aircraft state. Result still needs qualification.
    """
    jumps = np.flatnonzero(np.abs(np.diff(path.samples[:,5]))>.04)+1
    if not len(jumps):
        return path
    windows = []
    for i in jumps:
        a,b = max(0.,path.samples[i,0]-1.),min(path.duration,path.samples[i,0]+1.)
        if windows and a<=windows[-1][1]+1e-8:
            windows[-1]=(windows[-1][0],b)
        else:windows.append((a,b))
    parts,until = [],0.
    for a,b in windows:
        if a>until+1e-8:parts.append(path.slice(until,a))
        start,finish = path.at(a),path.at(b)
        p0,p1 = start[:2],finish[:2]
        v0 = start[3]*np.array([math.cos(start[2]),math.sin(start[2])])
        v1 = finish[3]*np.array([math.cos(finish[2]),math.sin(finish[2])])
        a0,a1 = sample_acceleration(start,path.wind),sample_acceleration(finish,path.wind)
        seconds = b-a
        matrix = np.array([[seconds**3,seconds**4,seconds**5],
            [3*seconds**2,4*seconds**3,5*seconds**4],[6*seconds,12*seconds**2,20*seconds**3]])
        c3,c4,c5 = np.linalg.solve(matrix,np.vstack((p1-p0-v0*seconds-.5*a0*seconds**2,
            v1-v0-a0*seconds,a1-a0)))
        rows = []
        for t in np.r_[np.arange(0.,seconds-1e-8,.05),seconds]:
            xy = p0+v0*t+.5*a0*t*t+c3*t**3+c4*t**4+c5*t**5
            velocity = v0+a0*t+3*c3*t*t+4*c4*t**3+5*c5*t**4
            acceleration = a0+6*c3*t+12*c4*t*t+20*c5*t**3
            air = velocity-path.wind
            va = float(np.linalg.norm(air))
            rate = float(air[0]*acceleration[1]-air[1]*acceleration[0])/(va*va)
            bank = -math.atan(va*rate/9.81)
            if not 16<=va<=22 or abs(bank)>math.radians(30):
                raise ValueError(f"own-route C2 join exceeds development envelope: Va={va},bank={math.degrees(bank)}")
            rows.append(np.r_[t,xy,math.atan2(velocity[1],velocity[0]),np.linalg.norm(velocity),bank])
        # Use the original endpoint samples, avoiding quadrature/interpolation
        # roundoff becoming a synthetic command jump at the window boundary.
        rows[0],rows[-1] = np.r_[0.,start],np.r_[seconds,finish]
        parts.append(TimedPath.from_samples(rows,path.wind,AIRSPEED))
        until=b
    if until<path.duration-1e-8:parts.append(path.slice(until,path.duration))
    return TimedPath.concatenate(parts)


def ramped_work(sample, first_k, last_k, seconds, wind):
    """Finite, explicitly integrated curvature ramp then fixed arc or line.

    This is a kinematic reference. Qualification must measure the actual
    controller/aircraft, including the entire later return and tail seam.
    """
    if not (0 < seconds <= 40 and all(math.isfinite(v) for v in (first_k,last_k,seconds))
            and max(abs(first_k),abs(last_k)) <= 1/70.):
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
        if abs(bank)>math.radians(30):
            raise ValueError("ramped reference exceeds shared nominal bank command envelope")
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
        runout = ramped_work(sample,reference_curvature(old,request.commit_at),0.,2.,wind)
        for returning,tail in recovery_options(Pose(*runout.at(runout.duration)[:3]),request.homes[request.drone],wind):
            yield TimedPath.concatenate((runout,returning)),tail,request.commit_at,request.commit_at,((2.,"recovery_start"),)
        return
    if request.kind == "survey":
        lead_end = request.commit_at+2.
        lead = history_prefix(old,lead_end).slice(request.commit_at,lead_end)
        start = Pose(*lead.at(lead.duration)[:3])
        goal = Pose(-250.,request.region_y-80.,0.)
        for geometry in csc_connectors(start,goal,100.):
            approach = TimedPath(geometry,wind,AIRSPEED)
            if not fenced((approach,)):
                continue
            survey = TimedPath(capsule_from_bottom(geometry[-1].end,request.region_y,
                                laps=max(1,math.ceil(request.duration/(1502/AIRSPEED)))) ,wind,AIRSPEED)
            for returning,tail in recovery_options(survey.path[-1].end,request.homes[request.drone],wind):
                try:complete=smooth_own_joins(TimedPath.concatenate((lead,approach,survey,returning)))
                except ValueError:continue
                yield (complete,tail,lead_end+approach.duration,lead_end+approach.duration+request.duration,
                       ((lead.duration+approach.duration+survey.duration,"recovery_start"),))
        return
    track = request.track
    if (track is None or track.consumer_drone != request.drone or track.available_at > request.now
            or track.measured_at > track.available_at or not track.evidence_ids):
        raise ValueError("consumer-scoped causally available observed seed required")
    # Fixed geometry from the old future gate, never an exact target-circle
    # acquisition. Forecasts rank navigation; actual frames alone serve jobs.
    seconds = min(8.,max(2.,request.duration))
    start_k = reference_curvature(old,request.commit_at)
    choices = [start_k,1/70.,-1/70.,1/80.,-1/80.,0.]
    candidates = []
    for curvature in dict.fromkeys(round(k,9) for k in choices):
        if (request.own_states and request.own_states[request.drone].airspeed>17.5
                and abs(curvature)>1/80.+1e-8 and abs(curvature)>abs(start_k)+1e-8):
            continue
        try:preview = ramped_work(sample,start_k,curvature,32.,wind)
        except ValueError:continue
        work = preview.slice(0.,seconds)
        forecast = optical_rank(preview,track,request.commit_at,start_k,curvature)
        candidates.append((forecast,work,curvature))
    # Only two complete recovery/fleet checks are budgeted. A bounded-library
    # refusal is recorded; it is not a claim that every physical route fails.
    for rank,work,curvature in sorted(candidates,key=lambda row:row[0])[:2]:
        runout = ramped_work(work.at(work.duration),curvature,0.,2.,wind)
        for returning,tail in recovery_options(Pose(*runout.at(runout.duration)[:3]),request.homes[request.drone],wind):
            complete = TimedPath.concatenate((work,runout,returning))
            yield complete,tail,request.commit_at,request.commit_at+seconds,((seconds,"work_end"),(seconds+2.,"recovery_start"))


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
    examined=0
    for work,tail,start,finish,seams in candidate_work(request,old):
        examined+=1
        combined = TimedPath.concatenate((prefix,work))
        if combined.duration > 565 or not fenced((combined,tail)):
            last_reason = "deadline/range/fence screen"
            continue
        if request.remaining_range_m and (len(request.remaining_range_m)!=4 or
                request.remaining_range_m[request.drone] < (combined.duration-request.now+20.)*27.+1000.):
            last_reason = "insufficient measured remaining range for complete return and reserve"
            continue
        candidate = RouteProgram(combined,tail,tuple((request.commit_at+t,label) for t,label in seams)
                                 +((combined.duration,"tail_start"),))
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
        geometry["complete_recovery_candidates_examined"]=examined
        return PlanProposal(request.serial,request.revision,request.commit_at,request.drone,request.tokens,
                            candidate,start,finish,geometry,"candidate admitted by development geometry screen")
    return PlanProposal(request.serial,request.revision,request.commit_at,request.drone,request.tokens,
                        None,None,None,None,last_reason)
