"""Finite analytic forward paths, usable in a planning process without PyFlyt.

No mission, generator, evidence or grader imports. Geometry is not a physical
certificate: measured tracking tubes must be supplied by qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from envs.inspection.control import FlightTarget, wrap_angle


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.x, self.y, self.heading)):
            raise ValueError("finite pose required")


@dataclass(frozen=True)
class Primitive:
    start: Pose
    length: float
    curvature: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (*self.start.__dict__.values(), self.length, self.curvature)) or self.length < 0:
            raise ValueError("finite forward primitive required")

    def pose(self, distance):
        if not math.isfinite(distance) or not -1e-8 <= distance <= self.length+1e-8:
            raise ValueError("distance outside primitive")
        s = max(0., min(self.length, distance))
        p, k = self.start, self.curvature
        h = p.heading+k*s
        if abs(k) < 1e-12:
            return Pose(p.x+s*math.cos(h), p.y+s*math.sin(h), h)
        return Pose(p.x+(math.sin(h)-math.sin(p.heading))/k,
                    p.y+(-math.cos(h)+math.cos(p.heading))/k, h)

    @property
    def end(self):
        return self.pose(self.length)


def primitives(start, pieces):
    result = []
    for curvature, length in pieces:
        if length < 1e-8:
            continue
        result.append(Primitive(start, length, curvature))
        start = result[-1].end
    return tuple(result)


def stripe_transfer(start, destination_y, radius=100.):
    """Two quarter arcs; start must be an eastbound bottom-leg port.

    destination_y is the bottom-leg coordinate, not the stripe center.
    """
    if abs(wrap_angle(start.heading)) > 1e-6 or abs(destination_y-start.y) < 2*radius-1e-8 or radius <= 0:
        raise ValueError("eastbound port and sufficient lateral displacement required")
    sign = math.copysign(1., destination_y-start.y)
    return primitives(start, [(sign/radius, math.pi*radius/2),
                              (0., abs(destination_y-start.y)-2*radius),
                              (-sign/radius, math.pi*radius/2)])


def csc_connectors(start, finish, radius=100.):
    """All four forward circle-straight-circle common-tangent candidates.

    This finite set does not claim to cover every Dubins shortest path.
    Includes longer turns up to2pi, excludes infeasible internal tangents.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("positive finite radius required")
    left = lambda h: np.array([-math.sin(h), math.cos(h)])
    candidates = []
    for first in (-1, 1):
        for last in (-1, 1):
            c0 = np.array([start.x, start.y])+first*radius*left(start.heading)
            c1 = np.array([finish.x, finish.y])+last*radius*left(finish.heading)
            delta = c1-c0
            distance = float(np.linalg.norm(delta))
            shift = (first-last)*radius
            if distance < abs(shift)+1e-8 or distance < 1e-8:
                continue
            heading = math.atan2(delta[1], delta[0])+math.asin(shift/distance)
            angle0 = (first*(heading-start.heading)) % (2*math.pi)
            angle1 = (last*(finish.heading-heading)) % (2*math.pi)
            path = primitives(start, [(first/radius, radius*angle0),
                                      (0., math.sqrt(max(0., distance*distance-shift*shift))),
                                      (last/radius, radius*angle1)])
            if not path or math.dist((path[-1].end.x, path[-1].end.y), (finish.x, finish.y)) > 1e-6:
                raise ArithmeticError("connector endpoint derivation failed")
            candidates.append(path)
    return tuple(sorted(candidates, key=lambda p: sum(s.length for s in p)))


def capsule_from_bottom(start, center_y, *, half_length=250., radius=80., laps=1):
    if abs(wrap_angle(start.heading)) > 1e-6 or abs(start.y-(center_y-radius)) > 1e-6 or not -half_length-1e-6 <= start.x <= half_length:
        raise ValueError("capsule entry must lie on its eastbound bottom leg")
    if laps < 1 or isinstance(laps, bool) or not isinstance(laps, int):
        raise ValueError("positive integer laps required")
    pieces = [(0., half_length-start.x), (1/radius, math.pi*radius),
              (0., 2*half_length), (1/radius, math.pi*radius), (0., start.x+half_length)]*laps
    return primitives(start, pieces)


def course_kinematics(course, curvature, wind, airspeed):
    tangent = np.array([math.cos(course), math.sin(course)])
    normal = np.array([-tangent[1], tangent[0]])
    parallel, cross = float(wind@tangent), float(wind@normal)
    if abs(cross) >= airspeed:
        raise ValueError("path infeasible in crosswind")
    q = math.sqrt(airspeed*airspeed-cross*cross)
    ground_speed = parallel+q
    if ground_speed <= 0:
        raise ValueError("path cannot progress forward")
    derivative = cross*(1+parallel/q)
    acceleration = derivative*curvature*ground_speed*tangent+ground_speed**2*curvature*normal
    air_velocity = ground_speed*tangent-wind
    yaw_rate = float(air_velocity[0]*acceleration[1]-air_velocity[1]*acceleration[0])/(airspeed*airspeed)
    return ground_speed, -math.atan(airspeed*yaw_rate/9.81)


class TimedPath:
    """Bounded time sampling done by the planner; tick lookup is O(log N).

    The follower controls longitudinal phase as well as cross-track error.
    Sample timing is a numerical prediction, not a promise of tracking error.
    """

    def __init__(self, path, wind=(0., 0.), airspeed=20., dt=.05):
        if (not path or len(path) > 64 or not math.isfinite(dt) or dt <= 0
                or not math.isfinite(airspeed) or airspeed <= 0 or sum(p.length for p in path) <= 0):
            raise ValueError("bounded nonempty path and positive time/speed required")
        for a,b in zip(path,path[1:]):
            if (math.dist((a.end.x,a.end.y),(b.start.x,b.start.y)) > 1e-7
                    or abs(wrap_angle(a.end.heading-b.start.heading)) > 1e-7):
                raise ValueError("adjacent primitives must be spatially and tangentially continuous")
        self.path = tuple(path)
        self.wind = np.array(wind, dtype=float)
        if self.wind.shape != (2,) or not np.isfinite(self.wind).all():
            raise ValueError("finite planar wind required")
        self.airspeed = airspeed
        rows, stamp = [], 0.
        for primitive in self.path:
            distance = 0.
            while distance < primitive.length-1e-9:
                pose = primitive.pose(distance)
                speed, bank = course_kinematics(pose.heading, primitive.curvature, self.wind, airspeed)
                rows.append((stamp, pose.x, pose.y, pose.heading, speed, bank))
                step = min(dt, (primitive.length-distance)/speed)
                # Midpoint quadrature for changing wind projection on arcs.
                midpoint = primitive.pose(min(primitive.length, distance+speed*step/2))
                middle_speed, _ = course_kinematics(midpoint.heading, primitive.curvature, self.wind, airspeed)
                step = min(step, (primitive.length-distance)/middle_speed)
                distance = min(primitive.length, distance+middle_speed*step)
                stamp += step
                if len(rows) > 60000:
                    raise ValueError("path exceeds bounded planner sample count")
        end = self.path[-1].end
        speed, bank = course_kinematics(end.heading, self.path[-1].curvature, self.wind, airspeed)
        rows.append((stamp, end.x, end.y, end.heading, speed, bank))
        self.samples = np.array(rows)
        self.samples[:, 3] = np.unwrap(self.samples[:, 3])
        self.samples.flags.writeable = False
        self.duration = stamp

    @classmethod
    def from_samples(cls, rows, wind, airspeed=18.):
        samples = np.asarray(rows, dtype=float)
        if (samples.ndim != 2 or samples.shape[1] != 6 or not 2 <= len(samples) <= 16000
                or not np.isfinite(samples).all() or abs(samples[0,0]) > 1e-8
                or not np.all(np.diff(samples[:,0]) > 0)):
            raise ValueError("bounded finite strictly timed samples required")
        result = object.__new__(cls)
        result.samples = samples.copy()
        result.samples[:,3] = np.unwrap(result.samples[:,3])
        result.samples.flags.writeable = False
        result.wind = np.asarray(wind,dtype=float).copy()
        if result.wind.shape != (2,) or not np.isfinite(result.wind).all():
            raise ValueError("finite planar wind required")
        result.airspeed = airspeed
        result.duration = float(samples[-1,0])
        result.path = ()
        return result

    @classmethod
    def concatenate(cls, paths):
        paths = tuple(paths)
        if not paths:
            raise ValueError("nonempty path sequence required")
        rows, stamp = [], 0.
        for i,p in enumerate(paths):
            if not np.array_equal(p.wind,paths[0].wind):
                raise ValueError("concatenated paths require the same wind")
            if i:
                a,b = paths[i-1].at(paths[i-1].duration),p.at(0)
                if np.linalg.norm(a[:2]-b[:2]) > 1e-5 or abs(wrap_angle(a[2]-b[2])) > 1e-5:
                    raise ValueError("continuous concatenation required")
            part = p.samples.copy()
            part[:,0] += stamp
            rows.extend(part[:-1])
            stamp += p.duration
        rows.append(part[-1])
        return cls.from_samples(rows,paths[0].wind,paths[0].airspeed)

    def slice(self, start, finish):
        if not 0 <= start < finish <= self.duration:
            raise ValueError("finite increasing slice inside path required")
        # Avoid duplicate boundary stamps caused by repeated0.05 additions.
        # Keeping a point~1e-15s beside the exact boundary would turn harmless
        # roundoff into an enormous implied interpolation speed.
        middle = self.samples[(self.samples[:,0] > start+1e-8) & (self.samples[:,0] < finish-1e-8)]
        rows = np.vstack((np.r_[start,self.at(start)],middle,np.r_[finish,self.at(finish)]))
        rows[:,0] -= start
        return self.from_samples(rows,self.wind,self.airspeed)

    def at(self, elapsed):
        if not math.isfinite(elapsed) or not 0 <= elapsed <= self.duration+1e-6:
            raise ValueError("path time outside committed interval")
        times = self.samples[:, 0]
        i = max(0, min(len(times)-2, int(np.searchsorted(times, elapsed, side="right")-1)))
        fraction = float(np.clip((elapsed-times[i])/(times[i+1]-times[i]), 0., 1.))
        return self.samples[i, 1:]*(1-fraction)+self.samples[i+1, 1:]*fraction

    def target(self, elapsed, position, altitude):
        return tracking_target(self.at(elapsed), position, altitude, self.wind)


def tracking_target(sample, position, altitude, wind):
    x, y, heading, speed, bank = sample
    tangent = np.array([math.cos(heading), math.sin(heading)])
    correction = .7*(np.array([x, y])-np.asarray(position)[:2])
    magnitude = float(np.linalg.norm(correction))
    if magnitude > 4:
        correction *= 4/magnitude
    ground_velocity = speed*tangent+correction
    requested_airspeed = float(np.clip(np.linalg.norm(ground_velocity-wind), 16., 24.))
    return FlightTarget(math.atan2(ground_velocity[1], ground_velocity[0]), altitude, requested_airspeed, bank)


@dataclass(frozen=True)
class RouteProgram:
    """Finite prefix with a target-independent periodic physical holding tail."""

    prefix: TimedPath
    tail: TimedPath

    def __post_init__(self):
        a, b = self.prefix.at(self.prefix.duration), self.tail.at(0)
        end = self.tail.at(self.tail.duration)
        if (np.linalg.norm(a[:2]-b[:2]) > 1e-5 or abs(wrap_angle(a[2]-b[2])) > 1e-5
                or np.linalg.norm(end[:2]-b[:2]) > 1e-5 or abs(wrap_angle(end[2]-b[2])) > 1e-5
                or not np.array_equal(self.prefix.wind, self.tail.wind)):
            raise ValueError("continuous prefix and closed periodic tail required")

    def at(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("causal program time required")
        if elapsed < self.prefix.duration:
            return self.prefix.at(elapsed)
        return self.tail.at((elapsed-self.prefix.duration) % self.tail.duration)

    def target(self, elapsed, position, altitude):
        return tracking_target(self.at(elapsed), position, altitude, self.prefix.wind)


def fleet_clearance(programs, *, sample_dt=.05):
    if len(programs) != 4 or not math.isfinite(sample_dt) or sample_dt <= 0:
        raise ValueError("exactly four programs and positive sampling interval required")
    for p in programs:
        for path in (p.prefix,p.tail):
            slope = np.linalg.norm(np.diff(path.samples[:,1:3],axis=0),axis=1)/np.diff(path.samples[:,0])
            if not np.isfinite(slope).all() or float(slope.max()) > 27+1e-6:
                raise ValueError("nominal interpolated path exceeds27m/s Lipschitz premise")
    period = programs[0].tail.duration
    if any(abs(p.tail.duration-period) > 1e-6 for p in programs):
        raise ValueError("different tail periods need a different infinite-tail proof")
    # Once all have reached equal-period tails, their joint geometry repeats.
    end = max(p.prefix.duration for p in programs)+period
    times = np.r_[np.arange(0., end, sample_dt), end]
    positions = [np.array([p.at(float(t))[:2] for t in times]) for p in programs]
    minimum = min(float(np.linalg.norm(positions[i]-positions[j], axis=1).min())
                  for i in range(4) for j in range(i+1, 4))
    maximum = max(float(np.abs(p).max()) for p in positions)
    return {"sampled_minimum_horizontal_m": minimum, "continuous_lower_bound_horizontal_m": minimum-27*sample_dt,
            "maximum_abs_xy_m_sampled": maximum, "continuous_abs_xy_upper_bound_m": maximum+13.5*sample_dt,
            "checked_seconds": end, "joint_periodic_tail_seconds": period,
            "scope": "complete ideal nominal prefix+periodic-tail geometry; no physical tracking certificate"}
