"""Private development inputs; never pass a CaseSpec to a policy process."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import secrets

import numpy as np

from envs.inspection.scenario import CLASSES, ROI, in_polygon


SCHEMA = "inspection-dev-case-v1"
OBSTACLES = ((-60, 20, 20, 18), (210, -170, 25, 15))


@dataclass(frozen=True)
class CaseEntity:
    key: str
    object_type: str
    true_class: str
    origin_xy: tuple[float, float]
    appeared_at: float = 0.0
    speed: float = 0.0
    rule_id: str = ""
    heading: float = 0.0
    radius: float = 80.0
    bend: float = 0.0

    def position(self, stamp):
        if stamp < self.appeared_at:
            return None
        phase = self.speed * (stamp - self.appeared_at) / self.radius
        dx = self.radius * math.sin(phase)
        dy = self.bend * self.radius * (1 - math.cos(phase))
        co, si = math.cos(self.heading), math.sin(self.heading)
        return (self.origin_xy[0] + co * dx - si * dy,
                self.origin_xy[1] + si * dx + co * dy)


@dataclass(frozen=True)
class CaseSpec:
    schema: str
    case_id: str
    lineage_group: str
    split: str
    level: str
    seed: int
    roi_polygon: tuple[tuple[float, float], ...]
    wind_xy: tuple[float, float]
    entities: tuple[CaseEntity, ...]
    # Private event calendar, present only in L3 inputs.
    priority_at: float | None
    link_drone: int | None
    link_start: float | None
    link_end: float | None
    camera_drone: int | None
    camera_start: float | None
    camera_end: float | None
    retire_drone: int | None
    retire_at: float | None

    def as_dict(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True,
                                         allow_nan=False).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value):
        expected = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("case fields do not match schema")
        converted = dict(value)
        converted["roi_polygon"] = tuple(tuple(p) for p in value["roi_polygon"])
        converted["wind_xy"] = tuple(value["wind_xy"])
        converted["entities"] = tuple(CaseEntity(**{**e, "origin_xy": tuple(e["origin_xy"])})
                                      for e in value["entities"])
        case = cls(**converted)
        validate_case(case)
        return case


def coverage_cells(polygon):
    """All centers of the public 20m lattice, without fixed-ROI clipping."""
    xmin, ymin = np.min(polygon, axis=0)
    xmax, ymax = np.max(polygon, axis=0)
    xs = range(math.floor((xmin - 10) / 20), math.ceil((xmax - 10) / 20) + 1)
    ys = range(math.floor((ymin - 10) / 20), math.ceil((ymax - 10) / 20) + 1)
    return {f"cell-{20*i+10}-{20*j+10}": (float(20*i+10), float(20*j+10))
            for i in xs for j in ys if in_polygon(20*i+10, 20*j+10, polygon)}


def weight_at(case, entity, stamp):
    changed = case.priority_at is not None and stamp >= case.priority_at
    return (3 if changed and entity.rule_id == "static-0" else
            2 if changed and entity.rule_id == "dynamic-inspection" else 1)


def capacity_bound(case):
    def minimum_count(weights, threshold):
        accumulated = 0.0
        for count, value in enumerate(sorted(weights, reverse=True), 1):
            accumulated += value
            if accumulated >= threshold * sum(weights) - 1e-9:
                return count
        return 0

    static = [weight_at(case, e, 600) for e in case.entities if e.rule_id and e.object_type == "static"]
    dynamic = [sum(weight_at(case, e, e.appeared_at + 3*i) for i in range(60))
               for e in case.entities if e.rule_id and e.object_type == "dynamic"]
    ns, nd = minimum_count(static, .6), minimum_count(dynamic, .5)
    return {"static_perfect_reports_needed": ns, "dynamic_perfect_reports_needed": nd,
            "combined_perfect_reports_needed": ns + nd, "budget": 10,
            "scope": "necessary capacity bound only; not a physical solvability proof"}


def _orientation(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def validate_case(case):
    # JSON's strict finite check covers every numerical field, including seeds.
    json.dumps(case.as_dict(), allow_nan=False)
    if (case.schema != SCHEMA or case.split != "dev" or case.level not in ("L1", "L2", "L3")
            or not case.case_id or not case.lineage_group or type(case.seed) is not int or case.seed < 0):
        raise ValueError("unsupported development case identity")
    polygon = case.roi_polygon
    if len(polygon) != 9 or any(len(p) != 2 for p in polygon):
        raise ValueError("a nine-vertex polygon is required")
    if any(type(v) not in (int, float) or abs(v) > 400 for p in polygon for v in p):
        raise ValueError("ROI exceeds the declared development envelope")
    area = abs(sum(a[0]*b[1] - b[0]*a[1] for a, b in zip(polygon, polygon[1:]+polygon[:1]))) / 2
    if area < 400000 or len(set(polygon)) != len(polygon):
        raise ValueError("degenerate ROI")
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i+1) % len(polygon)]
        for j in range(i+1, len(polygon)):
            if j in (i+1, (i-1) % len(polygon)):
                continue
            c, d = polygon[j], polygon[(j+1) % len(polygon)]
            if (_orientation(a, b, c)*_orientation(a, b, d) <= 0
                    and _orientation(c, d, a)*_orientation(c, d, b) <= 0):
                raise ValueError("ROI edges intersect")
    if len(case.wind_xy) != 2 or any(type(x) not in (int, float) for x in case.wind_xy) or math.hypot(*case.wind_xy) > 2:
        raise ValueError("unsupported wind")
    if not coverage_cells(polygon):
        raise ValueError("empty demand grid")
    keys, static, dynamic = set(), [], []
    for entity in case.entities:
        if entity.key in keys or not entity.key or len(entity.origin_xy) != 2:
            raise ValueError("duplicate key or malformed origin")
        keys.add(entity.key)
        expected = ("dynamic-inspection" if entity.object_type == "dynamic" and entity.true_class == CLASSES[3]
                    else f"static-{CLASSES.index(entity.true_class)}" if entity.object_type == "static" and entity.true_class in CLASSES[:3]
                    else "" if entity.object_type == "static" and entity.true_class == "normal" else None)
        if expected is None or entity.rule_id != expected:
            raise ValueError("entity has no unique valid rule")
        if entity.radius <= 0 or not 0 <= entity.speed <= 1.6 or abs(entity.bend) > 1:
            raise ValueError("unsupported trajectory")
        if entity.object_type == "static":
            static.append(entity)
            if entity.appeared_at != 0 or entity.speed != 0:
                raise ValueError("static entities cannot move or appear late")
        else:
            dynamic.append(entity)
            if not 100 <= entity.appeared_at <= 300 or entity.appeared_at + 180 > 600:
                raise ValueError("incomplete dynamic demand window")
        for stamp in np.linspace(entity.appeared_at, 600, 601):
            point = entity.position(stamp)
            if not in_polygon(*point, polygon):
                raise ValueError("entity trajectory leaves ROI")
            if any(abs(point[0]-x) <= hx+2 and abs(point[1]-y) <= hy+2 for x, y, hx, hy in OBSTACLES):
                raise ValueError("ground proxy intersects a building")
    if any(math.dist(a.origin_xy, b.origin_xy) < 45 for i, a in enumerate(static) for b in static[i+1:]):
        raise ValueError("static proxies are not sufficiently separated")
    if not all(any(e.true_class == c for e in static) for c in CLASSES[:3]):
        raise ValueError("missing static demand class")
    if (case.level == "L1") != (len(dynamic) == 0):
        raise ValueError("dynamic demand does not match level")
    if capacity_bound(case)["combined_perfect_reports_needed"] > 10:
        raise ValueError("report capacity is infeasible even with perfect reports")
    fields = (case.priority_at, case.link_drone, case.link_start, case.link_end,
              case.camera_drone, case.camera_start, case.camera_end, case.retire_drone, case.retire_at)
    if case.level != "L3":
        if any(x is not None for x in fields):
            raise ValueError("events outside L3")
    elif (any(x is None for x in fields)
          or any(type(i) is not int or not 0 <= i < 4 for i in (case.link_drone, case.camera_drone, case.retire_drone))
          or not 150 <= case.priority_at <= 210
          or not 190 <= case.link_start < case.link_end <= 285
          or not 280 <= case.camera_start < case.camera_end <= 370
          or not 345 <= case.retire_at <= 390):
        raise ValueError("invalid event calendar")
    return {"valid": True, "case_sha256": case.digest(), "coverage_cells": len(coverage_cells(polygon)),
            "capacity": capacity_bound(case),
            "trajectory_validation": "601 time samples/entity; not an analytic separation proof"}


def generate_dev_case(case_id, level, seed):
    """Rejection is geometry/capacity-only, before any policy score exists."""
    if level not in ("L1", "L2", "L3") or type(seed) is not int or seed < 0:
        raise ValueError("invalid development request")
    geometry_rng, entity_rng, event_rng, wind_rng = [np.random.default_rng(s)
        for s in np.random.SeedSequence(seed).spawn(4)]
    for _ in range(1000):
        sx, sy = geometry_rng.uniform(.93, .99, 2)
        theta = geometry_rng.uniform(-.025, .025)
        co, si = math.cos(theta), math.sin(theta)
        polygon = tuple((float(co*sx*x-si*sy*y+geometry_rng.uniform(-4, 4)),
                         float(si*sx*x+co*sy*y+geometry_rng.uniform(-4, 4))) for x, y in ROI)
        entities = []
        for k, count in enumerate(entity_rng.integers(1, 3, 3)):
            for _ in range(count):
                for _ in range(1000):
                    point = (float(entity_rng.uniform(-240, 240)),
                             float(entity_rng.choice([-300, -100, 100, 300])+entity_rng.uniform(-45, 45)))
                    if in_polygon(*point, polygon) and all(math.dist(point, e.origin_xy) >= 45 for e in entities):
                        break
                entities.append(CaseEntity("", "static", CLASSES[k], point, rule_id=f"static-{k}"))
        if level != "L1":
            stripes = entity_rng.choice(4, int(entity_rng.integers(2, 4)), replace=False)
            for k, stripe in enumerate(stripes):
                entities.append(CaseEntity("", "dynamic", CLASSES[3],
                    (float(entity_rng.uniform(-110, 110)), float(-300+200*stripe+entity_rng.uniform(-15, 15))),
                    float(entity_rng.integers(110+55*k, 135+55*k)), float(entity_rng.uniform(1.1, 1.6)),
                    "dynamic-inspection", float(entity_rng.uniform(-.12, .12)), float(entity_rng.uniform(60, 85))))
        for _ in range(int(entity_rng.integers(4, 8))):
            for _ in range(1000):
                point = tuple(float(x) for x in entity_rng.uniform(-330, 330, 2))
                if in_polygon(*point, polygon) and all(math.dist(point, e.origin_xy) >= 45 for e in entities):
                    break
            entities.append(CaseEntity("", "static", "normal", point))
        entity_rng.shuffle(entities)
        entities = tuple(CaseEntity(**{**asdict(e), "key": f"truth-{i:03d}"}) for i, e in enumerate(entities))
        events = [None]*9
        if level == "L3":
            link, camera, retire = [int(x) for x in event_rng.permutation(4)[:3]]
            link_start, camera_start = float(event_rng.integers(200, 246)), float(event_rng.integers(290, 321))
            events = [float(event_rng.integers(160, 206)), link, link_start, link_start+float(event_rng.integers(20, 36)),
                      camera, camera_start, camera_start+float(event_rng.integers(30, 46)), retire, float(event_rng.integers(350, 386))]
        angle, speed = wind_rng.uniform(-math.pi, math.pi), wind_rng.uniform(0, 2)
        case = CaseSpec(SCHEMA, case_id, case_id, "dev", level, seed, polygon,
                        (float(speed*math.cos(angle)), float(speed*math.sin(angle))), entities, *events)
        try:
            validate_case(case)
            return case
        except ValueError:
            continue
    raise RuntimeError("development generator exhausted its finite geometry attempts")


def create_dev_suite(destination):
    """Write once; seed material is private and is never a policy input."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    master = secrets.randbits(128)
    seeds = np.random.SeedSequence(master).spawn(24)
    records = []
    for index, seed_sequence in enumerate(seeds):
        level, ordinal = f"L{index//8+1}", index % 8
        seed = int.from_bytes(seed_sequence.generate_state(4).tobytes(), "little")
        case = generate_dev_case(f"dev-{level}-{ordinal:02d}", level, seed)
        path = destination / f"{case.case_id}.json"
        path.write_text(json.dumps(case.as_dict(), indent=2, allow_nan=False)+"\n")
        records.append({"case_id": case.case_id, "lineage_group": case.lineage_group,
                        "file": path.name, "case_sha256": case.digest(),
                        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "first_pilot": ordinal < 2, "validation": validate_case(case)})
    (destination / "manifest.json").write_text(json.dumps({"schema": SCHEMA, "split": "dev",
        "master_seed": master, "root_count": 24, "records": records,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "main_and_holdout_instantiated": False}, indent=2)+"\n")
