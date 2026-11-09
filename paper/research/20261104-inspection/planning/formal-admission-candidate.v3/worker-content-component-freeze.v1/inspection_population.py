"""Shared frozen population law with distinct development/formal identities.

Formal geometry generation is reachable through the namespace-verified adapter;
content validation by itself never supplies scientific or dispatch admission.
"""

from dataclasses import dataclass
import json
import math

import numpy as np

from envs.inspection.scenario import CLASSES, ROI, in_polygon
from eval.inspection_cases import (
    OBSTACLES, CaseEntity, CaseSpec, _edge_distances, _intersects, capacity_bound,
)


SCHEMA = "inspection-population-case-v2"
CLEAN_SIBLING_SCHEMA = "inspection-event-clean-sibling-v1"
EVENT_FIELDS = ("priority_at", "link_drone", "link_start", "link_end", "camera_drone",
                "camera_start", "camera_end", "retire_drone", "retire_at")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class PopulationCase(CaseSpec):
    domain: str

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("population case fields do not match schema")
        try:
            converted = dict(value)
            converted["roi_polygon"] = tuple(tuple(p) for p in value["roi_polygon"])
            converted["wind_xy"] = tuple(value["wind_xy"])
            converted["entities"] = tuple(CaseEntity(**{**e, "origin_xy": tuple(e["origin_xy"])})
                for e in value["entities"])
            case = cls(**converted)
            validate_population_case(case)
        except (TypeError, KeyError, IndexError) as error:
            raise ValueError("malformed population case") from error
        return case


@dataclass(frozen=True)
class EventCleanSibling(PopulationCase):
    parent_case: dict
    parent_sha256: str

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("clean sibling fields do not match schema")
        try:
            converted = dict(value)
            converted["parent_case"] = PopulationCase.from_dict(value["parent_case"]).as_dict()
            converted["roi_polygon"] = tuple(tuple(p) for p in value["roi_polygon"])
            converted["wind_xy"] = tuple(value["wind_xy"])
            converted["entities"] = tuple(CaseEntity(**{**e, "origin_xy": tuple(e["origin_xy"])})
                for e in value["entities"])
            case = cls(**converted)
            validate_event_clean_sibling(case)
        except (TypeError, KeyError, IndexError) as error:
            raise ValueError("malformed clean sibling") from error
        return case


def validate_event_clean_sibling(case):
    parent = PopulationCase.from_dict(case.parent_case)
    if (case.schema != CLEAN_SIBLING_SCHEMA or parent.level != "L3"
            or type(case.parent_sha256) is not str or case.parent_sha256 != parent.digest()):
        raise ValueError("clean intervention requires an intact L3 development parent")
    expected = {**parent.as_dict(), "schema": CLEAN_SIBLING_SCHEMA,
                **dict.fromkeys(EVENT_FIELDS)}
    actual = {k: v for k, v in case.as_dict().items() if k not in ("parent_case", "parent_sha256")}
    # JSON equality also distinguishes boolean and numeric substitutions.
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("clean sibling may remove only events; level, geometry, motion and noise stay fixed")
    return {"valid": True, "case_sha256": case.digest(), "parent_sha256": parent.digest(),
            "scope": "explicit event removal only, not a new independent root or flight qualification"}


def event_clean_sibling(parent):
    """Keep the parent's initial public level; suppress only private events."""
    validate_population_case(parent)
    case = EventCleanSibling(**{**{k: getattr(parent, k) for k in PopulationCase.__dataclass_fields__},
        "schema": CLEAN_SIBLING_SCHEMA,
        **dict.fromkeys(EVENT_FIELDS)}, parent_case=parent.as_dict(), parent_sha256=parent.digest())
    validate_event_clean_sibling(case)
    return case


def load_development_case(value):
    """Admit explicit development cases/interventions, never main/holdout."""
    if isinstance(value, dict) and value.get("schema") == CLEAN_SIBLING_SCHEMA:
        return EventCleanSibling.from_dict(value)
    if isinstance(value, dict) and value.get("schema") == SCHEMA:
        return PopulationCase.from_dict(value)
    return CaseSpec.from_dict(value)


def validate_development_case(case):
    if isinstance(case, EventCleanSibling):
        return validate_event_clean_sibling(case)
    if isinstance(case, PopulationCase):
        return validate_population_case(case)
    from eval.inspection_cases import validate_case
    return validate_case(case)


class PopulationGenerationExhausted(RuntimeError):
    def __init__(self, audit):
        super().__init__("finite development population generation exhausted; no silent replacement or range relaxation")
        self.audit = audit


def validate_population_case(case):
    if (case.schema != SCHEMA or case.split != "dev" or type(case.case_id) is not str or not case.case_id.startswith("devext-")
            or case.lineage_group != case.case_id or case.domain not in ("ID", "OOD")
            or case.level not in ("L1", "L2", "L3") or case.domain == "OOD" and case.level == "L1"
            or type(case.seed) is not int or not 0 <= case.seed < 2**128):
        raise ValueError("this implementation only generates new development population cases")
    return _validate_population_contents(case)


def validate_formal_population_case(case, phase, expected_case_id):
    """Validate formal content after a caller has verified its dataset proof."""
    if (phase not in ("main", "holdout") or type(expected_case_id) is not str
            or type(case) is not PopulationCase or case.schema != SCHEMA
            or case.split != phase or case.case_id != expected_case_id
            or case.lineage_group != expected_case_id or case.domain not in ("ID", "OOD")
            or case.level not in ("L1", "L2", "L3") or case.domain == "OOD" and case.level == "L1"
            or not expected_case_id.startswith(f"{phase}-{case.domain}-{case.level}-")
            or type(case.seed) is not int or not 0 <= case.seed < 2**128):
        raise ValueError("formal population identity differs from its admitted phase/root")
    return _validate_population_contents(case)


def decode_formal_population_case(value, phase, expected_case_id):
    """Content decoder, not a dataset reader or a qualification capability."""
    if type(value) is not dict or set(value) != set(PopulationCase.__dataclass_fields__):
        raise ValueError("formal population fields do not match schema")
    try:
        converted = dict(value)
        converted["roi_polygon"] = tuple(tuple(p) for p in value["roi_polygon"])
        converted["wind_xy"] = tuple(value["wind_xy"])
        converted["entities"] = tuple(CaseEntity(**{**e, "origin_xy": tuple(e["origin_xy"])})
                                      for e in value["entities"])
        case = PopulationCase(**converted)
        validate_formal_population_case(case, phase, expected_case_id)
    except (TypeError, KeyError, IndexError) as error:
        raise ValueError("malformed formal population case") from error
    return case


def _validate_population_contents(case):
    polygon = tuple(tuple(p) for p in case.roi_polygon)
    if (len(polygon) != 9 or len(set(polygon)) != 9
            or any(len(p) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 400 for x in p) for p in polygon)):
        raise ValueError("invalid prospective polygon")
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(polygon, polygon[1:]+polygon[:1])))/2
    if area < 400000:
        raise ValueError("prospective polygon is too small")
    for i in range(9):
        for j in range(i+1, 9):
            if j not in (i+1, (i-1) % 9) and _intersects(polygon[i], polygon[(i+1) % 9], polygon[j], polygon[(j+1) % 9]):
                raise ValueError("prospective polygon intersects itself")
    if len(case.wind_xy) != 2 or not all(_finite(x) for x in case.wind_xy) or math.hypot(*case.wind_xy) > 2:
        raise ValueError("wind exceeds the qualified envelope")
    entities = case.entities
    if (any(type(e.key) is not str or not e.key for e in entities)
            or len({e.key for e in entities}) != len(entities)):
        raise ValueError("duplicate hidden entity")
    for c in CLASSES[:3]:
        if sum(e.object_type == "static" and e.true_class == c for e in entities) not in (1, 2):
            raise ValueError("target count differs from the frozen population")
    if not 4 <= sum(e.true_class == "normal" for e in entities) <= 7:
        raise ValueError("normal count differs")
    moving = [e for e in entities if e.object_type == "dynamic"]
    if (len(moving) != 0 if case.level == "L1" else len(moving) not in (2, 3)):
        raise ValueError("dynamic population differs")
    for k, e in enumerate(sorted(moving, key=lambda e: e.appeared_at)):
        if not 110+55*k <= e.appeared_at <= 134+55*k or e.appeared_at % 1 != 0:
            raise ValueError("appearance distribution differs")
    for e in entities:
        if (len(e.origin_xy) != 2 or not all(_finite(x) for x in e.origin_xy)
                or any(not _finite(getattr(e, k)) for k in ("appeared_at", "speed", "heading", "radius", "bend"))):
            raise ValueError("entity numerical fields must be native finite numbers")
        expected = ("dynamic-inspection" if e.object_type == "dynamic" and e.true_class == CLASSES[3]
            else f"static-{CLASSES.index(e.true_class)}" if e.object_type == "static" and e.true_class in CLASSES[:3]
            else "" if e.object_type == "static" and e.true_class == "normal" else None)
        if expected is None or e.rule_id != expected:
            raise ValueError("entity has no unique valid rule")
        if e.object_type == "dynamic":
            if (not 1.1 <= e.speed <= 1.6 or not 60 <= e.radius <= 85 or not 110 <= e.appeared_at <= 244
                    or e.appeared_at+186 > 600 or e.bend != (0 if case.domain == "ID" else 1)
                    or abs(e.heading) > (.12 if case.domain == "ID" else math.pi)):
                raise ValueError("motion differs from the population")
        elif e.speed != 0 or e.appeared_at != 0:
            raise ValueError("static motion differs")
        x, y = e.origin_xy
        if e.object_type == "static":
            if e.true_class == "normal":
                supported = abs(x) <= 330 and abs(y) <= 330
            elif case.domain == "ID":
                supported = abs(x) <= 240 and any(abs(y-stripe) <= 45 for stripe in (-300, -100, 100, 300))
            else:
                supported = abs(x) <= 280 and abs(y) <= 280
        elif case.domain == "ID":
            supported = abs(x) <= 110 and any(abs(y-stripe) <= 15 for stripe in (-300, -100, 100, 300))
        else:
            center = (x-e.radius*math.sin(e.heading), y+e.radius*math.cos(e.heading))
            supported = all(abs(value) <= 200+1e-8 for value in center)
        if not supported:
            raise ValueError("entity origin lies outside the frozen population support")
        stamps = np.linspace(e.appeared_at, 600, 601)
        points = np.asarray([e.position(float(t)) for t in stamps])
        speed_bound = abs(e.speed)*max(1, abs(e.bend))
        margin = speed_bound*(600-e.appeared_at)/1200
        if not np.isfinite(points).all() or min(_edge_distances(points, polygon)) <= margin+1e-8:
            raise ValueError("continuous trajectory boundary guard")
        if not all(in_polygon(*p, polygon) for p in points):
            raise ValueError("trajectory leaves ROI")
        for x, y, hx, hy in OBSTACLES:
            distances = np.hypot(np.maximum(np.abs(points[:, 0]-x)-hx, 0), np.maximum(np.abs(points[:, 1]-y)-hy, 0))
            if min(distances) <= 2+margin:
                raise ValueError("continuous building guard")
    for i, a in enumerate(entities):
        for b in entities[i+1:]:
            start = max(a.appeared_at, b.appeared_at)
            stamps = np.linspace(start, 600, 601)
            differences = np.asarray([a.position(float(t)) for t in stamps])-np.asarray([b.position(float(t)) for t in stamps])
            margin = (abs(a.speed)*max(1, abs(a.bend))+abs(b.speed)*max(1, abs(b.bend)))*(600-start)/1200
            if min(np.linalg.norm(differences, axis=1)) < 45+margin:
                raise ValueError("continuous 45m identity separation guard")
    if case.domain == "ID" and len({min(range(4), key=lambda i: abs(e.origin_xy[1]-(-300+200*i))) for e in moving}) != len(moving):
        raise ValueError("ID dynamic stripes must be distinct")
    if capacity_bound(case)["combined_perfect_reports_needed"] > 10:
        raise ValueError("necessary report capacity bound")
    calendar = (case.priority_at, case.link_drone, case.link_start, case.link_end,
        case.camera_drone, case.camera_start, case.camera_end, case.retire_drone, case.retire_at)
    if case.level != "L3":
        if any(value is not None for value in calendar):
            raise ValueError("event outside L3")
    else:
        drones = (case.link_drone, case.camera_drone, case.retire_drone)
        times = (case.priority_at, case.link_start, case.link_end, case.camera_start, case.camera_end, case.retire_at)
        if (any(type(i) is not int or not 0 <= i < 4 for i in drones) or len(set(drones)) != 3
                or any(type(t) not in (int, float) or not math.isfinite(t) or t % 1 != 0 for t in times)
                or not 160 <= case.priority_at <= 205 or not 350 <= case.retire_at <= 385):
            raise ValueError("invalid prospective event calendar")
        ranges = ((200, 245, 20, 35, 290, 320) if case.domain == "ID" else (280, 300, 30, 40, 285, 305))
        if (not ranges[0] <= case.link_start <= ranges[1] or not ranges[2] <= case.link_end-case.link_start <= ranges[3]
                or not ranges[4] <= case.camera_start <= ranges[5] or not 30 <= case.camera_end-case.camera_start <= 45):
            raise ValueError("event distributions differ")
    return {"valid": True, "case_sha256": case.digest(), "area_m2": area,
        "scope": "geometry/identity/capacity only, not flight solvability or G4"}


def generate_development_population_case(case_id, domain, level, seed):
    if (not isinstance(case_id, str) or not case_id.startswith("devext-") or domain not in ("ID", "OOD")
            or level not in ("L1", "L2", "L3") or domain == "OOD" and level == "L1"
            or type(seed) is not int or not 0 <= seed < 2**128):
        raise ValueError("development-only population generation")
    return _generate_population_case(case_id, "dev", domain, level, seed)


def _generate_population_case(case_id, phase, domain, level, seed):
    """Same five streams and joint rejection law; caller owns pre-draw admission."""
    if phase not in ("dev", "main", "holdout"):
        raise ValueError("explicit population phase required")
    branches = np.random.SeedSequence(seed).spawn(5)
    geometry, objects, events, wind = [np.random.default_rng(s) for s in branches[:4]]
    sensor_seed = int.from_bytes(branches[4].generate_state(4).tobytes(), "little")
    audit = []
    for candidate in range(1000):
        record = {"candidate_index": candidate, "placements": []}
        audit.append(record)
        if domain == "ID":
            sx, sy = geometry.uniform(.93, .99, 2)
            theta = geometry.uniform(-.025, .025)
            co, si = math.cos(theta), math.sin(theta)
            polygon = tuple((float(co*sx*x-si*sy*y+geometry.uniform(-4, 4)),
                float(si*sx*x+co*sy*y+geometry.uniform(-4, 4))) for x, y in ROI)
        else:
            rotation = geometry.uniform(-.15, .15)
            angles = np.arange(9)*2*math.pi/9+geometry.uniform(-.03, .03, 9)+rotation
            radii = geometry.uniform(385, 398, 9)
            polygon = tuple((float(r*math.cos(t)), float(r*math.sin(t))) for r, t in zip(radii, angles))
        entities = []
        def place(normal=False):
            # One unconditional proposal per entity. Local rejection/resampling
            # would change the frozen joint law through state-dependent feasible
            # area normalizers. Only the complete joint candidate is rejected.
            if normal:
                point = tuple(float(x) for x in objects.uniform(-330, 330, 2))
            elif domain == "ID":
                point = (float(objects.uniform(-240, 240)), float(objects.choice([-300, -100, 100, 300])+objects.uniform(-45, 45)))
            else:
                point = tuple(float(x) for x in objects.uniform(-280, 280, 2))
            record["placements"].append({"entity_index": len(entities), "candidate_index": 0, "reason": "unconditional_proposal"})
            return point
        try:
            for k, count in enumerate(objects.integers(1, 3, 3)):
                for _ in range(count):
                    entities.append(CaseEntity("", "static", CLASSES[k], place(), rule_id=f"static-{k}"))
            if level != "L1":
                count = int(objects.integers(2, 4))
                stripes = objects.choice(4, count, replace=False) if domain == "ID" else [None]*count
                for k, stripe in enumerate(stripes):
                    appeared = float(objects.integers(110+55*k, 135+55*k))
                    speed, radius = float(objects.uniform(1.1, 1.6)), float(objects.uniform(60, 85))
                    heading = float(objects.uniform(-.12, .12) if domain == "ID" else objects.uniform(-math.pi, math.pi))
                    if domain == "ID":
                        origin = (float(objects.uniform(-110, 110)), float(-300+200*stripe+objects.uniform(-15, 15)))
                    else:
                        center = objects.uniform(-200, 200, 2)
                        origin = (float(center[0]+radius*math.sin(heading)), float(center[1]-radius*math.cos(heading)))
                    entities.append(CaseEntity("", "dynamic", CLASSES[3], origin, appeared, speed,
                        "dynamic-inspection", heading, radius, 0 if domain == "ID" else 1))
            for _ in range(int(objects.integers(4, 8))):
                entities.append(CaseEntity("", "static", "normal", place(True)))
            objects.shuffle(entities)
            entities = tuple(CaseEntity(**{**e.__dict__, "key": f"truth-{i:03d}"}) for i, e in enumerate(entities))
            calendar = [None]*9
            if level == "L3":
                link, camera, retire = [int(i) for i in events.permutation(4)[:3]]
                link_start = float(events.integers(200, 246) if domain == "ID" else events.integers(280, 301))
                camera_start = float(events.integers(290, 321) if domain == "ID" else events.integers(285, 306))
                link_duration = float(events.integers(20, 36) if domain == "ID" else events.integers(30, 41))
                calendar = [float(events.integers(160, 206)), link, link_start, link_start+link_duration,
                    camera, camera_start, camera_start+float(events.integers(30, 46)), retire, float(events.integers(350, 386))]
            angle, speed = wind.uniform(-math.pi, math.pi), wind.uniform(0, 2)
            case = PopulationCase(SCHEMA, case_id, case_id, phase, level, sensor_seed, polygon,
                (float(speed*math.cos(angle)), float(speed*math.sin(angle))), entities, *calendar, domain)
            validation = (validate_population_case(case) if phase == "dev" else
                          validate_formal_population_case(case, phase, case_id))
            record["status"] = "accepted"
            role = ("development geometry generation, not test or flight" if phase == "dev" else
                    "formal joint-law geometry generation after namespace verification; not flight solvability")
            return case, {"role": role, "root_generation_seed": seed,
                "sensor_seed": sensor_seed, "attempts": audit, "validation": validation}
        except ValueError as error:
            record["status"], record["reason"] = "rejected", str(error)
    raise PopulationGenerationExhausted(audit)
