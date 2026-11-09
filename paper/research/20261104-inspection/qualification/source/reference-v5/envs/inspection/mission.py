"""Inspection world, causal synthetic sensing and task-tool execution.

The worker owns this object. Actor-facing RPCs expose only public_state/tool;
truth, pending packets, sensor RNG and grading remain inside the worker.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import copy
import math

import numpy as np

from envs.inspection.flight import FlightWorld
from envs.inspection.protocol import Evidence, JobBook, Observation, ReportLedger, audit_value
from envs.inspection.scenario import CLASSES, Entity, generate_dev, in_polygon, public_card
from envs.inspection.scoring import final_static_quality, grade_dynamic_bins


class InspectionMission:
    def __init__(self, seed=0, *, reference=False, events=True):
        self.seed, self.events = seed, events
        self.entities = generate_dev(seed, reference=reference)
        self.card = public_card()
        self.card["level"] = "L3" if events else "L2"
        self.flight = FlightWorld(seed, jitter=True)
        self.jobs = JobBook(4)
        self.ledger = ReportLedger(10, set(CLASSES), self.truth_at)
        self.pending, self.delivered, self.sensor_audit = [], [], []
        self.pending_coverage, self.coverage_evidence = [], []
        self.trackers, self.covered = [{} for _ in range(4)], set()
        self.public_events, self.tool_audit, self.trace = [], [], []
        self.retired = set()
        self.cells = {f"cell-{x}-{y}": (float(x), float(y)) for x in range(-390, 400, 20)
                      for y in range(-390, 400, 20) if in_polygon(x, y)}
        self.entity_bodies = {}
        p = self.flight.world
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=.6)
        for entity in self.entities:
            position = entity.position(0)
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                     basePosition=[*(position or (0, 0)), .6 if position else -10])
            self.entity_bodies[entity.key] = body
        self.obstacles = []
        for x, y, half_x, half_y, height in [(-60, 20, 20, 18, 18), (210, -170, 25, 15, 22)]:
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_x, half_y, height/2])
            self.obstacles.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                                    basePosition=[x, y, height/2]))
        p.register_all_new_bodies()
        p.before_physics = self._move_entities
        self.distance_m = np.zeros(4)
        self._previous_positions = np.array([s[3].copy() for s in p.all_states])
        original_audit = p.after_physics
        def audit_and_distance(stamp):
            original_audit(stamp)
            positions = np.array([p.getBasePositionAndOrientation(d.Id)[0] for d in p.drones])
            self.distance_m += np.linalg.norm(positions-self._previous_positions, axis=1)
            self._previous_positions = positions
        p.after_physics = audit_and_distance
        self.telemetry = [None]*4
        self.job_cache = {}
        self.confirmation_views = {}
        self._refresh_public()

    @property
    def t(self):
        return self.flight.t

    def truth_at(self, stamp):
        return {entity.key: position for entity in self.entities if (position := entity.position(stamp)) is not None}

    def _move_entities(self, stamp):
        for entity in self.entities:
            if entity.object_type == "dynamic":
                position = entity.position(stamp)
                self.flight.world.resetBasePositionAndOrientation(self.entity_bodies[entity.key],
                                                                  [*(position or (0, 0)), .6 if position else -10],
                                                                  [0, 0, 0, 1])

    def online(self, drone_id, stamp=None):
        stamp = self.t if stamp is None else stamp
        return not (self.events and drone_id == 1 and 210 <= stamp < 240)

    def camera_quality(self, drone_id, stamp=None):
        stamp = self.t if stamp is None else stamp
        return .55 if self.events and drone_id == 2 and 300 <= stamp < 345 else 1.0

    def _events(self):
        if self.events and self.t >= 180 and self.card["task_card_version"] == 1:
            self.card["task_card_version"], self.card["effective_from"] = 2, 180.0
            for rule in self.card["rules"]:
                if rule["rule_id"] == "static-0":
                    rule["priority"] = 3
                elif rule["rule_id"] == "dynamic-inspection":
                    rule["priority"] = 2
            self.public_events.append({"t": 180.0, "type": "priority_update", "task_card_version": 2})
        if self.events and self.t >= 360 and 3 not in self.retired:
            self.retired.add(3)
            self.jobs.create(3, "exit", {"origin": "exogenous_capacity_exit", "rule_id": None,
                                         "task_card_version": self.card["task_card_version"]},
                             self.jobs.versions[3], self.t, preempt=True)
            self.public_events.append({"t": self.t, "type": "exogenous_capacity_exit", "drone_id": 3})

    def _active(self, drone_id):
        job_id = self.jobs.active.get(drone_id)
        job = self.jobs.jobs.get(job_id)
        return job if job and job.status in ("queued", "running") else None

    def _advance_jobs(self):
        for i in range(4):
            job = self._active(i)
            if job is None:
                continue
            if job.status == "queued":
                self.flight.modes[i] = "exit" if job.kind == "exit" else "survey"
                job.params["initial_laps"] = self.flight.routes[i].laps
                self.jobs.transition(job, "running", self.t)
            elapsed = self.t-job.started_at
            if job.kind == "survey" and elapsed >= job.params["duration"]:
                completed_laps = self.flight.routes[i].laps-job.params["initial_laps"]
                self.jobs.transition(job, "succeeded" if completed_laps >= 1 else "failed", self.t,
                                     "physical survey pass completed" if completed_laps >= 1 else "no complete pass")
            elif job.kind == "inspect" and elapsed > 90:
                self.jobs.transition(job, "failed", self.t, "no qualified second view within 90 seconds")
            elif job.kind == "observe" and elapsed >= job.params["duration"]:
                obtained = job.params.get("measurements", 0)
                self.jobs.transition(job, "succeeded" if obtained else "failed", self.t,
                                     "observation interval completed" if obtained else "no new measurement")
            elif job.kind == "exit" and self.flight.exit_dwell[i] >= 20-1e-6:
                self.jobs.transition(job, "succeeded", self.t, "physical orbit dwell satisfied")

    def _visible(self, position, ground, axis, half_angle, body=None):
        vector = np.array([ground[0], ground[1], .6 if body is not None else .01])-position
        distance = float(np.linalg.norm(vector))
        if distance == 0 or vector[2] >= 0:
            return False
        direction = vector/distance
        if -direction[2] < math.cos(math.radians(50)) or np.dot(direction, axis) < math.cos(half_angle):
            return False
        end = np.array([ground[0], ground[1], .6 if body is not None else .01])
        hit = self.flight.world.rayTest(position.tolist(), end.tolist())[0]
        # A ground-cell ray must be clear until its endpoint. Entity rays may
        # hit the proxy itself, but never an intervening building/other proxy.
        return hit[0] == body if body is not None else hit[0] < 0 or hit[2] >= .999

    def camera_origin(self, drone_id):
        drone = self.flight.world.drones[drone_id]
        position, quaternion = self.flight.world.getBasePositionAndOrientation(drone.Id)
        rotation = np.array(self.flight.world.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        return np.array(position)+rotation@np.array([0.0, 0.0, -.3])

    def _candidate(self, xy, object_type, stamp, drone_id):
        tracker = self.trackers[drone_id]
        nearby = [(math.dist(xy, c["xy"]), key) for key, c in tracker.items()
                  if c["object_type"] == object_type and math.dist(xy, c["xy"]) <= 25]
        key = min(nearby)[1] if nearby else f"candidate-{drone_id}-{len(tracker)+1:05d}"
        tracker[key] = {"xy": tuple(xy), "object_type": object_type, "measured_at": stamp}
        return key

    def _sample(self):
        stamp = self.t
        tick = round(stamp*2)
        for drone_id in range(4):
            if self.flight.modes[drone_id] == "exit":
                continue
            aircraft_position = self.flight.world.state(drone_id)[3].copy()
            position = self.camera_origin(drone_id)
            job = self._active(drone_id)
            inspect = job is not None and job.kind in ("inspect", "observe")
            aim = job.params["aim_xy"] if inspect else None
            axis = np.array([aim[0]-position[0], aim[1]-position[1], -position[2]]) if inspect else np.array([0., 0., -1.])
            axis /= np.linalg.norm(axis)
            half_angle = math.radians(10 if inspect else 55)
            quality = self.camera_quality(drone_id)
            coverage_cells = []
            if quality >= .8:
                # Coverage uses actual camera footprint and independent rays,
                # never hidden target coordinates or traversed route length.
                for cell_id, ground in self.cells.items():
                    if (math.dist(position[:2], ground) <= 150
                            and self._image_quality(position, ground, drone_id, inspect) >= .8
                            and self._visible(position, ground, axis, half_angle)):
                        coverage_cells.append(cell_id)
            self.pending_coverage.append((stamp+.5, drone_id, stamp, coverage_cells))
            local_index = 0
            for index, entity in enumerate(self.entities):
                truth = entity.position(stamp)
                if truth is None or not self._visible(position, truth, axis, half_angle, self.entity_bodies[entity.key]):
                    continue
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, tick, drone_id, index, 0]))
                if rng.random() < .1:
                    continue
                sigma = (1.0 if inspect else 4.0)*(1 if quality >= .8 else 3)
                estimate = tuple((np.array(truth)+rng.normal(0, sigma, 2)).tolist())
                accuracy = (.99 if inspect else .8) if quality >= .8 else .65
                alternatives = [c for c in CLASSES[:3] if c != entity.true_class]
                predicted = entity.true_class if rng.random() < accuracy else str(rng.choice(alternatives))
                if entity.true_class == "normal" and not inspect:
                    predicted = "normal" if rng.random() >= .3 else str(rng.choice(CLASSES[:3]))
                candidate = self._candidate(estimate, entity.object_type, stamp, drone_id)
                if inspect:
                    # Track association uses observed space, never entity keys.
                    if math.dist(estimate, aim) > 25:
                        continue
                    candidate = job.params["candidate_id"]
                    self.trackers[drone_id][candidate] = {"xy": estimate, "object_type": entity.object_type, "measured_at": stamp}
                    job.params["aim_xy"] = estimate
                evidence_id = f"obs-{drone_id}-{tick}-{local_index}"
                local_index += 1
                obs = Observation(evidence_id, candidate, drone_id, stamp, stamp+.5, estimate, predicted,
                                  entity.object_type, self._image_quality(position, truth, drone_id, inspect), (sigma*sigma, sigma*sigma),
                                  "inspect" if inspect else "survey", tuple(aircraft_position[:2]), tuple(position))
                self.pending.append((stamp+.5, Evidence(obs, entity.key), job.params.get("report_id") if inspect and job.kind == "observe" else None))
                self.sensor_audit.append({"evidence_id": evidence_id, "entity_key": entity.key, "t": stamp,
                                          "drone_id": drone_id, "mode": obs.sensor_mode,
                                          "truth_xy": truth, "estimate_xy": estimate, "predicted_class": predicted,
                                          "true_class": entity.true_class, "quality": obs.quality})
                if inspect and obs.quality >= .8:
                    views = self.confirmation_views.setdefault(job.job_id, [])
                    if not views or math.dist(views[0].aircraft_xy, obs.aircraft_xy) >= 20:
                        views.append(obs)
                    job.params["measurements"] = job.params.get("measurements", 0)+1
                    if job.kind == "observe" and "handoff_from_job" in job.params and "handoff_effective_at" not in job.params:
                        source = self.jobs.jobs[job.params["handoff_from_job"]]
                        if source.status == "running":
                            self.jobs.transition(source, "cancelled", stamp, "handoff acquired a qualified new view")
                        job.params["handoff_effective_at"] = stamp
                    if job.kind == "inspect" and len(views) >= 2 and job.status == "running":
                        self.jobs.transition(job, "succeeded", self.t, "two qualified spatially separated views acquired")
            if not inspect:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, tick, drone_id, 0, 1]))
                if rng.random() < .01:
                    xy = tuple((position[:2]+rng.uniform(-30, 30, 2)).tolist())
                    if not self._visible(position, xy, axis, half_angle):
                        continue
                    candidate = self._candidate(xy, "static", stamp, drone_id)
                    evidence_id = f"obs-{drone_id}-{tick}-{local_index}"
                    obs = Observation(evidence_id, candidate, drone_id, stamp, stamp+.5, xy, str(rng.choice(CLASSES[:3])),
                                      "static", self._image_quality(position, xy, drone_id, False), (16., 16.), "survey", tuple(aircraft_position[:2]), tuple(position))
                    self.pending.append((stamp+.5, Evidence(obs, None), None))
                    self.sensor_audit.append({"evidence_id": evidence_id, "entity_key": None, "t": stamp,
                                              "drone_id": drone_id, "mode": "survey", "true_class": "background",
                                              "predicted_class": obs.estimated_class, "quality": obs.quality})

    def _image_quality(self, position, ground, drone_id, inspect):
        angle = math.degrees(math.atan2(math.dist(position[:2], ground), position[2]))
        return self.camera_quality(drone_id)*(.95 if inspect else .95-.2*(angle/50)**2)

    def _deliver(self):
        remaining = []
        for ready_at, evidence, report_id in self.pending:
            obs = evidence.observation
            if ready_at > self.t or not self.online(obs.drone_id):
                remaining.append((ready_at, evidence, report_id))
                continue
            obs = replace(obs, arrived_at=self.t)
            self.ledger.register(Evidence(obs, evidence.entity_key))
            self.delivered.append(obs)
            if report_id is not None:
                self.ledger.submit(self.report_request(obs, report_id), self.t)
        self.pending = remaining
        remaining_coverage = []
        for ready_at, drone_id, stamp, cells in self.pending_coverage:
            if ready_at > self.t or not self.online(drone_id):
                remaining_coverage.append((ready_at, drone_id, stamp, cells))
            else:
                self.covered.update(cells)
                self.coverage_evidence.append({"drone_id": drone_id, "measured_at": stamp,
                                               "arrived_at": self.t, "cells": cells})
        self.pending_coverage = remaining_coverage

    @staticmethod
    def report_request(obs, report_id):
        return {"report_id": report_id, "object_type": obs.object_type, "candidate_id": obs.candidate_id,
                "measured_at": obs.measured_at, "estimate_xy": obs.estimate_xy,
                "claimed_class": obs.estimated_class, "evidence_ids": [obs.evidence_id]}

    def _refresh_public(self):
        for i in range(4):
            if self.online(i):
                self.telemetry[i] = {"drone_id": i, "measured_at": self.t,
                                     "position": self.flight.world.state(i)[3].tolist(),
                                     "camera_quality": self.camera_quality(i), "retired": i in self.retired,
                                     "remaining_range_m": self.card["range_budget_m_per_aircraft"]-self.distance_m[i],
                                     "job_version": self.jobs.versions[i]}
        for job in self.jobs.jobs.values():
            if self.online(job.drone_id):
                safe = asdict(job)
                safe["params"] = {k: v for k, v in safe["params"].items() if k != "aim_xy"}
                self.job_cache[job.job_id] = safe

    def step(self):
        self._events()
        self._advance_jobs()
        self.flight.step()
        self._events()  # apply an event at this exact tick before sensing/publication
        if self.flight.world.physics_steps % 120 == 0:
            self._sample()
            self._deliver()
            self._refresh_public()
        if self.flight.world.physics_steps % 24 == 0:
            self.trace.append({"t": self.t, "positions": [s[3].tolist() for s in self.flight.world.all_states],
                               "modes": self.flight.modes.copy(), "coverage_cells": len(self.covered),
                               "exit_dwell": self.flight.exit_dwell.tolist()})

    def public_state(self, since=0.0):
        return copy.deepcopy({"sim_time": self.t, "task_card": self.card,
                "aircraft": [{**row, "link_available": self.online(i)} for i, row in enumerate(self.telemetry)],
                "jobs": list(self.job_cache.values()), "events": self.public_events,
                "coverage": {"qualified_cells": len(self.covered), "required_cells": len(self.cells)},
                "observations": [o.public() for o in self.delivered if o.arrived_at > since],
                "report_slots_used": len(self.ledger.reports)})

    def tool(self, name, args):
        try:
            if name == "submit_or_revise_report":
                response = self.ledger.submit(args, self.t)
            elif name == "cancel_job":
                if set(args) != {"job_id", "expected_version"}:
                    raise ValueError("tool schema fields do not match")
                existing = self.jobs.jobs.get(args["job_id"])
                if existing is not None and not self.online(existing.drone_id):
                    raise ValueError("link unavailable")
                response = self.jobs.cancel(args["job_id"], args["expected_version"], self.t)
            elif name == "handoff_observation":
                required = {"from_drone", "to_drone", "candidate_id", "duration", "report_id",
                            "expected_from_version", "expected_to_version", "rule_id", "task_card_version"}
                if set(args) != required:
                    raise ValueError("tool schema fields do not match")
                source_id, destination_id = args["from_drone"], args["to_drone"]
                if (any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < 4 for i in (source_id, destination_id))
                        or source_id == destination_id or not self.online(source_id)):
                    raise ValueError("distinct connected source and destination required")
                source = self._active(source_id)
                if (source is None or source.kind != "observe" or source.params["candidate_id"] != args["candidate_id"]
                        or args["expected_from_version"] != self.jobs.versions[source_id]
                        or isinstance(args["expected_from_version"], bool) or not isinstance(args["expected_from_version"], int)):
                    raise ValueError("source observation or version is invalid")
                forwarded = {"drone_id": destination_id, "candidate_id": args["candidate_id"], "duration": args["duration"],
                             "report_id": args["report_id"], "expected_version": args["expected_to_version"],
                             "rule_id": args["rule_id"], "task_card_version": args["task_card_version"]}
                response = self.tool("observe_track", forwarded)
                if response["status"] == "accepted":
                    self.jobs.jobs[response["job_id"]].params["handoff_from_job"] = source.job_id
            elif name in ("assign_survey", "inspect_candidate", "observe_track", "enter_safe_exit"):
                schemas = {"assign_survey": {"drone_id", "region_id", "duration", "expected_version", "rule_id", "task_card_version"},
                           "inspect_candidate": {"drone_id", "candidate_id", "expected_version", "rule_id", "task_card_version"},
                           "observe_track": {"drone_id", "candidate_id", "duration", "report_id", "expected_version", "rule_id", "task_card_version"},
                           "enter_safe_exit": {"drone_id", "expected_version"}}
                if set(args) != schemas[name]:
                    raise ValueError("tool schema fields do not match")
                i = args["drone_id"]
                if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < 4:
                    raise ValueError("unknown aircraft")
                if not self.online(i):
                    raise ValueError("link unavailable")
                if i in self.retired and name != "enter_safe_exit":
                    raise ValueError("aircraft externally retired")
                if name != "enter_safe_exit" and (isinstance(args["task_card_version"], bool) or not isinstance(args["task_card_version"], int)
                                                  or args["task_card_version"] != self.card["task_card_version"]
                                                  or args["rule_id"] not in {r["rule_id"] for r in self.card["rules"]}):
                    raise ValueError("stale task card or unknown rule")
                params = {k: v for k, v in args.items() if k not in ("drone_id", "expected_version")}
                kind = {"assign_survey": "survey", "inspect_candidate": "inspect", "observe_track": "observe", "enter_safe_exit": "exit"}[name]
                if "duration" in params and (isinstance(params["duration"], bool) or not isinstance(params["duration"], (int, float))
                                              or not math.isfinite(params["duration"]) or params["duration"] <= 0):
                    raise ValueError("positive finite duration required")
                if kind == "survey" and args["region_id"] != f"stripe-{i}":
                    raise ValueError("this fixed-route executor requires the aircraft's declared stripe")
                if kind in ("inspect", "observe"):
                    visible = [o for o in self.delivered if o.candidate_id == args["candidate_id"]]
                    if not visible:
                        raise ValueError("unknown observed candidate")
                    params["aim_xy"] = visible[-1].estimate_xy
                    if kind == "observe" and args["report_id"] not in self.ledger.reports:
                        raise ValueError("observation upload needs an existing report")
                response = self.jobs.create(i, kind, params, args["expected_version"], self.t, preempt=kind == "exit")
            else:
                response = {"status": "rejected", "reason": "tool is not in the actor allowlist"}
        except (KeyError, ValueError, TypeError) as error:
            response = {"status": "rejected", "reason": str(error)}
        self.tool_audit.append(audit_value({"t": self.t, "name": name, "args": args, "response": response}))
        return response

    def weight(self, entity, stamp):
        return 3 if self.events and stamp >= 180 and entity.rule_id == "static-0" else (
            2 if self.events and stamp >= 180 and entity.rule_id == "dynamic-inspection" else 1)

    def grade(self):
        static, dynamic = [], []
        for entity in self.entities:
            if not entity.rule_id:
                continue
            report = next((r for r in self.ledger.reports.values() if r.associated_key == entity.key and not r.withdrawn
                           and r.object_type == entity.object_type), None)
            versions = report.versions if report else []
            if entity.object_type == "static":
                value = final_static_quality(versions, entity.origin_xy, entity.true_class, 600)
                static.append({"entity_key": entity.key, "weight": self.weight(entity, 600), "quality": value})
            else:
                value, bins = grade_dynamic_bins(versions, entity.position, entity.true_class, entity.appeared_at, 600,
                                                 weight_at=lambda t, e=entity: self.weight(e, t))
                observed, _ = grade_dynamic_bins(versions, entity.position, entity.true_class, entity.appeared_at, 600,
                                                 weight_at=lambda t, e=entity: self.weight(e, t), track="observed")
                dynamic.append({"entity_key": entity.key, "quality": value, "observed": observed,
                                "bins": [asdict(b) for b in bins]})
        c = len(self.covered)/len(self.cells)
        s = sum(x["quality"]*x["weight"] for x in static)/sum(x["weight"] for x in static)
        dynamic_denominator = sum(b["weight"] for x in dynamic for b in x["bins"])
        d = sum(b["weight"]*b["quality"] for x in dynamic for b in x["bins"])/dynamic_denominator
        q = (c+s+d)/3
        h = int(not self.flight.violations)
        remaining_range = self.card["range_budget_m_per_aircraft"]-self.distance_m
        e = int(all(x >= 20-1e-6 for x in self.flight.exit_dwell)
                and all(x >= self.card["exit_range_reserve_m"] for x in remaining_range))
        return {"C": c, "S": s, "D": d, "Q": q, "H": h, "E": e, "Y": h*e*q,
                "SR": int(h == 1 and e == 1 and c >= .8 and s >= .6 and d >= .5),
                "remaining_range_m": remaining_range.tolist(),
                "static_recall": sum(x["quality"] > 0 for x in static)/len(static),
                "static_denominator_weight": sum(x["weight"] for x in static),
                "dynamic_denominator_weight": dynamic_denominator,
                "static_entities": static, "dynamic_entities": dynamic,
                "association_ambiguities": self.ledger.association_ambiguities}

    def audit(self):
        return {"score": self.grade(), **self.business_audit()}

    def business_audit(self):
        return {"safety": self.flight.summary(), "trace": self.trace,
                "coverage_evidence": self.coverage_evidence, "required_cells": self.cells,
                "sensor_audit": self.sensor_audit, "delivered_observations": [o.public() for o in self.delivered],
                "jobs": self.jobs.public(), "job_transitions": self.jobs.transitions,
                "tool_calls": self.tool_audit, "reports": self.ledger.audit,
                "hidden_entities": [asdict(e) for e in self.entities], "public_events": self.public_events}

    def close(self):
        self.flight.close()
