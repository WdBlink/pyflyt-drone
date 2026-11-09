"""Development scenario adapter around the qualified mission components."""

from __future__ import annotations

from envs.inspection.mission import InspectionMission
from envs.inspection.protocol import audit_value
from envs.inspection.scoring import final_static_quality
from eval.inspection_cases import coverage_cells, weight_at
from eval.inspection_population import validate_development_case


class ScenarioMission(InspectionMission):
    def __init__(self, case, *, start_planner=True):
        validate_development_case(case)
        self.case = case
        super().__init__(case.seed, events=False, start_planner=start_planner, wind_xy=case.wind_xy)
        # Parent construction has not advanced physics. Reuse hidden proxy IDs;
        # never remove/reindex a live aircraft body or reset its physical state.
        p = self.flight.world
        old_bodies = list(self.entity_bodies.values())
        self.entities = list(case.entities)
        self.entity_bodies = {}
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=.6)
        for index, entity in enumerate(self.entities):
            position = entity.position(0)
            body = (old_bodies[index] if index < len(old_bodies) else
                    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape))
            p.resetBasePositionAndOrientation(body, [*(position or (0, 0)), .6 if position else -10], [0, 0, 0, 1])
            self.entity_bodies[entity.key] = body
        for body in old_bodies[len(self.entities):]:
            p.resetBasePositionAndOrientation(body, [0, 0, -100], [0, 0, 0, 1])
            p.setCollisionFilterGroupMask(body, -1, 0, 0)
        p.register_all_new_bodies()
        self.card["level"] = case.level
        self.card["roi_polygon"] = case.roi_polygon
        self.card["coverage_grid"] = {"spacing_m": 20, "origin_xy": [10, 10], "inclusion": "center inside ROI"}
        if case.level == "L1":
            self.card["rules"] = [r for r in self.card["rules"] if r["object_type"] == "static"]
            self.card["dynamic_requirement"] = None
        self.cells = coverage_cells(case.roi_polygon)
        self._refresh_public()

    def online(self, drone_id, stamp=None):
        stamp = self.t if stamp is None else stamp
        c = self.case
        return not (c.link_drone == drone_id and c.link_start <= stamp < c.link_end)

    def camera_quality(self, drone_id, stamp=None):
        stamp = self.t if stamp is None else stamp
        c = self.case
        return .55 if c.camera_drone == drone_id and c.camera_start <= stamp < c.camera_end else 1.0

    def _events(self):
        c = self.case
        if c.priority_at is not None and self.t >= c.priority_at and self.card["task_card_version"] == 1:
            self.card["task_card_version"], self.card["effective_from"] = 2, c.priority_at
            for rule in self.card["rules"]:
                if rule["rule_id"] == "static-0":
                    rule["priority"] = 3
                elif rule["rule_id"] == "dynamic-inspection":
                    rule["priority"] = 2
            self.public_events.append({"t": c.priority_at, "type": "priority_update", "task_card_version": 2})
        if c.retire_at is not None and self.t >= c.retire_at and c.retire_drone not in self.retired:
            i = c.retire_drone
            self.retired.add(i)
            self.executor.submit(i, "exit", {"origin": "exogenous_capacity_exit", "rule_id": None,
                "task_card_version": self.card["task_card_version"]}, self.jobs.versions[i], self.t, preempt=True)
            self.public_events.append({"t": self.t, "type": "exogenous_capacity_exit", "drone_id": i})

    def weight(self, entity, stamp):
        return weight_at(self.case, entity, stamp)

    def grade(self):
        if self.case.level != "L1":
            return super().grade()
        static = []
        for entity in self.entities:
            if not entity.rule_id:
                continue
            report = next((r for r in self.ledger.reports.values() if r.associated_key == entity.key
                           and not r.withdrawn and r.object_type == "static"), None)
            quality = final_static_quality(report.versions if report else [], entity.origin_xy, entity.true_class, 600)
            static.append({"entity_key": entity.key, "weight": self.weight(entity, 600), "quality": quality})
        c = len(self.covered) / len(self.cells)
        denominator = sum(row["weight"] for row in static)
        s = sum(row["weight"]*row["quality"] for row in static) / denominator
        q, h = (c+s)/2, int(not self.flight.violations)
        remaining = self.card["range_budget_m_per_aircraft"] - self.distance_m
        e = int(all(x >= 20-1e-6 for x in self.flight.exit_dwell)
                and all(x >= self.card["exit_range_reserve_m"] for x in remaining))
        return {"C": c, "S": s, "D": None, "Q": q, "H": h, "E": e, "Y": h*e*q,
                "SR": int(h == 1 and e == 1 and c >= .8 and s >= .6), "remaining_range_m": remaining.tolist(),
                "static_recall": sum(row["quality"] > 0 for row in static)/len(static),
                "static_denominator_weight": denominator, "dynamic_denominator_weight": 0,
                "static_entities": static, "dynamic_entities": [],
                "association_ambiguities": self.ledger.association_ambiguities}

    def business_audit(self):
        return {**super().business_audit(), "case_sha256": self.case.digest(),
                "private_case": audit_value(self.case.as_dict()), "adapter": "development-scenario-v1"}
