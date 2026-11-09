"""Development classical coordinators consuming only serialized public state.

B1 is a centralized rolling greedy utility auction, not a CBBA reproduction.
B2 uses explicit event/state priorities. Neither has been development-selected
or qualified by merely implementing these rules.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math

from envs.inspection.reference import ReferencePolicy


@dataclass(frozen=True)
class ClassicSettings:
    candidate_max_age: float = 30.
    failed_job_cooldown: float = 30.
    minimum_commit_seconds: float = 8.
    exit_lead_seconds: float = 130.
    dynamic_value: float = 4.
    static_merge_radius_m: float = 15.
    confirmation_delivery_wait_seconds: float = 2.

    def __post_init__(self):
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in vars(self).values()):
            raise ValueError("finite positive classical development settings required")


class _PublicClassicPolicy(ReferencePolicy):
    def __init__(self, settings=None):
        super().__init__()
        self.settings = settings or ClassicSettings()
        self.terminal_seen = set()
        self.cooldown = {}
        self.survey_requests = defaultdict(int)
        self.fsm_states = {}
        self.inspect_finished = {}
        self.handoff_reservations = {}

    def _ingest(self, state):
        for observation in state["observations"]:
            if observation["evidence_id"] not in self.seen:
                self.seen.add(observation["evidence_id"])
                self.history[observation["candidate_id"]].append(observation)
        for candidate, rows in self.history.items():
            rows.sort(key=lambda row: (row["measured_at"], row["evidence_id"]))
            self.history[candidate] = rows[-64:]
        for job in state["jobs"]:
            if job["status"] not in ("succeeded", "failed", "cancelled") or job["job_id"] in self.terminal_seen:
                continue
            self.terminal_seen.add(job["job_id"])
            candidate = job["params"].get("candidate_id")
            if job["kind"] == "inspect" and job["status"] == "succeeded":
                self.inspect_finished[candidate] = job["finished_at"]
            if job["kind"] in ("inspect", "observe") and job["status"] == "failed":
                self.cooldown[candidate] = state["sim_time"]+self.settings.failed_job_cooldown
            if job["kind"] == "observe" and job["status"] in ("failed", "cancelled"):
                self.observed.discard(candidate)
            reservation = self.handoff_reservations.get(candidate)
            if reservation is not None and reservation["job_id"] == job["job_id"]:
                self.handoff_reservations.pop(candidate)
        self.observed.update(j["params"]["candidate_id"] for j in state["jobs"]
                             if j["kind"] == "observe" and j["status"] == "succeeded")

    def _reports(self, state):
        commands = []
        for candidate, rows in sorted(self.history.items()):
            if candidate in self.reports or candidate in self.rejected:
                continue
            qualified = [r for r in rows if r["sensor_mode"] == "inspect" and r["quality"] >= .8]
            if (len(qualified) < 2 or not any(math.dist(a["aircraft_xy"], b["aircraft_xy"]) >= 20
                    for i, a in enumerate(qualified) for b in qualified[i+1:])):
                continue
            counts = Counter(r["estimated_class"] for r in qualified[-8:])
            winner, count = counts.most_common(1)[0]
            if count <= sum(counts.values())/2:
                continue
            self.confirmed.add(candidate)
            if winner == "normal":
                self.rejected.add(candidate)
                continue
            latest = next(r for r in reversed(qualified) if r["estimated_class"] == winner)
            duplicate = next((rid for rid, row in {**self.report_points, **self.pending_reports}.items()
                if latest["object_type"] == row["object_type"] == "static"
                and math.dist(latest["estimate_xy"], row["xy"]) <= self.settings.static_merge_radius_m), None)
            if duplicate is not None:
                if duplicate in self.report_points:
                    self.reports[candidate] = duplicate
                else:
                    self.report_aliases[duplicate].add(candidate)
                continue
            report = self.report_request(latest)
            rid = report["report_id"]
            if rid in self.pending_reports:
                continue
            if state["report_slots_used"]+len(self.pending_reports) >= state["task_card"]["report_budget"]:
                continue
            self.pending_reports[rid] = {"xy": latest["estimate_xy"], "object_type": latest["object_type"],
                                         "claimed_class": winner}
            commands.append(("submit_or_revise_report", report))
        return commands

    @staticmethod
    def _rule(card, observation):
        return next((r for r in card["rules"] if r["object_type"] == observation["object_type"]
                     and observation["estimated_class"] in r["classes"]), None)

    def _tasks(self, state, active, aircraft):
        tasks = []
        now, card = state["sim_time"], state["task_card"]
        for candidate, rows in sorted(self.history.items()):
            latest = rows[-1]
            if (candidate in self.rejected or now-latest["measured_at"] > self.settings.candidate_max_age
                    or len(rows) < 2 or self.cooldown.get(candidate, -1) > now):
                continue
            rule = self._rule(card, latest)
            if rule is None:
                continue
            sources = [j for j in active.values() if j["params"].get("candidate_id") == candidate
                       and j["kind"] in ("inspect", "observe")]
            if (candidate in self.handoff_reservations or any(j["kind"] == "inspect"
                    or aircraft[j["drone_id"]]["camera_quality"] >= .8 for j in sources)):
                continue
            source = sources[0] if sources else None
            if candidate in self.reports:
                if latest["object_type"] != "dynamic" or source is None and candidate in self.observed:
                    continue
                kind = "observe"
            elif candidate in self.confirmed:
                # Class resolved: a report is pending, or capacity is exhausted.
                continue
            elif (candidate in self.inspect_finished
                    and now-self.inspect_finished[candidate] < self.settings.confirmation_delivery_wait_seconds):
                # Physical completion can precede the last packet. Wait a finite
                # interval; unresolved class ties can then request new views.
                continue
            else:
                kind = "inspect"
            tasks.append(dict(candidate=candidate, observation=latest, rule=rule, kind=kind,
                              source=source, priority=rule["priority"]))
        return tasks

    def _cost(self, task, aircraft, now, active):
        distance = math.dist(aircraft["position"][:2], task["observation"]["estimate_xy"])
        age = now-task["observation"]["measured_at"]
        return 10.+distance/17.+.5*age+(8. if aircraft["drone_id"] in active else 0.)

    def _allocate(self, tasks, available, active, state):
        raise NotImplementedError

    def decide(self, state):
        self._ingest(state)
        commands = self._reports(state)
        now, card = state["sim_time"], state["task_card"]
        aircraft = {a["drone_id"]: a for a in state["aircraft"]}
        active = {j["drone_id"]: j for j in state["jobs"] if j["status"] in ("queued", "running")}
        available = []
        exit_at = card["deadline"]-self.settings.exit_lead_seconds
        for i, row in sorted(aircraft.items()):
            if i in self.pending or row["retired"] or not row["link_available"] or now-row["measured_at"] > 3.:
                self.fsm_states[i] = "UNAVAILABLE"
                continue
            job = active.get(i)
            home_y = card["regions"][i]["center_y"]
            reserve = card["exit_range_reserve_m"]+1000.+2*math.dist(row["position"][:2], (0., home_y))
            if now >= exit_at or row["remaining_range_m"] <= reserve:
                self.fsm_states[i] = "EXIT"
                if (job and job["kind"] == "exit" or any(j["drone_id"] == i and j["kind"] == "exit"
                        and j["status"] == "succeeded" for j in state["jobs"])):
                    continue
                commands.append(("enter_safe_exit", {"drone_id": i, "expected_version": row["job_version"]}))
                self.pending.add(i)
                continue
            if row["camera_quality"] < .8:
                self.fsm_states[i] = "HANDOFF_REQUIRED" if job and job["kind"] == "observe" else "SENSOR_RECOVERY"
                continue
            if any(r["drone_id"] == i for r in self.handoff_reservations.values()):
                self.fsm_states[i] = "HANDOFF_RECEIVER_PENDING"
                continue
            if job and (job["kind"] != "survey" or now-(job["started_at"] or job["created_at"]) < self.settings.minimum_commit_seconds):
                self.fsm_states[i] = job["kind"].upper()
                continue
            available.append(row)
        tasks = self._tasks(state, active, aircraft)
        assignments = self._allocate(tasks, available, active, state)
        assigned = set()
        for row, task in assignments:
            i = row["drone_id"]
            source = task["source"]
            if source is not None and (source["drone_id"] == i or not aircraft[source["drone_id"]]["link_available"]):
                continue
            assigned.add(i)
            if i in active:
                job = active[i]
                commands.append(("cancel_job", {"job_id": job["job_id"], "expected_version": row["job_version"]}))
                self.pending_cancellations[job["job_id"]] = i
                self.pending.add(i)
                self.fsm_states[i] = "REASSIGN_AFTER_CANCEL"
                continue
            args = dict(drone_id=i, candidate_id=task["candidate"], expected_version=row["job_version"],
                        rule_id=task["rule"]["rule_id"], task_card_version=card["task_card_version"])
            if task["kind"] == "observe":
                args.update(duration=min(card["dynamic_requirement"]["window_seconds"], exit_at-now),
                            report_id=self.reports[task["candidate"]])
                if source is not None:
                    args = {k: v for k, v in args.items() if k not in ("drone_id", "expected_version")}
                    args.update(from_drone=source["drone_id"], to_drone=i,
                                expected_from_version=aircraft[source["drone_id"]]["job_version"],
                                expected_to_version=row["job_version"])
                    name = "handoff_observation"
                else:
                    name = "observe_track"
            else:
                name = "inspect_candidate"
            commands.append((name, args))
            self.pending.add(i)
            self.fsm_states[i] = "HANDOFF_RECEIVER" if source else task["kind"].upper()
        for row in available:
            i = row["drone_id"]
            if i in assigned or i in active:
                continue
            regions = card["regions"]
            region = min(regions, key=lambda r: (self.survey_requests[r["region_id"]],
                abs(row["position"][1]-r["center_y"]), r["region_id"]))
            args = dict(drone_id=i, region_id=region["region_id"], duration=min(100., exit_at-now),
                        expected_version=row["job_version"], rule_id=card["rules"][0]["rule_id"],
                        task_card_version=card["task_card_version"])
            commands.append(("assign_survey", args))
            self.survey_requests[region["region_id"]] += 1
            self.pending.add(i)
            self.fsm_states[i] = "SEARCH"
        self.calls += len(commands)
        return commands

    def result(self, name, args, response):
        super().result(name, args, response)
        if name == "handoff_observation":
            self.pending.discard(args["to_drone"])
            if response.get("status") == "accepted":
                self.observed.add(args["candidate_id"])
                self.handoff_reservations[args["candidate_id"]] = {
                    "job_id": response["job_id"], "drone_id": args["to_drone"]}


class UtilityAuctionPolicy(_PublicClassicPolicy):
    """B1: re-auction available work with causal value/cost bids every snapshot."""

    def _allocate(self, tasks, available, active, state):
        bids = []
        for task in tasks:
            value = task["priority"]*(self.settings.dynamic_value if task["observation"]["object_type"] == "dynamic" else 1.)
            for row in available:
                cost = self._cost(task, row, state["sim_time"], active)
                bids.append((-value/cost, task["candidate"], row["drone_id"], row, task))
        assigned, candidates, result = set(), set(), []
        for _, candidate, drone, row, task in sorted(bids, key=lambda bid: bid[:3]):
            if drone not in assigned and candidate not in candidates:
                assigned.add(drone)
                candidates.add(candidate)
                result.append((row, task))
        return result


class EventStatePolicy(_PublicClassicPolicy):
    """B2: priority/confirmation/handoff states with nearest feasible dispatch."""

    def _allocate(self, tasks, available, active, state):
        free = {r["drone_id"]: r for r in available}
        result = []
        ordered = sorted(tasks, key=lambda t: (t["source"] is None, -t["priority"],
            t["kind"] != "observe", t["observation"]["measured_at"], t["candidate"]))
        for task in ordered:
            if not free:
                break
            row = min(free.values(), key=lambda a: (self._cost(task, a, state["sim_time"], active), a["drone_id"]))
            result.append((row, task))
            del free[row["drone_id"]]
        return result
