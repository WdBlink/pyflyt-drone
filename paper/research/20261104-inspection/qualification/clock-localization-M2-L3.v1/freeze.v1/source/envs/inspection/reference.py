"""B0 reference policy. Input is exclusively the serialized public stream."""

from __future__ import annotations

from collections import Counter, defaultdict
import math


class ReferencePolicy:
    def __init__(self):
        self.history = defaultdict(list)
        self.seen = set()
        self.reports = {}
        self.report_points = {}
        self.pending_reports = {}
        self.report_aliases = defaultdict(set)
        self.rejected = set()
        self.observed = set()
        self.confirmed = set()
        self.pending = set()
        self.pending_cancellations = {}
        self.calls = 0

    def result(self, name, args, response):
        if "drone_id" in args:
            self.pending.discard(args["drone_id"])
        elif name == "cancel_job" and args["job_id"] in self.pending_cancellations:
            self.pending.discard(self.pending_cancellations.pop(args["job_id"]))
        if name == "submit_or_revise_report":
            rid = args["report_id"]
            self.pending_reports.pop(rid,None)
            aliases = self.report_aliases.pop(rid,set())
            if response.get("status") == "accepted":
                for candidate in aliases:
                    self.reports[candidate] = rid
        if response.get("status") != "accepted":
            return
        if name == "submit_or_revise_report":
            self.reports[args["candidate_id"]] = args["report_id"]
            self.report_points[args["report_id"]] = {"xy":args["estimate_xy"],"object_type":args["object_type"],
                                                   "claimed_class":args["claimed_class"]}
        elif name == "observe_track":
            self.observed.add(args["candidate_id"])

    @staticmethod
    def report_request(obs):
        return {"report_id": "report-"+obs["candidate_id"], "object_type": obs["object_type"],
                "candidate_id": obs["candidate_id"], "measured_at": obs["measured_at"],
                "estimate_xy": obs["estimate_xy"], "claimed_class": obs["estimated_class"],
                "evidence_ids": [obs["evidence_id"]]}

    def decide(self, state):
        for obs in state["observations"]:
            if obs["evidence_id"] not in self.seen:
                self.seen.add(obs["evidence_id"])
                self.history[obs["candidate_id"]].append(obs)
        jobs = state["jobs"]
        for job in jobs:
            candidate = job["params"].get("candidate_id")
            if job["kind"] == "inspect" and job["status"] == "succeeded":
                self.confirmed.add(candidate)
            if job["kind"] == "inspect" and job["status"] == "failed" and candidate not in self.reports:
                self.rejected.add(candidate)
            if job["kind"] == "observe" and job["status"] in ("failed","cancelled"):
                self.observed.discard(candidate)
        commands = []
        for candidate, observations in self.history.items():
            qualified = [o for o in observations if o["sensor_mode"] == "inspect" and o["quality"] >= .8]
            if len(qualified) >= 2 and any(math.dist(qualified[0]["aircraft_xy"], o["aircraft_xy"]) >= 20 for o in qualified[1:]):
                counts = Counter(o["estimated_class"] for o in qualified[-8:])
                winner,count = counts.most_common(1)[0]
                if count <= sum(counts.values())/2:
                    continue
                latest = next(o for o in reversed(qualified) if o["estimated_class"] == winner)
                if winner == "normal":
                    self.rejected.add(candidate)
                elif (candidate not in self.reports and state["report_slots_used"]+len(self.pending_reports)
                      < state["task_card"]["report_budget"]):
                    # Per-aircraft handles are not global identities. Merge
                    # independently confirmed static locations from public
                    # measurements before consuming another report slot.
                    duplicate = next((rid for rid,row in {**self.report_points,**self.pending_reports}.items()
                                      if latest["object_type"] == row["object_type"] == "static"
                                      and math.dist(latest["estimate_xy"],row["xy"]) <= 15),None)
                    if duplicate is not None:
                        if duplicate in self.report_points:
                            self.reports[candidate] = duplicate
                        else:
                            self.report_aliases[duplicate].add(candidate)
                    else:
                        report = self.report_request(latest)
                        self.pending_reports[report["report_id"]] = {"xy":report["estimate_xy"],"object_type":report["object_type"],
                                                                   "claimed_class":report["claimed_class"]}
                        commands.append(("submit_or_revise_report", report))
        for aircraft in state["aircraft"]:
            i = aircraft["drone_id"]
            if i in self.pending or aircraft["retired"] or not aircraft["link_available"]:
                continue
            active = next((j for j in reversed(jobs) if j["drone_id"] == i and j["status"] in ("queued", "running")), None)
            version = aircraft["job_version"]
            if state["sim_time"] >= 470:
                if not active or active["kind"] != "exit":
                    # A terminal exit keeps its physical orbit without reissuing.
                    if any(j["drone_id"] == i and j["kind"] == "exit" and j["status"] == "succeeded" for j in jobs):
                        continue
                    commands.append(("enter_safe_exit", {"drone_id": i, "expected_version": version}))
                    self.pending.add(i)
                continue
            if active and active["kind"] in ("inspect", "observe", "exit"):
                continue
            candidates = []
            for candidate, observations in self.history.items():
                latest = observations[-1]
                if candidate in self.rejected or len(observations) < 2 or latest["estimated_class"] == "normal":
                    continue
                if candidate in self.confirmed and candidate not in self.reports:
                    # Completion telemetry may precede the last camera packet.
                    # Wait for that evidence instead of starting a second job.
                    continue
                if latest["drone_id"] != i or state["sim_time"]-latest["measured_at"] > 30:
                    continue
                if candidate in self.reports:
                    if latest["object_type"] != "dynamic" or candidate in self.observed:
                        continue
                candidates.append((latest["measured_at"], candidate, latest))
            selected = max(candidates, default=None)
            if selected and aircraft["camera_quality"] >= .8:
                if active:
                    commands.append(("cancel_job", {"job_id": active["job_id"], "expected_version": version}))
                    self.pending_cancellations[active["job_id"]] = i
                else:
                    _, candidate, latest = selected
                    rule = "dynamic-inspection" if latest["object_type"] == "dynamic" else next(
                        r["rule_id"] for r in state["task_card"]["rules"] if latest["estimated_class"] in r["classes"])
                    args = {"drone_id": i, "candidate_id": candidate, "expected_version": version,
                            "rule_id": rule, "task_card_version": state["task_card"]["task_card_version"]}
                    if candidate in self.reports:
                        args.update(duration=min(180.0, 470-state["sim_time"]), report_id=self.reports[candidate])
                        commands.append(("observe_track", args))
                    else:
                        commands.append(("inspect_candidate", args))
                self.pending.add(i)
            elif not active:
                commands.append(("assign_survey", {"drone_id": i, "region_id": f"stripe-{i}", "duration": 100.0,
                                                   "expected_version": version, "rule_id": "static-0",
                                                   "task_card_version": state["task_card"]["task_card_version"]}))
                self.pending.add(i)
        self.calls += len(commands)
        return commands
