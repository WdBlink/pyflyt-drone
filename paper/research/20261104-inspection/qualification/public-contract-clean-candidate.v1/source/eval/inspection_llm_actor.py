"""Development B3/B4 orchestration over the same public worker messages.

This module does not invoke a model, approve a budget or bypass worker tools.
Model failures consume the flight clock; no classical fallback issues commands.
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import math

from eval.inspection_llm_protocol import ModelReplyError, PUBLIC_TOOL_CONTRACT, TOOL_FIELDS, parse_reply, serialize_public_frame, unwrap_json_fence


MAX_USER_PAYLOAD_UTF8_BYTES = 14000
MAX_PUBLIC_FRAME_UTF8_BYTES = 10000
MAX_TOOL_HISTORY_UTF8_BYTES = 1500
MAX_MEMORY_UTF8_BYTES = 2000


class ModelContextBudgetError(ModelReplyError):
    """A local policy context failure, never a free infrastructure retry."""


SYSTEM = """You coordinate four civilian fixed-wing inspection aircraft during a
600-second continuous flight. Physics continues while you reason. Use only the
public task card, received observations, job state and tool responses. Candidate
handles are local tracks, not global truth identities. Future events and truth
are unavailable. An accepted tool ACK is not job success or delivered quality.
Use expected job versions and current task-card version. Evidence is usable only
after arrival; old evidence is never a new dynamic measurement. Prioritize legal
inspection, two high-quality distinct-view confirmations, correct finite-slot
reports and timely dynamic observation, then safe exit with sufficient time.
Do not report a normal candidate. Prefer revising one existing report over
consuming duplicate slots. Tool rejection requires reading the response, not
repeating stale commands. A failed observe job can still have delivered useful
packets. Retired aircraft cannot collect new evidence. Preserve shared safety.
For safe exit request enter_safe_exit early enough for all aircraft to complete
the job before t=600; airspeed and bank remain controlled by the shared executor.
Return exactly one JSON object with commands (at most8), optional note (<=512
characters), optional plan (at most8 strings, each<=256 characters). Every command
is {"name":tool_name,"args":{all required fields}}. No Markdown or prose outside
JSON. Empty commands is valid. Tool schemas (field names must match exactly):
"""+json.dumps(TOOL_FIELDS, separators=(",", ":"))+"""
assign_survey region_id is stripe-0..stripe-3, duration is seconds.
These stripes partition the ROI by nearest y center (-300,-100,100,300)m;
equal-distance boundaries belong to the lower index. This is survey responsibility.
inspect/observe use a received candidate_id and an applicable rule_id from the
task card. observe/handoff require an accepted report_id and duration<=180s.
Reports use object_type static/dynamic, actual evidence measured_at/estimate_xy,
claimed_class from the task card and evidence_ids of the received observations.
Copy received evidence IDs, job IDs and versions exactly; choose first report
IDs as described below and copy accepted report IDs exactly thereafter.
"""+PUBLIC_TOOL_CONTRACT


class PublicLLMActor:
    """One in-flight request; bounded public memory; deterministic scheduling."""

    def __init__(self, architecture, *, max_calls=32, max_request_seconds=60., ablation=None):
        if (architecture not in ("B3", "B4") or type(max_calls) is not int or not 1 <= max_calls <= 32
                or type(max_request_seconds) not in (int, float) or not 0 < max_request_seconds <= 60
                or ablation not in (None, "periodic", "no-cross-aircraft-memory", "no-progressive-confirmation-guidance")
                or ablation is not None and architecture != "B4"):
            raise ValueError("declare bounded B3/B4 and at most one B4 ablation")
        self.architecture = architecture
        self.ablation = ablation
        self.max_calls, self.max_request_seconds = max_calls, max_request_seconds
        self.state = None
        self.observations = OrderedDict()
        self.tool_results = []
        self.plan, self.note = [], ""
        self.history = []
        self.inflight = None
        self.last_started = -math.inf
        self.attempts = 0
        self.pending_triggers = {"initial"}
        self.previous_signature = None
        self.candidate_ids = set()

    def ingest(self, state):
        # Validate the exact source boundary before retaining or projecting it.
        serialize_public_frame(state)
        if self.state is not None and state["sim_time"] < self.state["sim_time"]:
            raise ValueError("public state clock regressed")
        for obs in state["observations"]:
            self.observations[obs["evidence_id"]] = copy.deepcopy(obs)
            if obs["candidate_id"] not in self.candidate_ids:
                self.pending_triggers.add("new_public_candidate")
                self.candidate_ids.add(obs["candidate_id"])
        while len(self.observations) > 256:
            self.observations.popitem(last=False)
        signature = (state["task_card"]["task_card_version"],
            tuple((a["drone_id"], a["retired"], a["link_available"], a["camera_quality"]) for a in state["aircraft"]),
            tuple((j["job_id"], j["status"]) for j in state["jobs"]), state["report_slots_used"])
        if self.previous_signature is not None and signature != self.previous_signature:
            self.pending_triggers.add("public_event_or_job_change")
        self.previous_signature = signature
        self.state = copy.deepcopy(state)

    def result(self, name, args, response):
        self.tool_results.append(copy.deepcopy({"name": name, "args": args, "response": response}))
        self.tool_results = self.tool_results[-12:]
        if response.get("status") != "accepted":
            self.pending_triggers.add("tool_rejection")
        elif name == "submit_or_revise_report":
            self.pending_triggers.add("report_receipt")

    def due(self):
        if self.state is None or self.inflight is not None or self.attempts >= self.max_calls:
            return False
        elapsed = self.state["sim_time"]-self.last_started
        if self.state["sim_time"] >= 590:
            return False
        return elapsed >= 20 or (self.architecture == "B4" and self.ablation != "periodic"
            and bool(self.pending_triggers) and elapsed >= 10)

    def _static_aliases(self):
        groups = []
        for obs in self.observations.values():
            if obs.get("object_type") != "static" or "estimate_xy" not in obs:
                continue
            group = next((g for g in groups if math.dist(obs["estimate_xy"], g["estimate_xy"]) <= 15), None)
            if group is None:
                group = {"candidate_ids": [], "estimate_xy": obs["estimate_xy"], "evidence_ids": []}
                groups.append(group)
            if obs["candidate_id"] not in group["candidate_ids"]:
                group["candidate_ids"].append(obs["candidate_id"])
            group["evidence_ids"] = (group["evidence_ids"]+[obs["evidence_id"]])[-4:]
        return groups[-16:]

    def _public_projection(self):
        frame = copy.deepcopy(self.state)
        # Same causal observation retention and public state projection for B3
        # and B4. Neither gets the hidden source or a larger context allowance.
        frame["observations"] = sorted(self.observations.values(),
            key=lambda r: (r["measured_at"], r["evidence_id"]))[-32:]
        active = [j for j in frame["jobs"] if j["status"] in ("queued", "running")]
        terminal = [j for j in frame["jobs"] if j["status"] not in ("queued", "running")][-8:]
        frame["jobs"] = terminal+active
        # No numeric truncation or fabricated observations; drop oldest public
        # entries consistently when the shared textual context allowance fills.
        while len(serialize_public_frame(frame).encode("utf-8")) > MAX_PUBLIC_FRAME_UTF8_BYTES:
            if len(terminal) > 4:
                terminal.pop(0)
                frame["jobs"] = terminal+active
            elif frame["observations"]:
                frame["observations"].pop(0)
            elif terminal:
                terminal.pop(0)
                frame["jobs"] = terminal+active
            else:
                raise ModelContextBudgetError("essential public card/telemetry/active jobs exceed the shared payload allowance")
        return frame

    def start_request(self):
        if not self.due():
            raise ValueError("request is not due or another request is unresolved")
        self.attempts += 1
        self.last_started = self.state["sim_time"]
        triggers = sorted(self.pending_triggers)
        self.pending_triggers.clear()
        frame = self._public_projection()
        if self.architecture == "B3":
            instruction = "Use one ReAct decision: briefly assess received state, then choose the next tool commands."
            memory = {"recent_exchanges": self.history[-2:]}
        else:
            instruction = ("Plan, execute and monitor: update the short persistent mission plan against actual job/report "
                "outcomes and current events; reassign when a public failure or capacity change invalidates it. "
                "Return the revised plan and immediate tool commands. A plan string is not an executed job.")
            memory = {"plan": self.plan, "note": self.note, "monitor_triggers": triggers}
            if self.ablation == "no-cross-aircraft-memory":
                memory = {"monitor_triggers": triggers}
            else:
                memory["public_static_alias_groups"] = self._static_aliases()
        content = copy.deepcopy({"public_state": frame, "recent_tool_results": self.tool_results[-8:], "memory": memory})
        def byte_count(value):
            return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8"))
        # Fixed disjoint allowances keep public state/tool retrieval identical
        # under a memory ablation; freed memory bytes never buy extra perception.
        while byte_count(content["recent_tool_results"]) > MAX_TOOL_HISTORY_UTF8_BYTES:
            content["recent_tool_results"].pop(0)
        def encoded_content():
            return json.dumps(content, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        while byte_count(content["memory"]) > MAX_MEMORY_UTF8_BYTES:
            history = content["memory"].get("recent_exchanges", [])
            aliases = content["memory"].get("public_static_alias_groups", [])
            plan = content["memory"].get("plan", [])
            if history:
                history.pop(0)
            elif aliases:
                aliases.pop(0)
            elif content["memory"].get("note"):
                content["memory"]["note"] = ""
            elif plan:
                plan.pop()
            else:
                raise ModelContextBudgetError("bounded memory projection cannot fit")
        if byte_count(content) > MAX_USER_PAYLOAD_UTF8_BYTES:
            raise ModelContextBudgetError("shared whole-payload allowance exceeded")
        system = SYSTEM
        if self.ablation == "no-progressive-confirmation-guidance":
            system = system.replace("two high-quality distinct-view confirmations, ", "")
        messages = [{"role": "system", "content": system+instruction},
                    {"role": "user", "content": encoded_content()}]
        self.inflight = {"request_ordinal": self.attempts, "source_sim_time": frame["sim_time"],
            "messages": messages, "messages_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()}
        return copy.deepcopy(self.inflight)

    def finish_request(self, response):
        if self.inflight is None:
            raise ValueError("no request is in flight")
        request, self.inflight = self.inflight, None
        if self.state["sim_time"] >= 600:
            return [], {"status": "ignored_after_episode_deadline", "request": request}
        # A runtime/transport failure is classified by the owning runner. This
        # boundary never turns it into an automatic retry or a successful call.
        wall = response.get("wall_seconds")
        if (response.get("status") != "completed" or type(wall) not in (int, float)
                or not math.isfinite(wall) or wall < 0 or wall > self.max_request_seconds
                or response.get("deadline_exceeded") is not False):
            return [], {"status": "ignored_noncompleted_or_late", "request": request}
        framing = {"json_fence_removed": unwrap_json_fence(response.get("text"))[1]}
        try:
            reply, commands = parse_reply(response.get("text"))
            json.dumps(reply, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ModelReplyError, UnicodeError) as error:
            return [], {"status": "policy_format_failure", "reason": str(error), "request": request,
                        "wire_framing": framing}
        self.plan, self.note = reply.get("plan", []), reply.get("note", "")
        self.history.append({"source_sim_time": request["source_sim_time"], "reply": reply})
        self.history = self.history[-2:]
        return commands, {"status": "decoded_not_tool_acceptance", "request": request, "reply": reply,
                          "wire_framing": framing}
