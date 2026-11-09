"""Candidate A. Frozen causal field, predicted own motion, reserved full recovery.

DESIGN ONLY. Bodies deliberately raise NotImplementedError. No production import.
Proposed constants are unqualified and do not amend gate-contract.v11.json.

Callers first. The actor tool schemas stay unchanged. Mission parses wire requests
into the existing job domain and owns every sensor, packet, report and grader.

1. Mission._tool (the existing observe_track / inspect_candidate tools)
   receipt = executor.submit(intent, expected_version, now, delivered_seed)
   # Receipt is a JobBook ACK. It is never acquisition or observation evidence.

2. Mission.step, at120Hz, independently of planner/actor progress
   targets, exits = executor.advance(now, own_fleet, actual_exit_dwell)
   flight.step(targets, exit_flags=exits)
   # flight/control remain the only actual surface/physics integration path.

3. Mission._sample / cancellation
   aim, narrow = executor.camera(drone, now)
   observations = actual_sensor_frame(aim, narrow)  # original2Hz/ray/provenance
   executor.local_measurement(observations, now)   # one complete four-camera frame
   executor.cancel(job_id, expected_version, now)
   # Neither operation changes the reserved flight program. Only a fresh,
   # fleet-certified proposal can do that. A new measurement can change aim.

Module map (replace rather than maintain a second work execution path)
  execution.py: sole JobBook/program writer, local filters, bounded queues,
                activation/cancel/handoff, tick evaluation and auditing.
  planning.py: private worker, same field evaluator and reduced own predictor,
               finite candidate selection and complete fleet admission.
  guidance.py: one pure frozen-field law, used by real tick and predictor.
  paths.py: keep timed recovery/survey and fleet geometry; no target CSC branch.
  control.py/flight.py: keep actual controller and physics; expose a read-only
                       controller-memory snapshot to the isolated predictor.
  mission.py/protocol.py/scoring.py: retain sensing, evidence and score ownership.

There is no public plan/commit API and no alternate live-track steering API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


XY = tuple[float, float]
XYZ = tuple[float, float, float]
Drone = int
Token = tuple[int, str | None]
FourTokens = tuple[Token, Token, Token, Token]


@dataclass(frozen=True)
class JobIntent:
    drone: Drone
    kind: Literal["survey", "inspect", "observe", "exit"]
    # Existing validated domain params, no wire/IPC/proposal knobs for the actor.
    candidate: str | None
    duration_s: float | None
    region: str | None
    report_id: str | None
    rule_id: str | None
    task_card_version: int


@dataclass(frozen=True)
class ControllerMemory:
    altitude_integral: float
    speed_integral: float


@dataclass(frozen=True)
class OwnNav:
    stamp: float
    position: XYZ
    ground_velocity: XYZ  # WORLD frame; PyFlyt state[2] is BODY frame
    attitude_rpy: XYZ
    body_rates: XYZ
    measured_airspeed: float
    controller: ControllerMemory


@dataclass(frozen=True)
class CausalAnchor:
    consumer: Drone
    measured_at: float
    available_at: float  # consumer-local availability, not future ground arrival
    position: XY
    velocity: XY
    support_ids: tuple[str, ...]
    # Only the owner's LocalTrack can mint this from actual samples. Cross-
    # aircraft initialization requires an actually delivered seed. No truth ID,
    # appearance time, future path, hidden motion parameter, or grader field.


@dataclass(frozen=True)
class _Field:
    anchor: CausalAnchor
    radius_m: Literal[80, 90]
    turn: Literal[-1, 1]
    altitude_m: float = 96.
    airspeed_m_s: float = 18.
    # The anchor stays FROZEN during this committed finite segment. A local
    # camera update cannot change navigation outside a new fleet admission.


@dataclass(frozen=True)
class _Route:
    timed_path: object  # existing immutable TimedPath, own geometry only
    altitude_m: float


@dataclass(frozen=True)
class _EntryBlend:
    old_law: _Field | _Route
    new_law: _Field
    duration_s: float = .75  # hypothesis, included in predictor/tube qualification
    # Blend desired AIR-velocity vectors and feedforward bank with smoothstep.
    # Then convert to ground course; actual FlightWorld still crab-corrects it.
    # Reject low vector norm, speed outside profile, or unqualified bank slew.
    # Refresh spacing >=1s, so old_law has finished any previous entry blend.


@dataclass(frozen=True)
class _Work:
    starts_at: float
    ends_at: float
    entry: _EntryBlend | None
    law: _Field | _Route
    # Proposed6s field, refreshed no more often than1Hz. Actual duration is
    # clipped by remaining job/exit/range budget and anchor age <=8s. No live
    # target estimate is consulted at evaluate time.


@dataclass(frozen=True)
class _Recovery:
    starts_at: float
    returning: _Route
    home_tail: _Route
    tail_period_s: float
    # Fixed own geometry from work endpoint, including roll/heading/speed
    # capture and a smooth join. Home tail is the original R80,18m/s home circle.
    # No CausalAnchor is reachable through this type.


@dataclass(frozen=True)
class _OwnPrediction:
    time_s: tuple[float, ...]
    states: tuple[OwnNav, ...]
    # Numerical navigation prediction only. Cannot implement Observation,
    # Evidence, Report, or be passed to local_measurement().


@dataclass(frozen=True)
class _TubeProfile:
    identity: str
    source_controller_law_hash: str
    qualified: bool
    work_xy_by_age: tuple[tuple[float, float], ...]
    recovery_xy_by_phase: tuple[tuple[float, float], ...]
    capture_state_limits: tuple[float, ...]
    # Also identifies z/airspeed/bank bounds, entry domains, PID memory,
    # seam/slew conditions, wind<=2, and the whole600s scope. The final profile
    # is measured prospectively on development fixtures and independently
    # reviewed. No values are supplied here; old6m/2m is NOT inherited.


@dataclass(frozen=True)
class _Reservation:
    predicted: _OwnPrediction
    tube: _TubeProfile
    recovery: _Recovery
    # Covers the old immutable prefix to future commit, the finite work,
    # EVERY recovery transition and the periodic tail. It is not just work.


@dataclass(frozen=True)
class _Committed:
    revision: int
    commit_at: float
    work: _Work
    recovery: _Recovery
    reservation: _Reservation
    # Constructor private to executor's accepted-proposal reducer. No optional
    # recovery, target-dependent tail, or release-on-cancel representation.


@dataclass(frozen=True)
class _Request:
    serial: int
    fleet_revision: int
    tokens: FourTokens
    requested_at: float
    commit_at: float
    intent: JobIntent
    anchor: CausalAnchor | None
    own: tuple[OwnNav, OwnNav, OwnNav, OwnNav]
    committed: tuple[_Committed, _Committed, _Committed, _Committed]
    wind: XY
    remaining_range_m: tuple[float, float, float, float]
    tube_profile_id: str
    # Copies only. Planner has no mission/world/entity/scorer/seed references.


@dataclass(frozen=True)
class _Proposal:
    serial: int
    fleet_revision: int
    tokens: FourTokens
    commit_at: float
    replacement: _Committed | None
    rejection: str | None
    # Bounded wire bytes are parsed/validated within execution.py, never by
    # Mission or actor. Exact source/profile/request hashes join the audit.


def _field_target(field: _Field, nav: OwnNav, at: float, wind: XY):
    """The SAME function supplies actual FlightTarget and predictor input."""
    # center = frozen.position + frozen.velocity*(at-frozen.measured_at)
    # radial = unit(own.position_xy-center), tangent = turn*rotate90(radial)
    # radial_speed = clip(-.7*(range-radius), -5, 5)
    # Solve |v_center + radial_speed*radial + s*tangent - wind|=18 for s>0.
    # Return course of that ground velocity and differentiated bank feedforward.
    # No exact center/radius capture requirement and no guessed own pose.
    # Bound all finite inputs and feasible airspeed; include actual controller
    # heading feedback/saturation/entry response in prediction and qualification.
    raise NotImplementedError


def _rollout(request: _Request, work: _Work, recovery: _Recovery) -> _OwnPrediction:
    # Propagate the OLD law from actual snapshot to future commit. Do not start
    # from old nominal sample as if actual aircraft were there. Include measured
    # roll/yaw/speed/z/vz, controller integrals, crab correction, shared exact
    # outer-loop formulas and fitted closed-loop roll/pitch/speed responses.
    # Integrate fixed1/60s internal steps. Actual physics stays240Hz/control120Hz.
    # Use _field_target for both branches. Reduced dynamics are NOT physics;
    # profile bounds cover model mismatch and numerical error prospectively.
    raise NotImplementedError


def _plan(request: _Request) -> _Proposal:
    # Work is bounded and purely causal. Enumerate KEEP current continuation
    # first, then R80/R90 x CW/CCW (at most5 choices). For inspect/observe,
    # begin service eligibility at every actual2Hz camera opportunity, including
    # acquisition/entry. No planned_service_from prevents mid-entry refresh.
    # Extrapolation is navigation only; camera margin uses actual gimbal/off-
    # nadir, optical lever, altitude envelope and a calibrated target-error
    # forecast. Ray occlusion is UNKNOWN unless public evidence supports it;
    # never ask world.rayTest against hidden current/future target positions.
    # Retain/score present legal views, earliest predicted visibility, longest
    # predicted blind gap over6s, then optical margin and switch cost. Hysteresis
    # retains the current branch unless a measured loss or material gain exists.
    # Inspect additionally ranks actual potential20m view separation; only real
    # measurements may satisfy it. No hidden appearance clock enters ranking.
    # For each candidate build fixed own recovery+home tail FIRST and roll out
    # the entire candidate. The target CSC/C2 acquisition branch is deleted;
    # existing bounded CSC is allowed ONLY for target-independent transfer/home.
    # Admit candidate against all four complete reservations, including old
    # prefix, every return, and home tails. Pick best admitted candidate, not
    # first geometry pass. No candidate or unqualified profile => rejection.
    # Max5 candidates, fixed sample bound and wall deadline. Worker may stop at
    # its budget; the owner never blocks or relies on this function returning.
    raise NotImplementedError


def _fleet_admit(request: _Request, candidate: _Committed) -> bool:
    # Validate predictor interpolation slope <=27m/s and reserve min horizontal
    # separation >=15+r_i(t)+r_j(t)+27*sample_dt (dt=.1s suggested). Horizontal
    # separation is conservative for the unchanged physical15m3D requirement.
    # Fence: nominal abs XY + tube +13.5*dt <=550; z tube within60..100;
    # qualified actual airspeed/bank envelope within15..25m/s and45deg.
    # Require all endpoint state/PID-memory sets inside the recovery's qualified
    # entry domain; sample position continuity alone does not qualify a join.
    # Check fixed-timed finite prefixes plus one joint equal-period home cycle.
    # Other three programs remain immutable. Cancelled jobs retain reservations.
    # Any changed fleet revision invalidates proposal. Clearance near work end
    # does not excuse conflict during another aircraft's much later recovery.
    # Reserve actual range upper bound plus1000m and complete home dwell20s
    # before600s, using own navigation/range state. No real-battery claim.
    raise NotImplementedError


class FlightExecutor:
    """One owner of jobs, local causal navigation estimates and flight programs."""

    def submit(self, intent: JobIntent, expected_version: int, now: float,
               delivered_seed=None):
        # Preserve current JobBook version/ACK/timeout semantics. Cross-aircraft
        # seed must already have arrived. No program mutation on acceptance.
        raise NotImplementedError

    def local_measurement(self, observations, now: float) -> None:
        # Mission exposes all four real frames before this reducer. A local
        # frame is available to its own consumer now, even when its sealed
        # ground-arrival field is future. Require measured_at<=now; separately
        # require an actually arrived packet for cross-aircraft seed injection.
        # Deduplicate IDs and old timestamps.
        # Update only that aircraft's filter and same approved job's real sample
        # bins/views. Handoff source can end only after actual new receiver
        # sample passes original gates AND exact source job/version still match.
        # Samples may update camera aim immediately; navigation remains frozen.
        raise NotImplementedError

    def camera(self, drone: Drone, now: float):
        # Current local causal aim; narrow if age<=3s, otherwise existing real
        # wide reacquisition and observable association. No predicted Evidence.
        raise NotImplementedError

    def cancel(self, job_id: str, expected_version: int, now: float):
        # Idempotent checked terminal transition and version invalidation.
        # KEEP physical program/reservation. Reassignment needs a fresh full
        # replacement certificate. Cancellation never creates a free airspace.
        raise NotImplementedError

    def advance(self, now: float, own_fleet, actual_exit_dwell):
        # O(4) fixed-law evaluation + bounded poll. No geometry/search/join/wait.
        # Poll at most one bounded response; validate request/profile/hash,
        # fleet_revision and all job tokens, age, deadline and entry domain.
        # Activate only at exact future tick after ACTUAL state/PID capture
        # check; otherwise discard and continue old fully reserved program.
        # New measurement arrivals alone do not invalidate an immutable anchor.
        # At most one request outstanding, at most one ready response. Proposed
        # .5s lead is a hypothesis, measured in real1x before qualification.
        # Refresh can be requested during acquisition; request every>=1s and
        # only when benefit/remaining runway warrants it, never block on it.
        # Late results discarded. Worker no-reply5wall seconds records infra
        # failure and terminates non-blockingly; old work naturally expires to
        # already committed recovery and periodic home tail. No emergency plan
        # is synthesized in tick, no main clock pause, no early unreserved turn.
        # Ground disconnection changes transport only. Job cannot succeed from
        # predicted service or ACK. Preserve90s acquisition timeout and real bins.
        raise NotImplementedError

    def audit_physics(self, stamp: float, actual_fleet) -> None:
        # Every240Hz step checks original hard limits AND own-prediction tubes,
        # transition/PID domains and reservation validity. Any tube breach is an
        # operational qualification failure even if hard safety remains legal.
        # Stop admitting new work; retain reserved program to recovery. A breach
        # beyond the certified envelope invalidates the safety claim; no code
        # path can retroactively enlarge a bound or promise guaranteed rescue.
        raise NotImplementedError
