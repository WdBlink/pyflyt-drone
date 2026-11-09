"""Candidate B: one timed motion graph, causal observation rings, atomic fleet plans.

CALLER USAGE FIRST. This file is a design contract, not executable production code.

1. Actor usage stays at mission level; region is not an aircraft-owned resource:

    ack = mission.tool("assign_survey", {
        "drone_id": 0, "region_id": "stripe-3", "duration": 90.0,
        "expected_version": snapshot["aircraft"][0]["job_version"],
        "rule_id": "static-0", "task_card_version": card_version,
    })
    # accepted == queued. Public progress can say waiting_traffic / transiting /
    # surveying; only actual entry + a complete physical lap satisfies the task.

2. Causal observation and overlapping handoff use existing actor tools:

    ack = mission.tool("observe_track", {
        "drone_id": 0, "candidate_id": delivered_obs["candidate_id"],
        "duration": 30.0, "report_id": existing_report_id,
        "expected_version": current_version, "rule_id": "dynamic-inspection",
        "task_card_version": card_version,
    })
    handoff_ack = mission.tool("handoff_observation", {
        "from_drone": 0, "to_drone": 2,
        "candidate_id": delivered_obs["candidate_id"], "duration": 30.0,
        "report_id": existing_report_id,
        "expected_from_version": source_version,
        "expected_to_version": destination_version,
        "rule_id": "dynamic-inspection", "task_card_version": card_version,
    })
    # Source keeps observing until a qualified new receiver sample exists.
    # ACK, geometric proximity and a predicted camera cone cannot complete it.

3. Worker integration; the mission's existing clock/sensor transport remains:

    executor = FleetExecutor(Airspace.from_public_card(card))
    # Within InspectionMission.step(), events/queued actor requests are applied
    # by JobBook first. No planner call may block the physical clock.
    frame = FleetFrame(
        now=flight.t, navigation=flight.navigation_snapshot(),
        orders=jobs.execution_snapshot(), local_views=local_view_inbox.drain(),
    )
    decision = executor.tick(frame)
    jobs.apply_execution_events(decision.events)  # exact token/version guards
    camera_commands = decision.camera
    flight.step(targets=decision.targets)        # all four setpoints, one world
    # after both physics integrations: sensor sampling uses camera_commands;
    # local sensor records enter next tick's inbox, independent of actor link.
    # Existing immutable Evidence + delayed report transport remain mission owned.

Module map after implementation:
- execution.py: this one public tick surface; private estimator, motion graph,
  fleet reservation, service monitor and safe continuation share one owner.
- flight.py/control.py: navigation snapshot, physics/controller and per-240 Hz
  audit only; remove mutable modes/routes/headings/exit state as command inputs.
- mission.py/protocol.py: validate actor RPCs; JobBook owns versions/status;
  sensor/transport/ledger own evidence. Route inputs contain no entity keys.
No module may directly write executor route state or call planner internals.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Generic, NewType, TypeAlias, TypeVar

from envs.inspection.control import FlightTarget


Seconds = NewType("Seconds", float)
Metres = NewType("Metres", float)
RegionId = NewType("RegionId", str)
CandidateHandle = NewType("CandidateHandle", str)
EvidenceId = NewType("EvidenceId", str)
ReportId = NewType("ReportId", str)
XY: TypeAlias = tuple[float, float]
XYZ: TypeAlias = tuple[float, float, float]
T = TypeVar("T")


class AircraftId(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3


@dataclass(frozen=True)
class Four(Generic[T]):
    """Fixed fleet shape; tuple index is AircraftId, never a region identifier."""
    values: tuple[T, T, T, T]


@dataclass(frozen=True)
class Region:
    region_id: RegionId
    center_y: Metres
    lower_y: Metres
    upper_y: Metres


@dataclass(frozen=True)
class Airspace:
    """Immutable projection of the public card plus qualified control limits.

    No generator, seed, target speed, hidden count, appearance time or scorer.
    Provisional tube/bank assumptions must be supplied by fresh qualification;
    a planner cannot mint a valid safety certificate by choosing them itself.
    """
    regions: tuple[Region, ...]
    fence_min_xy: XY
    fence_max_xy: XY
    altitude_limits: tuple[Metres, Metres]
    airspeed_limits: tuple[float, float]
    minimum_separation: Metres
    deadline: Seconds
    exit_reserve: Metres
    sample_period: Seconds
    observation_bin: Seconds
    observation_grace: Seconds

    @classmethod
    def from_public_card(cls, public_card: dict) -> Airspace:
        """Boundary parsing: finite values, nonempty unique regions, legal bounds."""
        raise NotImplementedError


@dataclass(frozen=True)
class JobToken:
    job_id: str
    aircraft: AircraftId
    version: int


@dataclass(frozen=True)
class ObservedSeed:
    """Only a measurement already delivered to this request's knowledge scope.

    Direct actor requests may seed from public delivered observations. Handoff
    may transfer that seed, not the source's undelivered private estimator.
    """
    candidate: CandidateHandle
    evidence_id: EvidenceId
    measured_at: Seconds
    available_at: Seconds
    estimate_xy: XY
    covariance_xy: tuple[float, float]


@dataclass(frozen=True)
class Survey:
    region: RegionId
    duration: Seconds


@dataclass(frozen=True)
class Inspect:
    seed: ObservedSeed


@dataclass(frozen=True)
class Observe:
    seed: ObservedSeed
    duration: Seconds
    report: ReportId
    handoff_from: JobToken | None = None


@dataclass(frozen=True)
class SafeExit:
    """Stable public exit station is independent of current survey region."""
    reason: str


Intent: TypeAlias = Survey | Inspect | Observe | SafeExit


@dataclass(frozen=True)
class Order:
    token: JobToken
    accepted_at: Seconds
    intent: Intent


@dataclass(frozen=True)
class OrderSlot:
    """JobBook is the sole writer. Cancellation increments current_version.

    Absent active order means no mission action; the aircraft still follows
    the executor's reserved physical continuation, never an unbounded heading.
    """
    current_version: int
    active: Order | None


@dataclass(frozen=True)
class Navigation:
    aircraft: AircraftId
    position: XYZ
    ground_velocity: XYZ
    air_heading: float
    bank: float
    local_wind_xy: XY
    remaining_range: Metres


@dataclass(frozen=True)
class LocalView:
    """Truth-free camera output, available onboard before actor uplink.

    available_at is LOCAL availability, not Observation.arrived_at. No record
    may reach tick before it exists. The separate evidence packet keeps its
    original measured_at and eventual actor arrived_at, including after cancel.
    Association uses measured space/covariance; it cannot use evaluator IDs.
    """
    aircraft: AircraftId
    evidence_id: EvidenceId
    candidate: CandidateHandle
    measured_at: Seconds
    available_at: Seconds
    estimate_xy: XY
    covariance_xy: tuple[float, float]
    quality: float
    aircraft_xy: XY
    sampled_under: JobToken | None


@dataclass(frozen=True)
class FleetFrame:
    now: Seconds
    navigation: Four[Navigation]
    orders: Four[OrderSlot]
    local_views: tuple[LocalView, ...]


@dataclass(frozen=True)
class Nadir:
    pass


@dataclass(frozen=True)
class PointAt:
    candidate: CandidateHandle
    estimate_xy: XY
    support: tuple[EvidenceId, ...]
    estimate_at: Seconds


Camera: TypeAlias = Nadir | PointAt


class Phase(Enum):
    WAITING_TRAFFIC = "waiting_traffic"
    TRANSITING = "transiting"
    ACQUIRING = "acquiring"
    SURVEYING = "surveying"
    OBSERVING = "observing"
    COASTING = "coasting"
    REACQUIRING = "reacquiring"
    EXITING = "exiting"


@dataclass(frozen=True)
class Progress:
    """Every event is keyed by the exact token; JobBook ignores stale tokens.

    status is queued/running/succeeded/failed; cancel remains JobBook-owned.
    A handoff_acquired fact asks JobBook to cancel exactly handoff_from, iff it
    is still the same running observation. Never cancel its successor job.
    """
    token: JobToken
    at: Seconds
    status: str
    phase: Phase
    reason: str
    event_sequence: int
    supporting_evidence: tuple[EvidenceId, ...] = ()
    handoff_from: JobToken | None = None


@dataclass(frozen=True)
class Gap:
    token: JobToken
    started_at: Seconds
    ended_at: Seconds | None
    reason: str
    event_sequence: int
    # sampling, low_quality, association_ambiguous, lost, camera_unavailable,
    # waiting_traffic, transiting, cancellation, retirement, deadline


@dataclass(frozen=True)
class FleetDecision:
    targets: Four[FlightTarget]
    camera: Four[Camera]
    events: tuple[Progress | Gap, ...]
    plan_revision: int


class FleetExecutor:
    """Single writer of all trajectory/estimator/reservation/service state.

    tick is bounded work at the existing 120 Hz control cadence. It handles:
    version reconciliation, causal local measurements, incremental search,
    atomic four-aircraft plan commit, path following and service facts.
    It never steps physics, owns truth, selects tasks/reports or grades results.

    _committed is always a coherent four-aircraft plan with finite prefixes and
    compatible periodic tails; task cancellation changes ownership/monitoring,
    not the need to fly. A new estimate cannot move an active path center by
    assignment: it proposes a certified suffix replacement with continuous
    capture, while the committed prefix and fallback remain valid.
    """
    def __init__(self, airspace: Airspace):
        raise NotImplementedError

    def tick(self, frame: FleetFrame) -> FleetDecision:
        """Validate monotonic clock, four IDs, versions and local availability.

        Process bounded search expansions; if there is no certified replacement,
        continue the committed plan. Report queued/failed with public geometry,
        conflict, stale-track, range or deadline reasons. No sleep or solver wait.
        If real flight leaves the qualified tube, mark execution uncertified;
        fly its best bounded continuation and fail qualification, never hide it.
        """
        raise NotImplementedError


# PRIVATE SHAPE BELOW. One module owns these, with no public stage-by-stage API.
@dataclass(frozen=True)
class _Pose:
    xy: XY
    course: float
    altitude: Metres


@dataclass(frozen=True)
class _Line:
    start: _Pose
    length: Metres


@dataclass(frozen=True)
class _Arc:
    start: _Pose
    signed_radius: Metres
    angle: float


@dataclass(frozen=True)
class _MovingRing:
    """Causal center/velocity segment, not pursuit of an instantaneous point.

    Typical proposed observation geometry: altitude 96 m, radius 90 m, 18 m/s
    airspeed; feasibility is checked from estimated motion and uncertainty, not
    an assumed scenario speed. Use constant-air-speed translated-ring feedforward
    derived in probe_geometry.py. Ramp/capture transients need new qualification.
    Segment validity is short; its suffix includes a target-independent exit.
    """
    center: XY
    velocity_xy: XY
    estimate_at: Seconds
    valid_until: Seconds
    radius: Metres
    altitude: Metres
    support: tuple[EvidenceId, ...]
    airspeed: float = 18.0


_Primitive: TypeAlias = _Line | _Arc | _MovingRing


@dataclass(frozen=True)
class _PeriodicTail:
    """A real closed loop with absolute phase, not stationary waiting.

    Terminal tails are the four original separated R=80 exit circles; their
    destinations do not constrain survey assignment. Finite waiting/service
    capsules (R=80, half-length=250) are expanded in the prefix, which must
    leave time for this terminal circle before the public deadline. Terminal
    tails have phase-independent 40m geometric clearance; finite-prefix vs
    tail conflicts still need time/phase checks. Shared capsules need spacing.
    """
    loop: tuple[_Line | _Arc, ...]
    phase_at: Seconds
    period: Seconds


@dataclass(frozen=True)
class _Tube:
    horizontal: Metres
    vertical: Metres
    # Proposed hypotheses: horizontal<=6m, vertical<=2m, not yet qualified.


@dataclass(frozen=True)
class _Program:
    aircraft: AircraftId
    owner: JobToken | None
    starts_at: Seconds
    prefix: tuple[_Primitive, ...]
    tail: _PeriodicTail
    tracking_tube: _Tube
    # The currently executable task prefix and its target-independent safe
    # continuation are a single program. Do not keep a second stale backup map.


@dataclass(frozen=True)
class _CertifiedFleet:
    revision: int
    programs: Four[_Program]
    qualified_envelope_id: str
    # Constructor is private to _certify: interval geometry, 15m separation plus
    # both qualified tubes, altitude/speed/bank, deadline/range+exit reserve.


@dataclass
class _ActorState:
    """Per-aircraft private state, never fused by a shared candidate dictionary."""
    token: JobToken | None
    estimator: _CausalEstimator | None
    service: _ServiceWindow | None


class _CausalEstimator:
    """Constant-velocity filter with covariance, measurement-space association.

    Alpha-beta/Kalman estimates use measured_at order, not packet arrival order;
    reject duplicates/out-of-order updates rather than rewinding flight. Bounded
    coasting grows uncertainty; stale/ambiguous measurements cannot count as
    qualified views or silently create a new candidate identity.
    """
    def update(self, view: LocalView, now: Seconds) -> None:
        raise NotImplementedError

    def predict(self, now: Seconds) -> _MovingRing | None:
        raise NotImplementedError


class _ServiceWindow:
    """Owns actual service evidence, gaps and finite deadlines.

    queued -> admitted/running -> acquisition -> service -> terminal.
    Observe duration starts at first qualified new acquisition, with explicit
    acquisition timeout and room for exit by the public mission deadline.
    Survey duration starts at measured target-route capture, not departure.
    This start-time clarification must be versioned/documented on migration.

    Observe success: requested service interval completed, a qualified view in
    each public-card observation bin (including partial final bin), no missing
    run longer than observation_grace. Predictions never fill bins. Sampling
    gaps are recorded from missing expected 2 Hz samples; visibility between
    samples is unknown. Sensor facts are distinct from upload/score success.
    Inspect success: two qualified measured aircraft positions >=20m apart.
    Exit success: actual continuous 20s physical dwell with radius/course bounds.
    """
    def consume(self, views: tuple[LocalView, ...], now: Seconds) -> tuple[Progress | Gap, ...]:
        raise NotImplementedError


class _MotionGraph:
    """Exact integrated forward-only primitives; discretization is search only.

    Nodes include measured current pose, survey loop ports, terminal safe loops,
    and task-created observation-ring capture poses. Edges are circles/lines,
    with roll/capture tolerances supplied by qualification. Hybrid A* retains
    continuous endpoints; spatial/heading bins never snap physical state.
    An approximate goal must enter a qualified capture envelope or be rejected.

    All region-pair edges include the constructive R=100 S connector below.
    Arbitrary pose -> ring/survey/exit connectors use bounded incremental search
    over straight/left/right primitives, both turn senses, R>=100 at 20m/s
    transit; the selected observation ring uses R=90 at 18m/s. Speed changes
    are flown and qualified; never instant changes to aircraft velocity.
    Lattice refinement is bounded; search failure is public, not infeasibility
    proof. Initial compatibility and returning safely must be tested separately.
    """
    def stripe_connector(self, source: Region, destination: Region) -> tuple[_Primitive, ...]:
        """(-250,c-80,E) -> (-50,c'-80,E): quarter/vertical/quarter.

        R=100, |c'-c|>=200; length=pi*100+|c'-c|-200. Valid for every pair
        of the four public stripes, in x[-250,-50], y[-380,220]. Waiting for a
        departure port means flying the existing full loop and reserving it.
        """
        raise NotImplementedError


class _IncrementalPlanner:
    """Owned, bounded search state; does not advance the simulation clock.

    Existing programs are constraints. FIFO/job-token priority and trying later
    departure phases solve traffic admission only; never choose another task,
    candidate or region on behalf of the actor. Joint suffix changes may move
    holding phases to avoid deadlock, but cannot cancel another mission silently.
    Every candidate includes its full continuation to compatible periodic tails.
    """
    def advance(self, frame: FleetFrame, committed: _CertifiedFleet,
                max_expansions: int) -> _CertifiedFleet | None:
        raise NotImplementedError


def _certify(programs: Four[_Program], frame: FleetFrame,
             airspace: Airspace) -> _CertifiedFleet | None:
    """Continuous-time bounds, then atomic replace of the entire fleet plan.

    Fence erosion uses the qualified position tube; horizontal pair distance >=15+ri.horizontal+rj.horizontal.
    Never credit the existing 8m altitude steps as complete 15m separation.
    For sampled interval bounds, subtract (vmax_i+vmax_j)*dt/2, not just endpoint
    checks. Example dt=.05s, vmax_ground=27: extra clearance 1.35m. Closed
    periodic tails need an all-phase bound or rejection, never horizon truncation.
    Also certify target-independent fallback, range+1000m reserve and exit dwell.
    All final tails are the four disjoint original exit circles; their static
    40m gap makes the infinite-tail check independent of unknown target futures.
    A normal-frame update can re-check a plan; it cannot invent a certificate.
    """
    raise NotImplementedError
