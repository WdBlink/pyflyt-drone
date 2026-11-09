"""Candidate A: one physical executor, causal tracks, atomically reserved paths.

DESIGN SKETCH ONLY. This file does not connect to, advance, or reset PyFlyt.
Bodies are contracts/pseudocode; they intentionally raise NotImplementedError.
Types below are domain types, never actor RPC envelopes or evaluator evidence.

CALLER FIRST (the type surface below is derived from these three call sites)

1. Existing mission tool boundary, after parsing the current actor envelope:

    ack = executor.submit(SurveyTask(
        ticket=Ticket(drone=0, expected_version=7),
        authority=current_rule_authority,
        region=RegionId("stripe-3"), duration_s=80.0,
    ))
    # ack is QueuedAck or RejectedAck. QueuedAck says nothing about completion.
    # No aircraft-index/region-index equivalence check exists.

2. Mission physical worker; the world keeps advancing without planner replies:

    out = executor.advance(TickInput(
        navigation=flight.navigation_frame(),
        detections=local_sensor_packets_since_last_tick,
        public_events=events_at_this_tick,
    ))
    flight.step(out.flight)  # exactly one shared four-aircraft world step
    sensors.set_directives(out.cameras)
    # On the 2 Hz camera tick, mint detections from actual visibility/noise.
    # Executor receives only stripped observable packets next control tick.
    # Mission joins out.bindings with evaluator-private provenance, seals
    # immutable Observation/Evidence, then uses the existing transport/ledger.
    # No report identity/correctness feedback goes back into the executor.

3. Continued tracking and physical handoff, using only an observed handle:

    ack = executor.submit(ObserveTask(
        ticket=Ticket(drone=1, expected_version=11),
        authority=current_rule_authority,
        candidate=CandidateHandle("candidate-0-00012"),
        duration_s=90.0, report=ReportId("vehicle-1"),
    ))
    # Later, via existing handoff_observation tool; not a second hidden actor:
    ack = executor.submit(HandoffTask(
        source=Ticket(1, 12), destination=Ticket(2, 3),
        source_job=JobId("job-0027"),
        candidate=CandidateHandle("candidate-0-00012"),
        authority=current_rule_authority,
        duration_s=60.0, report=ReportId("vehicle-1"),
    ))
    # The source continues until a destination qualified new sample exists
    # AND both aircraft have committed separation-safe continuation paths.
    progress = executor.public_jobs()

Proposed production map:
  mission.py: tools, synthetic sensor boundary, transport, reports, truth, grade
  execution.py: this sole public domain interface, job owner, tracks, reservations
  _inspection_paths.py: pure private path construction/certification worker
  flight.py/control.py: navigation, targets, existing autopilot, shared physics/audit
  protocol.py: existing immutable actor/evidence protocol and JobBook data helper
The path worker never receives Mission, FlightWorld, seed, entities or Evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NewType, Protocol, TypeAlias


DroneId = NewType("DroneId", int)
RegionId = NewType("RegionId", str)
CandidateHandle = NewType("CandidateHandle", str)
ReportId = NewType("ReportId", str)
JobId = NewType("JobId", str)
DetectionId = NewType("DetectionId", str)
PlanId = NewType("PlanId", int)
ReservationRevision = NewType("ReservationRevision", int)
XY: TypeAlias = tuple[float, float]
XYZ: TypeAlias = tuple[float, float, float]


@dataclass(frozen=True)
class Ticket:
    drone: DroneId | int
    expected_version: int


@dataclass(frozen=True)
class RuleAuthority:
    """Minted at the mission boundary after existing public-card validation."""
    rule_id: str
    task_card_version: int


@dataclass(frozen=True)
class SurveyTask:
    ticket: Ticket
    authority: RuleAuthority
    region: RegionId
    duration_s: float


@dataclass(frozen=True)
class InspectTask:
    ticket: Ticket
    authority: RuleAuthority
    candidate: CandidateHandle


@dataclass(frozen=True)
class ObserveTask:
    ticket: Ticket
    authority: RuleAuthority
    candidate: CandidateHandle
    duration_s: float
    report: ReportId


@dataclass(frozen=True)
class HandoffTask:
    source: Ticket
    destination: Ticket
    source_job: JobId
    candidate: CandidateHandle
    authority: RuleAuthority
    duration_s: float
    report: ReportId


@dataclass(frozen=True)
class CancelTask:
    job: JobId
    expected_version: int


@dataclass(frozen=True)
class ExitTask:
    ticket: Ticket


Task: TypeAlias = SurveyTask | InspectTask | ObserveTask | HandoffTask | CancelTask | ExitTask


@dataclass(frozen=True)
class QueuedAck:
    job: JobId
    drone: DroneId
    new_version: int


@dataclass(frozen=True)
class RejectedAck:
    reason: str
    current_version: int | None


Ack: TypeAlias = QueuedAck | RejectedAck


@dataclass(frozen=True)
class AircraftNavigation:
    """Own-aircraft measured state. No Aviary or other mutable object leaks."""
    drone: DroneId
    xyz: XYZ
    velocity_xyz: XYZ
    heading_rad: float
    roll_rad: float
    pitch_rad: float
    measured_airspeed_m_s: float


@dataclass(frozen=True)
class NavigationFrame:
    control_tick: int
    measured_at: float
    aircraft: tuple[AircraftNavigation, ...]
    measured_uniform_wind_xy: XY
    distance_flown_m: tuple[float, ...]
    exit_dwell_s: tuple[float, ...]  # independently measured by flight audit


@dataclass(frozen=True)
class LocalDetection:
    """Sensor-owned packet before observed-space identity association.

    Detection IDs are opaque; the executor cannot decode seed/index/entity.
    Evaluator privately owns DetectionId -> entity-key provenance. The route
    boundary receives no counts, true class, entity object, sensor RNG or ray.
    local_available_at is distinct from the later ground transport arrival.
    """
    detection: DetectionId
    drone: DroneId
    measured_at: float
    local_available_at: float
    estimate_xy: XY
    covariance_xy: tuple[float, float, float]  # xx, xy, yy
    measured_class: str
    observed_type: str
    quality: float
    aircraft_xyz: XYZ
    camera_xyz: XYZ
    sensor_mode: str
    exposure_job: JobId | None
    exposure_version: int


class PublicEventKind(Enum):
    CARD_REVISED = "card_revised"
    AIRCRAFT_RETIRED = "aircraft_retired"
    DEADLINE = "deadline"


@dataclass(frozen=True)
class PublicExecutionEvent:
    at: float
    kind: PublicEventKind
    drone: DroneId | None = None
    new_card_version: int | None = None


@dataclass(frozen=True)
class TickInput:
    navigation: NavigationFrame
    detections: tuple[LocalDetection, ...]
    public_events: tuple[PublicExecutionEvent, ...]


@dataclass(frozen=True)
class Region:
    region: RegionId
    center_y: float
    lower_y: float
    upper_y: float


@dataclass(frozen=True)
class ExecutionMap:
    """Constructed from the public card, never from generate_dev/entities."""
    regions: tuple[Region, ...]
    flight_bounds_xy: tuple[float, float]
    altitude_bounds_agl: tuple[float, float]
    minimum_separation_m: float
    deadline: float
    range_budget_m: float
    exit_reserve_m: float
    initial_task_card_version: int
    declared_rule_ids: tuple[str, ...]
    # Declared exit homes remain independent of the active survey assignment.
    exit_centers: tuple[XY, ...]


@dataclass(frozen=True)
class TrackEstimate:
    """Derived filter state; all source samples already locally available."""
    candidate: CandidateHandle
    estimated_at: float
    last_measured_at: float
    mean_xy: XY
    velocity_xy: XY
    covariance_xy_vxy: tuple[float, ...]  # 4x4 in a fixed documented order
    source_detection_ids: tuple[DetectionId, ...]
    generation: int


@dataclass(frozen=True)
class EvidenceBinding:
    """One association decision; Mission seals immutable evidence exactly once."""
    detection: DetectionId
    candidate: CandidateHandle
    job: JobId | None
    job_version: int
    report: ReportId | None


@dataclass(frozen=True)
class ObservationGap:
    from_at: float
    until_at: float | None
    reason: str  # loss, ambiguous association, geometry, sensor degradation, etc.


@dataclass(frozen=True)
class JobProgress:
    job: JobId
    drone: DroneId
    version: int
    status: str  # existing queued/running/succeeded/failed/cancelled protocol
    phase: str  # awaiting_route/transit/acquiring/observing/recovering/exit_dwell
    work_started_at: float | None
    last_measured_at: float | None
    qualified_samples: int
    evidence_pending: bool
    empty_monitor_bins: tuple[int, ...]
    gaps: tuple[ObservationGap, ...]
    reason: str | None


@dataclass(frozen=True)
class CameraDirective:
    drone: DroneId
    exposure_job: JobId | None
    exposure_version: int
    mode: str  # survey, inspect, reacquire, off
    aim_xy: XY | None


@dataclass(frozen=True)
class AircraftCommand:
    """Domain equivalent of existing FlightTarget; flight.py adapts directly."""
    drone: DroneId
    course_rad: float
    altitude_agl_m: float
    airspeed_m_s: float
    bank_feedforward_rad: float
    exit_center: XY | None  # audit purpose; not a claim that dwell succeeded


@dataclass(frozen=True)
class ExecutionTick:
    flight: tuple[AircraftCommand, ...]  # always exactly four, shared same tick
    cameras: tuple[CameraDirective, ...]
    bindings: tuple[EvidenceBinding, ...]
    job_changes: tuple[JobProgress, ...]


class SharedFlightExecutor:
    """The only owner of jobs, versions, tracks and committed flight intent.

    Holds one JobBook privately, one per-aircraft filter/task state, and one
    single-writer reservation table. All writes happen on the physics worker.
    Async planner gets immutable snapshots and proposes; it never commits.
    """

    def __init__(self, public_map: ExecutionMap, initial_navigation: NavigationFrame):
        """Install four qualified, disjoint initial capsule continuations."""
        raise NotImplementedError

    def submit(self, task: Task) -> Ack:
        """Atomic version check + job creation; ACK means queued only.

        Boundary validates finite durations/map IDs/observed handles/report
        existence before constructing Task. This owner revalidates current
        versions, retirement and current public authority. Handoff checks both
        tickets atomically. Cancellation is terminal; a repeated stale ticket
        rejects without adding a second physical recovery. Exit preempts logical work but must
        still physically transit, reserve a path and earn audited dwell.
        """
        raise NotImplementedError

    def advance(self, inputs: TickInput) -> ExecutionTick:
        """Bounded-cost physical tick; never waits for planning/coordinator.

        1. Check monotonic tick; apply public events and available sensor packets.
        2. Associate using estimates/innovation gates; ambiguity becomes a gap.
        3. Update causal per-candidate filters. No extrapolation past this tick
           is accepted as a measurement or evidence.
        4. Poll planner pipe without waiting; commit only complete fresh fleet
           revisions at a future capture gate. Otherwise keep reserved flight.
        5. Interpolate existing committed plan and output exactly four targets.
        6. Evaluate physical/sensor job postconditions; journal transitions.

        An expired planning deadline triggers an already-reserved recovery
        continuation. No missing planner reply can stop physics or alter time.
        """
        raise NotImplementedError

    def public_jobs(self) -> tuple[JobProgress, ...]:
        """Only protocol-approved progress; never filter private provenance."""
        raise NotImplementedError


# PRIVATE planner contracts below are design aids, not the actor/public API.
# Production AST boundary tests forbid cross-importing these outside executor.


@dataclass(frozen=True)
class Pose:
    xyz: XYZ
    course_rad: float
    bank_rad: float


@dataclass(frozen=True)
class PathKnot:
    at: float
    pose: Pose
    airspeed_m_s: float


@dataclass(frozen=True)
class TrackingTube:
    horizontal_error_m: float
    altitude_error_m: float
    max_ground_speed_m_s: float


@dataclass(frozen=True)
class ReservedContinuation:
    """Work prefix plus finite recovery to an indefinitely reserved safe hold.

    Knots are a rendering of analytic/rate-limited primitives, not free
    waypoints. Geometry between knots must be certified too. The recovery is
    already included in reservations, not planned only after sensor loss.
    New observations can replace work only by atomically replacing this item.
    """
    drone: DroneId
    job: JobId | None
    job_version: int
    plan: PlanId
    activates_at: float
    capture_pose: Pose
    work_knots: tuple[PathKnot, ...]
    recovery_knots: tuple[PathKnot, ...]
    hold_region: RegionId
    hold_phase_at_arrival: float
    tube: TrackingTube
    source_track_generation: int | None
    decision_latest_at: float


@dataclass(frozen=True)
class FleetReservation:
    revision: ReservationRevision
    continuations: tuple[ReservedContinuation, ...]


@dataclass(frozen=True)
class PlanRequest:
    issued_at_tick: int
    reservation: FleetReservation
    navigation: NavigationFrame
    tasks: tuple[Task, ...]
    tracks: tuple[TrackEstimate, ...]
    public_map: ExecutionMap


@dataclass(frozen=True)
class FleetProposal:
    based_on_revision: ReservationRevision
    based_on_tick: int
    revised: FleetReservation
    latest_commit_tick: int
    rejected_jobs_and_reasons: tuple[tuple[JobId, str], ...]
    certificate: FeasibilityCertificate


@dataclass(frozen=True)
class FeasibilityCertificate:
    """Computed over continuous primitives/tubes, not just path knots.

    The private worker also performs the expensive pure geometric checks;
    physics owner checks revision/tickets/capture state and bounded-size
    certificate integrity. It does not rerun a search on the control thread.
    """
    min_pair_distance_after_tubes_m: float
    min_fence_clearance_after_tubes_m: float
    altitude_interval_after_tubes_m: tuple[float, float]
    max_required_bank_rad: float
    max_required_bank_rate_rad_s: float
    remaining_range_after_recovery_m: tuple[float, ...]
    generated_from_reservation: ReservationRevision
    valid_until_capture_tick: int


class PathPlanningWorker(Protocol):
    def propose(self, request: PlanRequest) -> FleetProposal:
        """Pure observable-only planning; may be slow, cannot mutate flight.

        Enumerate bounded-curvature/rate primitives and tangent capture gates;
        generate moving-center loiter for measured tracks; include recovery;
        certify fence, bank, climb, range and all pairwise swept tubes before
        returning. Fixed deterministic traffic priority is not task allocation.
        """
        ...


def commit_proposal(current: FleetReservation, proposal: FleetProposal,
                    navigation: NavigationFrame,
                    active_job_versions: tuple[int, ...]) -> FleetReservation:
    """Only physics-worker owner calls this pure whole-fleet admission check.

    Reject stale revisions/tickets/activation gates or capture-state mismatch.
    Require the already-running prefix through activation to be unchanged.
    Require continuous pair separation >15m+both tubes+numerical margin and
    every tube strictly inside [-550,550]^2 / AGL60..100. On rejection current
    remains authoritative. No partially committed handoff or route swaps.
    """
    raise NotImplementedError
