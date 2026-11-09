"""Candidate B: optics-constrained fixed arc cells with tangent phase gates.

CALLER'S USAGE (the spec; existing actor tools stay unchanged)

    # mission: accept an actor's already delivered candidate, then keep stepping.
    reply = executor.submit(i, "inspect", params, expected_version, now, seed=seed)
    targets, exits, modes = executor.advance(now, own_navigation, actual_exit_dwell)
    flight.step(targets, exit_flags=exits)  # never waits for planning

    # mission: all cameras expose first; no hidden association data crosses here.
    for obs in actual_frame:
        executor.local_measurement(obs, now)
    # Two actual views separated by20m finish inspect. A later observe request
    # keeps its plane's physical arc, rather than acquiring a new target circle.
    executor.submit(i, "observe", params, next_version, now, seed=delivered_seed)

    # cancellation changes job authority; occupancy remains until physically left.
    executor.cancel(job_id, expected_version, now)
    # Receiver acquisition still must be an actual new qualified observation;
    # only the executor may cancel the exact old source token afterwards.

MODULE MAP (two edited modules, one reused representation)

execution.py owns JobBook, per-plane LocalTrack, committed RouteProgram and cell
metadata, all versions, terminal transitions, wall deadlines and replay events.
planning.py owns the pure finite arc/straight/ramp construction, optical ranking,
and complete fleet admission, in the existing separate planning process.
paths.py TimedPath/RouteProgram/fleet_clearance are reused. flight.py continues
to accept FlightTarget; mission.py continues to own frames/provenance/transport/
reporting/grading. No mission/simulator/Entity/future/grader enters a request.

This file is a type/pseudocode contract, NOT a production implementation.
All quantities marked provisional require fresh prospective qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from envs.inspection.control import FlightTarget
from envs.inspection.paths import Pose, RouteProgram, TimedPath
from envs.inspection.planning import ObservedTrack, PlanProposal, PlanRequest
from envs.inspection.protocol import Observation

__all__ = ("FlightExecutor",)

XY = tuple[float, float]


@dataclass(frozen=True)
class _Envelope:
    """One versioned input for physical admission, never a hard-safety waiver.

    No claim that the old failed6m/2m tube passed. An illustrative10m/3m
    candidate requires separation >=15+10+10=35m, and |nominal XY|<=540m;
    use fleet_clearance's continuous lower/upper bounds, not sampled distances.
    Qualification must also cover ramps, phase, speed, roll and entry capture.
    In development qualification_id is None; every output remains unqualified.
    """

    qualification_id: str | None
    xy_tube_m: float
    altitude_error_m: float
    course_error_rad: float
    speed_error_m_s: float
    roll_error_rad: float
    roll_rate_error_rad_s: float
    max_command_bank_rad: float
    ramp_seconds: float  # provisional2s, not an asserted actual roll-rate bound


@dataclass(frozen=True)
class _OwnNavigation:
    """Snapshot from the selected aircraft's actual navigation and airdata.

    Ground course differs from air heading in wind. Preserve both. At commit
    the owner checks live values against the proposed gate; no state is reset.
    """

    at: float
    xyz: tuple[float, float, float]
    ground_velocity: XY
    air_heading_rad: float
    airspeed_m_s: float
    roll_rad: float
    roll_rate_rad_s: float


@dataclass(frozen=True)
class _ViewForecast:
    """Consumer-scoped navigation forecast; deliberately NOT a certificate.

    Build only from that aircraft's local history or actually delivered seed.
    Growing position/velocity/turn allowances are development policy, calibrated
    from causal innovations and declared sensor covariance. Gaussian noise has
    no bounded support; LocalTrack's4m/s clamp is NOT a truth-speed bound.
    Age/residual growth reduces admitted work, never mints a measurement.
    """

    track: ObservedTrack
    position_allowance_m: float
    velocity_allowance_m_s: float
    acceleration_allowance_m_s2: float


@dataclass(frozen=True)
class _Cell:
    """Executor-owned navigation metadata; the target need not be at center.

    Fixed center in ground coordinates DURING an arc. Center changes only
    through a continuous finite maneuver. No current track update translates
    an already committed path. phase comes from the owned path, not target ID.
    This survives inspect->observe/new job versions as physical state only.
    """

    center_xy: XY
    radius_m: float
    direction: Literal[-1, 1]
    arc_starts_at: float
    arc_ends_at: float  # never mistake a pending ramp/runout for this circle
    navigation_handle: str  # observed candidate handle, never hidden entity ID


@dataclass(frozen=True)
class _Gate:
    """A future sample of the OLD reserved program, not a fabricated pose."""

    at: float
    pose: Pose
    ground_speed_m_s: float
    bank_rad: float
    reference_roll_rate_rad_s: float
    reference_curvature_m_inv: float
    reference_acceleration_xy: XY


@dataclass(frozen=True)
class _EditRequest:
    base: PlanRequest  # exactly4 owned programs, tokens, revision, wind, homes
    own_now: _OwnNavigation
    gate: _Gate
    cell: _Cell | None
    view: _ViewForecast | None
    envelope: _Envelope


@dataclass(frozen=True)
class _PreparedEdit:
    """One atomic replacement. None means keep ALL of the old reservation.

    proposal.program is a complete RouteProgram: immutable history to gate,
    bounded work, reserved target-independent recovery, closed home tail.
    The cell/revision cannot be committed separately from that program.
    """

    proposal: PlanProposal
    next_cell: _Cell | None
    work_ends_at: float
    forecast_expires_at: float


def _prepare_edit(request: _EditRequest) -> _PreparedEdit:
    """Pure planning worker; bounded library, no guidance or physics rollout.

    PSEUDOCODE
    1. Validate finite fields, selected consumer, measured<=available<=now,
       exact gate==old.at(commit_at), four tokens, own navigation freshness.
       Do not certify initial capture using the reference as actual navigation.
    2. survey/exit retain the existing analytic transfer/recovery constructor;
       use SAME enlarged tube/complete-fleet admission as every other job.
    3. inspect/observe:
       - FIRST TRY continuing the current cell/reference, if forecast optical
         margins remain adequate. New measurements point the real camera and
         change forecast ranking; they never demand a new target-centered CSC.
       - For first acquisition choose tangent left/right R80/R90 arcs beginning
         at the gate's current pose/heading, after an explicit curvature ramp.
         No waiting2s for confirmation before requesting this necessary turn.
       - For a held cell, consider the next <=3s phase gate (e.g.30deg phase
         marks, also allow points on straight segments; finish any <=2s ramp
         before an edit). Treat cell as active only inside its arc interval.
         Else continue its still-reserved entry maneuver or construct a new
         tangent arc from the actual future gate, never translate that maneuver.
         Insert a forward
         tangent runout, then resume a fixed circle with the same direction.
         A provisional displacement uses target velocity projected on that
         tangent and includes motion during the runout. Clamp to0..64m.
         Consider only0, half, full displacement, and R80/R90; <=16 candidates.
         At a favorable gate constant drift v along tangent and circle period T
         suggest d=v*T/(1-v/g), g=wind-corrected straight groundspeed. Reject
         nonpositive/infeasible denominators; do not extrapolate this formula
         to a target turn. Forecast the actual ramp+line+arc geometry instead.
       - Work lookahead4..12s, shortened by stale/uncertain inputs. Circle center
         stays fixed over each arc; target forecast may move anywhere in view.
         Roll-down/up shifts nominal geometry: recompute the next center from
         each ramp's END pose. Never pretend a ramp is a zero-length switch.
       - A far/unfavorably headed initial state may require a genuine view gap:
         use bounded arc/straight capture candidates ranked by earliest actual
         view opportunity, not arrival at a target circle. Replan at next gate
         using actual frames. No claim of global Dubins completeness. If the
         bounded family cannot admit a path, retain old program and fail/queue
         visibly within existing finite timeout. Do not fake timely acquisition.
    4. Optical screen EVERY point of finite work, with interpolation omission
       allowance. d_nominal+own_tube+0.3m lever-arm+target_allowance(t) must fit
       (command_altitude-z_error-0.3-0.6)*tan50deg. Narrow aim error must also
       fit the10deg half-angle with a conservative slant-range lower bound.
       Apply both constraints with actual current altitude during transitions;
      96m is a command, not permission to replace current Z. These are nominal
       view forecasts, never proof of rays, detections, class or evidence.
       Wide stale reacquisition keeps existing actual-frame association incl.
       false candidates. Wide quality>=.8 needs off-nadir<=43.30deg; a visible
       lower-quality wide frame may refresh a track but cannot count service.
    5. Rank candidates: preserve viable cell first; then earliest predicted
       qualified-view opportunity, smallest worst optical deficit, least center
       change. Existing viable work cannot be discarded for prettier centering.
       Do not optimize against hidden bin origins or a hidden entity identity.
    6. Only the best TWO candidates receive expensive recovery/fleet checks.
       Append existing recovery from the actual constructed endpoint, including
       tangent runouts, then original equal-period home tail. Require continuity
       of position/course/nominal airspeed/bank, fence, bank/speed screen,
       deadline (finite return<=565s), actual spent+future range+1000m reserve,
       and all4 complete prefixes+one JOINT periodic-tail period. Require
       continuous clearance>=15+epsilon_i+epsilon_j and fence<=550-epsilon_i.
       Keep all other programs byte-identical. A rejected new recovery leaves
       the OLD work/recovery/tail reserved. No independent emergency shortcut.
    7. Return a single immutable prepared edit or refusal. The worker never
       writes jobs/programs/cell metadata. Cost limit exhausted => refusal.
    """
    raise NotImplementedError("design sketch")


def _ramped_arc_runout(gate: _Gate, radius: float, direction: int,
                       straight_m: float, work_seconds: float,
                       wind_xy: XY, envelope: _Envelope) -> tuple[TimedPath, _Cell]:
    """One finite maneuver constructor, not a general navigation framework.

    Use constant-curvature arcs/lines and a bounded cubic curvature ramp
    k(s)=k0+(k1-k0)*(3u^2-2u^3), u=s/L. Heading is its explicit integral;
    bounded midpoint integration produces XY and TimedPath.from_samples.
    course_kinematics supplies wind-corrected timing/air-heading bank. Start
    curvature matches the old reference; final curvature matches the next arc.
    This is a geometrical kinematic construction, not controller simulation.
    Preserve actual roll/speed capture guard and fresh physical qualification.
    """
    raise NotImplementedError("design sketch")


class FlightExecutor:
    """Same public surface; exactly one writer, one nonblocking main clock.

    Existing submit/cancel/local_measurement/aim/camera_narrow/mark_handoff and
    audit_tracking semantics stay. Internally replace target-centered work
    generation and keep one _Cell per aircraft, atomically with its program.
    Only fields/methods relevant to the changed contract are sketched here.
    """

    def submit(self, drone: int, kind: str, params: dict,
               expected_version: int, now: float, *, seed: Observation | None = None,
               preempt: bool = False) -> dict:
        """ACK is not success. Local/delivered seed only; no hidden inspection.

        inspect must request an immediate viable turn if the current route is
        about to leave view; actual views can still finish it before commit.
        When inspect finishes, preserve the physical cell for a subsequent
        observe request. A new token invalidates outstanding edits, not flight.
        """
        raise NotImplementedError("retain JobBook validation; remove blind2s acquisition delay")

    def local_measurement(self, obs: Observation, now: float) -> None:
        """Retain causal/dedup LocalTrack update and actual service samples.

        Only actual new qualified samples count. inspect needs two real views
        >=20m apart; observe keeps real gaps, original job postconditions and
        original hidden180s/60bin grader unchanged. Receiver actual sample can
        cancel only the exact still-active old(source_job, source_version).
        No prediction-only timeout reset and no fabricated evidence publication.
        """
        raise NotImplementedError("existing proven protocol behavior")

    def cancel(self, job_id: str, expected_version: int, now: float) -> dict:
        """Repeat-safe terminal guard; never remove physical reservations.

        A cancelled plane follows its committed work then reserved home return.
        Repeating cancel is rejected as stale/terminal with no state mutation,
        matching current JobBook; it need not pretend the second ACK succeeded.
        enter_safe_exit may request an earlier reserved replacement. If that
        edit fails, the old target-independent return remains complete.
        """
        raise NotImplementedError("existing version and terminal transition authority")

    def advance(self, now: float, navigation, exit_dwell) -> tuple[
        list[FlightTarget], list[bool], list[str]
    ]:
        """120Hz control tick; physics still audits every240Hz integration.

        Poll <=one bounded result. Commit only when serial/revision/all4tokens
        match, result arrived before future gate, and LIVE navigation passes
        pose/course/crab/speed/roll/rate/altitude capture (provisional envelope).
        Atomic program+cell replacement, then revision++. No path reanchoring,
        teleport, pause, catch-up burst, shadow writer or optimizer in this tick.

        A retained physical cell can span task transitions, but an unapproved
        old task cannot auto-publish into a new report. Mission still seals the
        route on frame creation. Jobs succeed only through existing actual
        postconditions, never forecast_expiry/work_end/ACK.

        Request renewals early, max one outstanding. Deterministic earliest
        physical deadline plus round-robin breaks renewal starvation across4
        aircraft; no automatic global target assignment or policy advantage.
        Future gates are >=.5s away; no response/late/invalid/capture-fail keeps
        old route. Existing wall timeout/late-serial audit/process fault handling
        stays nonblocking. Planner death, stale view or ground link loss cannot
        stop physics; finite work expires into its already reserved recovery.
        Resource deadline prevents new work; all4 planes must actually dwell
        on their own exit circles20s with1000m remaining before600s.
        """
        raise NotImplementedError("existing lookup path plus atomic bounded edit")
