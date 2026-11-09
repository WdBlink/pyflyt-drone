"""Candidate section 6.2 evidence selection rules, before protocol freeze.

This pure grading component consumes evaluator-only records. It is not an
actor API and does not implement sensing, report admission or identity matching.
All required bins remain in the denominator. Tests use hand-computed fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence


def localization_quality(error: float) -> float:
    if not math.isfinite(error) or error < 0:
        raise ValueError("localization error must be finite and nonnegative")
    if error <= 3:
        return 1.0
    if error <= 5:
        return 2 / 3
    if error <= 10:
        return 1 / 3
    return 0.0


@dataclass(frozen=True)
class GradingMeasurement:
    evidence_id: str
    measured_at: float
    arrived_at: float
    estimate_xy: tuple[float, float]
    claimed_class: str
    identity_correct: bool
    quality_valid: bool
    provenance_valid: bool
    received_order: int = 0

    def __post_init__(self) -> None:
        if not self.evidence_id or not all(math.isfinite(x) for x in (
            self.measured_at, self.arrived_at, *self.estimate_xy
        )):
            raise ValueError("measurement IDs and numerical values must be valid")
        if len(self.estimate_xy) != 2 or self.measured_at < 0 or self.arrived_at < self.measured_at:
            raise ValueError("measurement must precede arrival and contain a planar estimate")


@dataclass(frozen=True)
class BinResult:
    index: int
    starts_at: float
    ends_at: float
    weight: float
    evidence_id: str | None
    quality: float
    zero_reason: str | None


def grade_dynamic_bins(
    measurements: Sequence[GradingMeasurement],
    truth_xy_at: Callable[[float], tuple[float, float]],
    true_class: str,
    appeared_at: float,
    deadline: float,
    weight_at: Callable[[float], float] = lambda t: 1.0,
    *,
    bins: int = 60,
    bin_seconds: float = 3.0,
    grace_seconds: float = 6.0,
    track: str = "delivered",
) -> tuple[float, list[BinResult]]:
    """Select legal timely latest evidence first; only then check correctness.

    The caller supplies one fixed report-to-entity identity association and all
    its frozen measurements, including incorrect ones. `arrived_at` is when the
    complete report and required evidence were available to ground auditing.
    `observed` still requires final audit upload by the episode deadline.
    """
    if track not in ("delivered", "observed"):
        raise ValueError("track must be delivered or observed")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    if not all(math.isfinite(x) for x in (appeared_at, deadline, bin_seconds, grace_seconds)):
        raise ValueError("time parameters must be finite")
    if appeared_at < 0 or bin_seconds <= 0 or grace_seconds < 0:
        raise ValueError("invalid window parameters")
    if appeared_at + bins * bin_seconds + grace_seconds > deadline:
        raise ValueError("the full observation window and grace must fit before deadline")
    ids = [m.evidence_id for m in measurements]
    if len(set(ids)) != len(ids):
        raise ValueError("a new measurement must have a unique evidence ID")
    results = []
    for index in range(bins):
        start = appeared_at + index * bin_seconds
        end = start + bin_seconds
        weight = float(weight_at(start))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("each required bin must have a finite positive demand weight")
        arrival_limit = end + grace_seconds if track == "delivered" else deadline
        candidates = [m for m in measurements
                      if (start <= m.measured_at <= end if index == 0 else start < m.measured_at <= end)
                      and m.provenance_valid and m.arrived_at <= arrival_limit]
        selected = min(candidates, key=lambda m: (-m.measured_at, m.evidence_id), default=None)
        quality, reason = 0.0, "no_legal_timely_measurement"
        if selected is not None:
            if not selected.identity_correct:
                reason = "incorrect_identity"
            elif selected.claimed_class != true_class:
                reason = "incorrect_class"
            elif not selected.quality_valid:
                reason = "insufficient_quality"
            else:
                truth = truth_xy_at(selected.measured_at)
                if len(truth) != 2 or not all(math.isfinite(x) for x in truth):
                    raise ValueError("truth must provide a finite planar position at measurement time")
                error = math.hypot(selected.estimate_xy[0] - truth[0], selected.estimate_xy[1] - truth[1])
                quality = localization_quality(error)
                reason = None if quality else "localization_error"
        results.append(BinResult(index + 1, start, end, weight,
                                 selected.evidence_id if selected else None, quality, reason))
    denominator = math.fsum(r.weight for r in results)
    return math.fsum(r.weight * r.quality for r in results) / denominator, results


def final_static_quality(measurements: Sequence[GradingMeasurement],
                         truth_xy: tuple[float, float], true_class: str, deadline: float) -> float:
    """Latest arrived legal version of one already associated static report."""
    if not math.isfinite(deadline) or deadline < 0 or len(truth_xy) != 2 or not all(math.isfinite(x) for x in truth_xy):
        raise ValueError("static truth and deadline must be valid")
    # Static versions are ordered by reception; localization/class are never
    # used to pick a version. Evidence IDs provide an explicit stable tie break.
    legal = [m for m in measurements if m.provenance_valid and m.arrived_at <= deadline]
    selected = min(legal, key=lambda m: (-m.arrived_at, -m.received_order, m.evidence_id), default=None)
    if selected is None or not selected.identity_correct or not selected.quality_valid or selected.claimed_class != true_class:
        return 0.0
    return localization_quality(math.hypot(selected.estimate_xy[0] - truth_xy[0],
                                          selected.estimate_xy[1] - truth_xy[1]))
