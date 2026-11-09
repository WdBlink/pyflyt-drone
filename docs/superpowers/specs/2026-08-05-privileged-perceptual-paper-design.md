# Privileged-to-Perceptual Fixed-Wing Paper Design

## Authority

The normative research contract is
[`ctx/spec/02-privileged-perceptual-guidance.md`](../../../ctx/spec/02-privileged-perceptual-guidance.md).
The accepted scope decision is
[`ctx/decisions/0002-privileged-perceptual-mainline.md`](../../../ctx/decisions/0002-privileged-perceptual-mainline.md).
This document records the implementation-facing architecture and does not
replace either source.

The executable identities and mutable enumerations are defined only by
`paper/protocol/method-matrix.json`,
`paper/protocol/evaluation-protocol.json`, and
`paper/protocol/loop-guard.json`.

## Design

The work is organized around one strict information boundary. A deployment
policy consumes only synchronized onboard RGB-D, proprioception, and their
actor-visible causal measurement contract.
Exact target geometry, dense obstacle clearance, realized wind, and detector
error are training-only fields. Environment termination and evaluator metrics
may use ground truth, but those channels never enter the deployed action path.

The proposed policy has five independently removable mechanisms:

1. a recurrent deployment belief built from proprioception and detector
   tracklets;
2. a training-only privileged critic;
3. auxiliary privileged belief supervision;
4. a constrained safety objective;
5. a boundary curriculum sampler that concentrates training near the
   visibility-safety conflict boundary without changing the frozen evaluation
   distribution.

Each mechanism has a semantic switch and leave-one-out cell. The protocol also
requires the critic-privilege×belief-supervision and
constraint×curriculum factorials; other cross-interactions are limitations
unless separately declared.

The evaluator owns scenario generation. It materializes a seeded scenario
manifest before an episode and passes the same manifest to ground-truth,
degraded-ground-truth, and FastSAM perception adapters. Adapters return the
same actor measurement while evaluator-only telemetry retains failure types.
FastSAM uses frozen appearance/template and temporal evidence; ambiguity is
actor-visible missing, never a mask union or truth fallback.

Training and evaluation write content-addressed manifests. Small aggregate
evidence belongs under `paper/`; large checkpoints and raw trajectories remain
ignored. Confirmatory evaluation uses the paired episode-level manifest and
estimator defined by the evaluation protocol. Any required non-completed or
corrupt record makes the primary analysis NONPASS.

## Error handling

- Any actor-visible privileged field fails the leakage test and blocks training.
- Perception adapters with unequal non-perception parameters fail scenario
  parity and block comparison.
- A missing detector artifact yields a typed unavailable cell; it never counts
  as a zero-detection scientific result unless the protocol explicitly studies
  detector outage.
- A failed safety constraint blocks superiority and SOTA language even when
  task success improves.
- Failure of the confirmatory gate produces an evidence dossier and mechanism
  analysis, not a rewritten threshold.

## Verification architecture

Contract tests cover observation permissions, adapter parity, seeded replay,
detector failure, metric semantics, and provenance. Short simulator smoke runs
precede pilot training. Pilot results may calibrate runtime and detector-noise
parameters but cannot change the confirmatory test set or claim gate. Fresh
reviewers audit the protocol, implementation, experiment ledger, statistics,
and final claim-to-evidence mapping.
