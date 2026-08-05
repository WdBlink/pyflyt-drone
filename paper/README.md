# Privileged-to-Perceptual Fixed-Wing Guidance Paper

This directory is the repository-local paper workspace. Scientific intent and
claim gates are maintained in
[`ctx/spec/02-privileged-perceptual-guidance.md`](../ctx/spec/02-privileged-perceptual-guidance.md).
The mainline choice remains in
[`ADR-0002`](../ctx/decisions/0002-privileged-perceptual-mainline.md), while the
causal sensor/evidence decision is recorded in
[`ADR-0003`](../ctx/decisions/0003-causal-perception-and-evidence-gates.md).

Executable method, evaluation, and heartbeat contracts live in
[`paper/protocol/`](protocol/README.md). The tracked JSON files there are the
machine-readable execution source; prose documents point to them instead of
copying mutable method, field, cell, or threshold enumerations.

No result, superiority, safety, deployment, or SOTA claim is admitted merely
because a manuscript file exists. The current maximum performance wording is
“best among the frozen evaluated internal methods,” and only its signed gate
receipt may admit it. Any incomplete, missing, corrupt, unrun, blocked, or
failed required evidence produces NONPASS.

Generated checkpoints, raw rollouts, detector weights, logs, and render outputs
remain outside Git. Small manifests, aggregate tables, figure source data, and
provenance records may be committed when they contain no machine-specific paths.
