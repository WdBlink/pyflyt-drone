# Privileged-to-Perceptual Fixed-Wing Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement one task at a time,
> with a fresh independent reviewer after every implementation task.

**Goal:** Produce a reproducible, evidence-gated fixed-wing visual guidance
study and a claim-bounded paper package.

**Architecture:** A common seeded evaluator owns scenario truth and exposes
three perception adapters. Deployment policies receive a restricted actor
observation; training policies may receive a separate privileged critic input.
Experiments and writing are downstream of deterministic leakage, parity,
provenance, and scientific gates.

**Tech Stack:** Python 3.11, Gymnasium, PyFlyt/PyBullet,
Stable-Baselines3/torch, pytest, NumPy/SciPy, FastSAM/Ultralytics, LaTeX.

## Global Constraints

- Normative behavior follows
  `ctx/spec/02-privileged-perceptual-guidance.md`.
- Machine execution follows the three JSON contracts under `paper/protocol/`;
  prose does not duplicate their mutable identities or enumerations.
- Generated checkpoints, logs, weights, raw frames, and large trajectories are
  never committed.
- An implementation agent never reviews its own output.
- Ground truth may terminate and score an episode but may not enter a deployed
  actor observation.
- Confirmatory thresholds and held-out scenarios are not changed after the
  first confirmatory run starts.

## File map

- `envs/fixedwing_envs/`: common task state and perception-independent dynamics.
- `envs/perception/`: typed perception adapters and degradation models.
- `envs/observations/`: actor/critic observation contracts and leakage guards.
- `train/`: common configuration-driven baseline and proposed-method training.
- `eval/`: seeded scenario runner, ledger, metrics, and statistics gate.
- `tests/`: deterministic contract and smoke tests.
- `paper/`: protocol, aggregate evidence, figures, bibliography, and manuscript.

## Tasks

### Task 1: Freeze literature and evaluator contract

**Files:**
- Modify: `ctx/spec/02-privileged-perceptual-guidance.md`
- Create: `paper/protocol/literature-matrix.md`
- Create: `paper/protocol/method-matrix.json`
- Create: `paper/protocol/evaluation-protocol.json`
- Create: `paper/protocol/loop-guard.json`
- Create: `scripts/validate_paper_protocol.py`
- Create: `tests/test_paper_protocol.py`

**Interfaces:**
- Produces: immutable method identities, five separable mechanisms, comparator
  admission rules, episode-level pairing/statistics, causal perception,
  evidence lifecycle, loop guards, and conservative primary-source anchors.

- [ ] Verify every comparator from a primary source and classify it as matched,
  diagnostic, or contextual.
- [ ] Freeze the validation rule that selects the strongest deployment-comparable
  baseline before held-out evaluation.
- [ ] Validate contracts and positive/negative/boundary loop guards before fresh
  R1.4 review.

### Task 2: Write negative contract tests

**Files:**
- Create: `tests/test_actor_observation_contract.py`
- Create: `tests/test_perception_adapter_contract.py`
- Create: `tests/test_scenario_replay.py`
- Create: `tests/test_metric_contract.py`

**Interfaces:**
- Consumes: Task 1 protocol.
- Produces: binary oracles for privilege leakage, adapter parity, replay,
  detector failure, and metric semantics.

- [ ] Write tests that fail against the current `target_vector` flattening path.
- [ ] Write tests that reject unequal scenario manifests and silent FastSAM
  fallback.
- [ ] Run the exact tests and record the expected failures.
- [ ] Commit only the tests and their protocol fixtures.

### Task 3: Implement the common perception/evaluation seam

**Files:**
- Create: `envs/perception/`
- Create: `envs/observations/`
- Modify: `envs/fixedwing_envs/fixedwing_objlock_env.py`
- Modify: `envs/fixedwing_envs/objlock_yolo_env.py`
- Modify: `envs/fixedwing_envs/flatten_objlock_env.py`
- Create: `eval/run_objlock_benchmark.py`

**Interfaces:**
- Consumes: typed protocol and failing tests from Tasks 1–2.
- Produces: `actor_obs`, `critic_obs`, typed `PerceptionTracklet`, seeded
  `ScenarioManifest`, and append-only rollout records.

- [ ] Implement the smallest seam that makes the contract tests pass.
- [ ] Run `pytest -q` and the repository compile smoke command.
- [ ] Execute two identical short scenario replays and compare manifest hashes.
- [ ] Commit the implementation and evidence summary.

### Task 4: Implement matched baselines

**Files:**
- Create: `train/objlock_experiment.py`
- Create: `configs/experiments/`
- Create: `tests/test_baseline_matrix.py`

**Interfaces:**
- Produces: diagnostic oracle policies plus deployment-comparable feed-forward,
  recurrent, and asymmetric recurrent configurations through one entry point.

- [ ] Add one configuration per frozen method identity without changing the
  common evaluator.
- [ ] Run a bounded train/save/load/evaluate smoke for every method.
- [ ] Validate information permissions and parameter-budget reporting.
- [ ] Commit the configurations, entry point, tests, and small smoke ledger.

### Task 5: Implement the proposed policy

**Files:**
- Create: `train/policies/`
- Create: `train/curriculum/`
- Create: `tests/test_privileged_belief_policy.py`
- Create: `tests/test_safe_curriculum.py`

**Interfaces:**
- Consumes: common actor/critic contracts and scenario difficulty features.
- Produces: recurrent actor belief, training-only privileged belief losses,
  privileged value path, constraint costs, and frozen-test-independent sampler.

- [ ] Implement recurrence, critic privilege, belief supervision, constrained
  objective, and boundary curriculum behind separate semantic switches.
- [ ] Run every leave-one-out cell plus the two targeted factorial interactions.
- [ ] Prove by test that privileged tensors cannot reach action inference.
- [ ] Prove that curriculum updates training sampling only.
- [ ] Run checkpoint round-trip and short simulator smoke tests.
- [ ] Commit implementation and evidence.

### Task 6: Pilot and freeze confirmatory execution

**Files:**
- Create: `paper/protocol/pilot-report.md`
- Modify: `paper/protocol/evaluation-protocol.json`
- Create: `paper/protocol/confirmatory-manifest.json`

**Interfaces:**
- Produces: runtime estimates, detector-degradation calibration, feasibility
  verdict, and an immutable confirmatory matrix.

- [ ] Run only the predeclared pilot seeds and label every output as pilot.
- [ ] Stop if the oracle and clean visual sanity checks fail.
- [ ] Freeze the confirmatory manifest before any confirmatory training.
- [ ] Commit protocol changes and aggregate pilot evidence, not checkpoints.

### Task 7: Execute confirmatory experiments and statistics

**Files:**
- Create: `paper/results/rollout-ledger.csv`
- Create: `paper/results/aggregate.csv`
- Create: `paper/results/research-gate.json`
- Create: `eval/analyze_objlock_benchmark.py`

**Interfaces:**
- Consumes: immutable confirmatory manifest and trained checkpoints.
- Produces: complete result ledger, paired intervals/tests, safety gate, and
  admitted claim identifiers.

- [ ] Execute every expected cell or record a typed missing/failed outcome.
- [ ] Recompute statistics only from content-addressed rollout records.
- [ ] Run the scientific gate without changing thresholds.
- [ ] Commit small aggregates and provenance; keep raw artifacts external.

### Task 8: Build figures and paper package

**Files:**
- Create: `paper/paper.tex`
- Create: `paper/references.bib`
- Create: `paper/figures/`
- Create: `paper/reviewer-readiness.md`
- Create: `paper/next-steps.md`

**Interfaces:**
- Consumes: admitted claim IDs and aggregate evidence.
- Produces: manuscript or explicit NONPASS research dossier.

- [ ] Generate every result table and figure from committed aggregate data.
- [ ] Map each quantitative sentence to an admitted claim/evidence ID.
- [ ] Build LaTeX and scan references, warnings, and forbidden claims.
- [ ] Commit the claim-bounded terminal package.
