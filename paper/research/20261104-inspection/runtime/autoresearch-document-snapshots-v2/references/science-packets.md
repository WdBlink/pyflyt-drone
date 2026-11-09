# Science packet interface

These version 1 packets implement bounded evidence checks. They do not determine
scientific quality or understand arbitrary prose. The host still performs the
independent domain and visual reviews. All paths below are relative to the run
root and must be present in the current packet's `artifacts` list. Artifacts
in previous accepted packets may be referenced by the controller's `_history`.
Do not include private controller fields in submitted packets.

## Experiment design frozen by plan

The plan packet contains `experiment_design_sha256` equal to the SHA-256 of the
experiment design bytes. Create the design before accepting the plan. Use:

```json
{
  "experiment_id": "E1", "claim_id": "C1",
  "hypothesis": "Candidate reduces the measured error on these cases",
  "independent_variable": "method", "dependent_variable": "error",
  "control_variables": ["same inputs", "same evaluator"],
  "metric": {"name": "error", "unit": "m", "direction": "lower"},
  "statistical_unit": "case", "cases": ["a", "b", "c"], "seeds": [0],
  "conditions": ["baseline", "candidate"],
  "failure_policy": "penalty", "failure_value": 10,
  "statistics": {"estimator": "mean", "interval": "sd", "paired": true},
  "data_roles": [{"id": "dev", "role": "development", "used_for_selection": true}],
  "planned_at": "2026-11-03T00:00:00Z"
}
```

Supported independent units are cases. Repeated seeds are averaged inside each
case first; seeds are not treated as extra independent cases. Supported interval
modes are `none`, `sd`, `se`, and `ci95_normal`. The last is an explicitly
exploratory normal approximation (`mean ± 1.96 SE`), not a test of significance.
Non-`none` modes require at least two cases. Unknown estimators, nested unit
structures, missing pairings, and success-only filtering fail explicitly.

Named multi-arm designs use two or more distinct stable condition identifiers
(letters, digits, `_`, `.`, `-`; at most 96 characters). Three, seven, or more
conditions retain exactly the full case × seed × condition grid, bounded by
100,000 planned slots. No arm is relabeled or dropped, even if it is absent from
a requested contrast. The legacy `["baseline", "candidate"]` design without
`contrasts` retains its version 1 `delta` output (candidate minus baseline).
Other designs require explicit contrasts inside the frozen `statistics` plan:

```json
"conditions": ["control", "method_a", "method_b"],
"statistics": {
  "estimator": "mean", "interval": "sd", "paired": true,
  "contrasts": [
    {"id": "a_vs_control", "minuend": "method_a", "subtrahend": "control"},
    {"id": "b_vs_a", "minuend": "method_b", "subtrahend": "method_a"}
  ]
}
```

Each contrast is `minuend minus subtrahend` after averaging the same planned
seeds within each independent case. IDs and ordered pairs must be unique;
unknown arms and self-comparisons fail. Version 2 statistics contain every
condition, `condition_order`, `contrasts`, and `contrast_order`; each contrast
has its mean, SD, SE, interval, case differences and adverse case IDs. There is
no implicit primary contrast or overall winner. Explicit-contrast results must
keep `outcome: "inconclusive"`; their values remain usable descriptive evidence.
Intervals are not multiplicity-adjusted, confirmatory tests, or support for
arbitrary statistical units. This capability does not establish scientific
completion. Operational failures and unstarted slots remain recorded but block
statistics; protocol-declared measurement failures retain the frozen penalty.
The table reports every condition and contrast; figures plot every condition.
Long condition labels still require visual review of the rendered deliverable.

An experiment packet contains `design`, `results`, and `summary` paths. The
summary is a nonempty readable Markdown file. The results file contains:

```json
{
  "design": "experiment/design.json", "design_sha256": "<sha256>",
  "started_at": "2026-11-03T01:00:00Z", "outcome": "inconclusive",
  "reported_denominator": 6,
  "runs": [
    {"id": "a-base-0", "case_id": "a", "seed": 0, "condition": "baseline",
     "status": "completed", "value": 2.0, "raw": "experiment/raw/a-base-0.json"}
  ]
}
```

Include one row per planned `(case_id, seed, condition)`. `raw` is a JSON
measurement record containing those identity fields plus `status` and `value`;
the values are cross-checked, not merely cited. Statuses are `completed`,
`failed`, `stopped`, or `unknown`. The last blocks advancement until reconciled.
Noncompleted values are `null` and require `reason` plus `failure_kind: "measurement"`, both matching the raw record. Operation failures (including `failure_kind: "operation"`) block advancement and remain in the operation ledger; they cannot be recast as a measurement. Known measurement failures and stops
remain visible and use the frozen penalty. The penalty cannot make a failed
run better than a completed measurement. The declared denominator is the full
planned count, including failures and stops. `outcome` is `positive`, `negative`,
or `inconclusive`; a positive sample claim requires an improving observed mean.
No metric outcome turns an operation error into a scientific result.

## Recomputed tables and figures

Call `render_statistics(root, results_path, table_path, figure_path)`. The results
file points to the design. This function checks raw records and computes the
statistics, writes a TeX table, a figure, a `<figure_path>.trace.json` companion,
and `<table_path>.stats.json`. It returns those paths, `caption`, and the
computed `statistics` object. SVG needs only Python's standard library; PDF
requires Matplotlib, and absence is an explicit capability error.

A figures packet contains `design`, `results`, `statistics` (the generated
`<table_path>.stats.json`), `table`, `figure`, `trace` (the generated companion),
and `caption` (the exact returned text). Include all generated files in artifacts.
The validator recomputes statistics from raw records, compares the table with
the deterministic table renderer, verifies figure/trace hashes and the bound
statistics digest. PDF files contain that digest in metadata; SVG includes it
in its description. This binds the plotted result, without claiming a hash
proves visual readability. Final PDF page review is a separate requirement.

## Literature and inspiration evaluation

A literature packet contains `catalog`, a JSON path:

```json
{
  "question_formed_at": "2026-11-03T00:00:00Z",
  "searches": [{"query": "task AND method", "method": "keyword", "result_ids": ["S1", "S2"]}],
  "naive_baseline_ids": ["S1"], "hard_negative_ids": ["S2"],
  "candidates": [
    {"id": "S1", "title": "A source", "url": "https://example.org/source",
     "published_at": "2026-10-01T00:00:00Z", "exposure": "prospective",
     "purpose": "Compare its alternative explanation with this hypothesis",
     "decision": "include", "read_status": "read", "read_path": "literature/source.txt",
     "verification": {"status": "primary_checked", "evidence": "literature/metadata.txt"},
     "evidence_spans": [{"path": "literature/source.txt", "quote": "An exact source excerpt"}]},
    {"id": "S2", "title": "A hard negative", "url": "https://example.org/negative",
     "published_at": "2026-10-01T00:00:00Z", "exposure": "prospective",
     "purpose": "Topically relevant but incompatible statistical unit",
     "decision": "exclude", "exclusion_reason": "Cannot support the planned comparison",
     "read_status": "registered"}
  ]
}
```

Included sources require actual read material, a primary-check evidence file,
and exact resolvable spans. Metadata is an asserted source check, not a network
recheck by this validator. Sources published after question formation must be
marked `retrospective`; they cannot be represented as prospective inspiration.
Every candidate is assigned a purpose and include/exclude decision; naive
baseline and hard-negative IDs must resolve. Software fixtures label their
sources explicitly as synthetic and never claim a real journal or acceptance.

## Writing facts and fidelity

A write packet contains `manuscript` (TeX or Markdown), `assertions` (JSON), and
`bibliography` (a nonempty BibTeX file). The assertions document is:

```json
{
  "assertions": [
    {"id": "A1", "claim_id": "C1", "text": "The candidate mean error was 2.000000 m.",
     "statistics": "figures/results.tex.stats.json", "measure": "candidate.mean",
     "value": 2.0, "unit": "m", "denominator": 3,
     "condition": "all-planned-cases", "strength": "descriptive", "kind": "numeric"}
  ],
  "limitations": [
    {"kind": "scope", "text": "These measurements describe only the planned cases."}
  ],
  "prose_review": {
    "problem": "What was estimated?", "choice": "Why this comparison?",
    "finding": "Which measured result answers it?", "scope": "Where does it apply?"
  },
  "revisions": []
}
```

Each assertion's exact text and each limitation must occur in the manuscript.
`measure` is `baseline.mean`, `candidate.mean`, or `delta.mean` for legacy designs.
Named designs use `conditions.<condition-id>.mean` or
`contrasts.<contrast-id>.mean`; only frozen, recomputed entries are admitted. Values, units,
denominators, experiment claim IDs and conditions are cross-checked with raw
statistics. Only `strength: descriptive` is automatically admissible. Unqualified
universal, causal, or significant claims need a supported additional protocol;
this adapter refuses them rather than interpreting prose as proof. `kind` may
be `numeric` or `comparison`. The supported condition is `all-planned-cases`.
Failures or stopped runs require a `failure` limitation and an `adverse` limitation
when the candidate loses any paired case; fewer than five independent cases
require a `small_sample` limitation. All limitation texts appear in the manuscript.

Each substantive revision has `before`, `after`, `reason`, `facts_before`, and
`facts_after`. The facts objects describe identity, metric, data, baseline,
denominator, unit, interval and conclusion strength. They must be equal, and
the `after` text must occur in the manuscript. Equality is a structured
fidelity check; it does not replace the independent reviewer reading both prose
versions and comparing them with evidence. A revision-free initial draft uses
an empty revisions array, not fabricated editing history.

## Citation identity, claim bindings, and twenty-citation batches

The literature packet additionally includes `bibliography`, `citation_plan`, and
`citation_verification` paths. The plan is JSONL, one record per citation key:

```json
{"key":"fixture","candidate_id":"S1","title":"Synthetic source","authors":["Offline Fixture"],"year":2026,"venue":"Software fixture","identifier":"fixture:S1","url":"https://example.invalid/S1","taxonomy_cell":"case-comparison","depth":"B","status":"verified","claims":[{"id":"LC1","text":"The source uses the case as its independent unit.","evidence_spans":[{"path":"literature/source.txt","quote":"case as its independent unit"}]}]}
```

`status` is `candidate`, `read`, or `verified`, distinct from inclusion and
A/B/C/D use depth. Only a read primary source can become verified. A D-depth
candidate is dropped and cannot silently appear in the manuscript. Identity
(title, authors, year, venue, identifier) and citation key are explicit; the
BibTeX entry must agree with title/year, and each key is unique. BibTeX macros
that cannot be resolved by the bounded parser must be materialized first.
A conference `@inproceedings` upgrade additionally requires
`acceptance:{status:"accepted",authority:"official_proceedings"|"official_decision",evidence:path,version:text}`.
The official record must identify the venue and the host must match the actual
publication version. A self-reported arXiv acceptance remark is not this record.
An optional `lqs` has `dimensions` (recency, impact, venue, institution, acceptance),
`total`, and a bound `convention` path. Source weights are 0.30/0.25/0.20/0.10/0.15;
known values are 0–10. Any unknown dimension is `null` and forces `total:null`,
rather than inventing missing citation or reputation inputs.

The verification JSON is:

```json
{"records":[{"key":"fixture","status":"verified","checked_at":"2026-11-03T02:00:00Z","identity":{"title":"Synthetic source","authors":["Offline Fixture"],"year":2026,"venue":"Software fixture","identifier":"fixture:S1"},"source_sha256":"<read_path SHA-256>","evidence":["literature/metadata.txt"],"claim_checks":[{"claim_id":"LC1","status":"supported","evidence_spans":[{"path":"literature/source.txt","quote":"case as its independent unit"}]}]}],"batches":[]}
```

Identity evidence is a nonempty local source/verification record containing the
checked title, each author, year, venue, and identifier. Claims have exact
resolvable source spans. This verifies record relationships and byte identity;
the independent source reader remains responsible for semantic support and
external truth. It is not a network verification service.

For every full block of 20 non-D plan records, provide a batch record
`{keys:[the twenty keys in plan order], checked_at:ISO, evidence:path}`. Blocks
cannot skip keys or use a changed order. A final smaller block can be recorded
but is not a substitute for any missing full block. Final writing and delivery
require every non-D reference and claim to be verified/supported, including the
last block below 20. For 21 citations, the first twenty need a batch receipt;
the twenty-first still needs its individual verified identity and supported claim
records. It does not need a second aggregate batch receipt. The eight-role delivery gate rejects unresolved citation
identity or support; a historical 80% verification proportion is not enough.

## Section work and source snapshots

A write packet includes `section_packets`, a nonempty list:

```json
[{"id":"S1","path":"paper/main.tex","purpose":"Explain the supported comparison and its limits","claim_ids":["C1"],"citation_keys":["fixture"],"evidence":["figures/results.tex.stats.json"],"baseline":null,"line_limit_exception":null}]
```

Every authored TeX source reached by `\\input`/`\\include` must have a section
packet; generated accepted tables are derived source artifacts, not independent
sections. Each section's claims must occur in assertions, citations must resolve
to its actual source text, and evidence must be bound. Source paths cannot be
owned twice. New `.tex` sources above 300 lines fail. An existing large source
may declare `baseline:{path,sha256}` and
`line_limit_exception:{reason,baseline_sha256}`; the baseline is a bound, hashed
source artifact already above the limit. This explicit deviation preserves an
existing source rather than demanding an unrelated split. An initial draft has
`baseline:null`; a substantive revision also records the preserved baseline and
fidelity comparisons. The round controller owns the five-large-files workload
limit and before-round source hashes; this module owns source-size validation.

## Non-empirical survey/review path

When the approved Brief is `paper_type: review` or `survey` and explicitly marks
experiment not applicable, no synthetic numeric experiment is required. A
writing assertion may instead be:

```json
{"id":"A1","claim_id":"LC1","kind":"source_claim","strength":"source_bounded","text":"The source uses the case as its independent unit.","citation_keys":["fixture"]}
```

Source-bounded assertions are also valid background claims in empirical papers;
they never waive the empirical experiment stage. The exact claim must be present
in the verified citation plan and manuscript,
with each cited key's supported claim check. It cannot be upgraded to a causal
or universal claim merely by removing the numeric fields.

A figures packet can use `mode: evidence_diagram`, `figure`, `table`, `trace`,
`caption`, `claim_ids`, `citation_keys`, and `evidence_spans`. Its trace is
`{kind:"evidence_diagram",figure_sha256,table_sha256,source_hashes:{path:sha256},claim_ids:[...],citation_keys:[...]}`.
It requires the same verified source claims and literal evidence spans; SVG or
PDF must be a real file and table must be nonempty TeX. Diagram layout and
whether its interpretation is scientifically justified remain independent
visual/semantic review obligations. Numeric figures always use the stricter
raw-data recomputation path and cannot opt into this mode to bypass it.

## Eight-role final delivery

`validate('delivery', packet, root)` and `validate_delivery(packet, root)` both
check a final packet with `completion`, `_history`, `_brief`, `_plan` and the
full `artifacts` list. `completion` has exactly eight nonempty path arrays:
`source`, `pdf`, `bibliography`, `citation_plan`, `citation_verification`,
`review`, `revision`, and `snapshot`. Every underlying file, not only its
manifest, belongs in the artifact closure. `source` is actual `.tex`; `pdf`
starts with a PDF header; bibliography/plan/verification must be the accepted
current records; review must contain actual nonempty review evidence.

A revision role includes JSON `{status:"initial_draft"|"revised",changes:[]}`;
a revised draft supplies changes including source `baseline`, `after`, and
`reason`. An initial draft records that identity rather than fabricating a
revision. At least one snapshot role is JSON `{files:{relative_path:sha256}}`
covering every other completion-role artifact. Every entry must resolve and
match its current bytes; snapshot inventories do not replace the artifacts.
The controller still verifies independent review, actual compilation, R6,
quality gates, final binding freshness, and clean-package rebuilding.
