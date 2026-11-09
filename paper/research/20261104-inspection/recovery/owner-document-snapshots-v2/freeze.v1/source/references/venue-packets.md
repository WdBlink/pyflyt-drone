# Venue packets and scriptable preparation

This contract implements the user's local Deli venue-calibration method and the
additional request to compare journals before selecting a target. Deli's source
method starts with a chosen venue; it does not claim an automatic journal selector.
The new comparison prepares evidence for a human choice and never invents rankings,
submission costs, deadlines or editorial acceptance.

## Inputs before a venue is selected

A draft Brief can use `venue: "undecided"`. Intake can read actual project material
and formulate candidate claims without manufacturing a target. Scientific scope,
resources and research obligations remain under the controller's Brief contract.
The host obtains actual candidate author guidelines, scope pages and other needed
sources. Source snapshots must be local, in the packet's `artifacts`, with their
HTTPS URL and SHA-256. Every candidate quote must exist in its actual source text.

`venue_selection` has these fields:

- `mode` is `existing_target` or `candidate_comparison`.
- `selected_id` names one stable candidate ID; `rationale` explains the proposed
  choice and `criteria` lists the actual comparison dimensions.
- `candidates` has at least two entries for an undecided comparison. A target
  already chosen by the user can use one `existing_target` entry.
- Each candidate has `id`, `venue`, `kind` (`journal` or `conference`), `scope_fit`,
  `article_type_fit`, `constraints`, `advantages`, `limitations`, and `evidence`.
  `evidence` entries have `path`, `url`, `sha256`, and an exact `quote`.
- `cost` and `timing` each contain `status: known|unknown` and `basis`. Unknown
  costs or schedules remain unknown. A populated string is not proof that its
  factual interpretation is correct; the host/reviewer must read the sources.

`venue_tools.comparison_report(packet, root)` validates this structure and produces
an evidence-bearing report with every candidate, the comparison criteria, the
proposed candidate and its rationale. `comparison_digest(report)` computes its
stable canonical identity. The controller records the actual human selection,
then injects `_venue_confirmation` containing `candidate_id`, `venue`, and that
`comparison_sha256`. A model-written `confirmed` field has no authority. An
existing user-selected Brief target requires no repeated selection gate.

## Accepted-paper sampling

The default is **8–15 accepted/ published papers in the same concrete direction
and comparable article type**. For conferences identify the two most recent
completed editions in `sampling_policy.recent_editions`; for journals use the two
years before the explicit ISO `as_of` date. A journal `publication_date` gives an
exact day. Year-only metadata at the two-year boundary cannot establish recency;
retain such a paper only with an explicit deviation reason.

`sampling_policy` contains:

- `kind: journal|conference` and `as_of: YYYY-MM-DD`.
- `recent_editions: [year, year]` for conferences. These labels are sourced host
  declarations, not a program's discovery of which editions are newest.
- `sample_count_deviation_reason` for fewer than 8 or more than 15 papers.
- `time_window_deviation_reason` for older, boundary-uncertain or future samples.
- `measurement_warning_review` describing the actual warnings, their inspection
  and remaining measurement limits when the parser reports warnings.

Deviation reasons must be substantive text (at least 20 characters). Describe the
search scope, available accepted studies, exclusion reasons or additional cost
benefit. A reason does not establish its own correctness. Do not fill the quota
with unrelated articles. Fewer than 8 papers makes the count budget a reference
range: the script downgrades quantity-band failures to warnings. Required fields,
artifact integrity, design coverage and promised scientific outputs remain enforced.

A software fixture must explicitly set `software_fixture: true` and provide
`fixture_exception_reason`. Only the sampling count/time-window exception uses
this marker. A one-paper software fixture does not demonstrate that a real venue
has been surveyed; it still needs the complete source, selection, visual and page
accounting structures.

Every sample retains its unique ID, real PDF, venue label, year, acceptance-evidence
snapshot and direction-match explanation. Duplicate PDF bytes cannot count as
multiple samples. Journals and conferences do not share one profile. The same
article's preprint and accepted edition cannot be silently counted twice.

## Measure first, then inspect every visual

`venue_tools.design_template(packet, root)` reads each real PDF using the migrated
`venue_profile` helper. It pre-fills the measurable visual ID, number, caption,
page and detected panel labels. It deliberately leaves semantic fields empty.
The returned `papers[SAMPLE_ID]` contains `visual_read`, `figures`, and `tables`.
Copy the first into `sample.visual_read` and the last two into `sample.design`.
Complete them only after actual reading of rendered PDF pages.

Every measured figure and table must have one structured row:

```json
{
  "id": "S01-F1",
  "number": 1,
  "page": 2,
  "caption": "exact caption returned by measurement",
  "type": "method_overview",
  "placement": "body",
  "question": "What question does this visual answer?",
  "claim": "The conditional claim it supports, rather than its topic.",
  "evidence": "The actual data, model or comparison that supports it.",
  "design_takeaway": "The useful visual argument or encoding to consider.",
  "limitations": "The conditions under which this design does not transfer.",
  "panels": []
}
```

Each detected panel needs its own `label`, `question`, `claim`, `evidence`,
`design_takeaway` and `limitations`. `visual_read` records the actual `pages`,
`method: rendered_pdf`, and a `notes` explanation. All measured caption numbers
must be covered exactly once; stable IDs must be unique and captions must match
measured bytes. Merely matching the number of `type/placement` entries is insufficient.
The validator checks this correspondence, not whether a model understood an image.

Panel labels and caption page numbers come from text heuristics. If actual rendered
inspection corrects missed or wrongly grouped panels, record
`panel_measurement_note` on that visual; never silently present parser output as
visual truth. The profile preserves the measurement, the reading declaration and
all completed design notes for the planner/writer/reviewer.

## Keep page rules separate from sample habits

`page_budget` includes `target` (positive integer or `null`), `limit_type`
(`hard_limit|target|unlimited|unknown`), `basis`
(`official_requirement|sample_inference|unknown`), explicit
`includes_references` and `includes_appendix`, and a `reason`.

For an official rule, `source` binds the selected author-rules snapshot by
`path/url/sha256/quote`. The exact quote must exist. Do not substitute an adjacent
year or a sample median for that rule. If no usable page rule is known, use unknown
basis/limit and `target: null`; reference and appendix inclusion can remain null.

An inferred target names `sample_ids` and cannot be a hard limit. Mixed
`pages_basis` cannot form a single inferred target. Total-page inference includes
references and appendix; references-heading inference must disclose shared-page
and appendix measurement limits in `measurement_limitations`. Inference is not a
proof of compliance with official exclusions.

The profile's `page_accounting.selected_rule` preserves this declared source rule,
while `page_accounting.sample_statistics` separately preserves `pages_total`,
`pages_main_body`, and `pages_basis` for every measured PDF. Poppler's PDF total is
an actual page count. The References-heading count is a heuristic inclusive of
its shared page and cannot certify exact charged pages under every venue policy.

## Generate and validate the handoff

`planning.calibrate(packet, root)` runs the actual PDF measurements, selection
comparison, sampling policy, complete per-visual checks, source-bound page rule,
profile aggregation, and figure/table-plan check. It returns four artifacts:

1. `measurements` with actual PDF hashes and caption/panel records.
2. `venue_profile`, including full design notes, sampling policy and page accounting.
3. `venue_plan_check`, including warnings rather than silently hiding deviations.
4. `venue_comparison`, the stable source-bearing candidate report.

The packet names a run-relative JSON output path for every key and lists all four
in `artifacts`. The host or typed helper saves these outputs. Venue validation
recomputes them from the source PDFs and compares content (excluding generation
timestamps). It also checks the controller's selection receipt and the frozen
scientific direction/paper type. The downstream plan uses the same profile; writing
keeps the chosen template bytes and literal class/options; R6 reads the frozen plan,
current plan and actual PDF.

All helper calls are deterministic and local. They do not search the internet,
call a model, approve a target, start experiments, submit a paper, or provide an
editorial acceptance prediction. Source collection, visual/semantic assessment and
the actual human choice remain explicit host responsibilities.
