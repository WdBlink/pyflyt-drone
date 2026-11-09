# Research workflow packet and command reference

Paths are relative to a private run root. Python 3.10+, Poppler and TeX are
required for the complete paper path; PDF statistics also require Matplotlib.
No models, credentials, server, queue or remote executor are installed.

The executable example is `paper/tests/run_fixture.py`. It creates synthetic
materials, rules, sources and outcomes explicitly labelled SOFTWARE FIXTURE.
Its schema is an example, not a research design or a default scientific budget.
`science-packets.md` specifies the supported statistical and prose checks.

## Brief and approval

`init --brief brief.json` accepts an object with `question`, `scope`, `angle`, `audience`, `venue`, `direction`,
`paper_type`, `resources`, `reviewers` and nonempty `deliverable_plan`. Each
item has a stable letter-first `id`, `kind` (`figure`, `table`, `section` or
`manuscript`), `title`, `purpose`, `claim`, `acceptance_criteria` and
`evidence_requirements`. Array entries describe actual obligations. Resources
state the real hardware, access, cost basis and actual authorization. The single
optional top-level `budget` contains the disclosed task limits; its default is
`{}` with no implicit total cap. See [budget.md](budget.md). Unknowns
remain unknown. The fixture uses `software_fixture: true`; real research does
not. Reviewers name the actual independent actors expected to return reviews.

`approve --brief-sha256 HASH --actor HUMAN` relays the user's approval of this
exact displayed Brief. It is not a replacement for real human approval, a
cryptographic attestation or a legacy Harness signed receipt. State cannot be
initialized again to reset historical work. Changing obligations requires an
explicitly approved revised Brief in a new run, retaining the old run as evidence.

When the actual user instruction delegates preparation and execution, do not
invent a later human approval of an unseen Brief. `approve` can retain that
actual authority with `--authorization authority.json`. This does not answer
the mandatory venue and plan-budget questions in [user-input.md](user-input.md).
The authorization record is `{mode:'delegated',path,sha256,quote,scope}`:
an actual saved instruction, its exact hash/quote and the concrete delegated
scope. Name the host as the decision actor. The controller records `delegated
user instruction`, binds the selected Brief and rechecks the source on use.
This validates record integrity; the host must verify the actual authorization
and cannot expand it to paid resources, private data or publication.

## Shared packet fields and boundaries

For current runs, use the mechanical routes in [production.md](production.md).
The public CLI resolves unambiguous JSON input paths against the run from any
cwd and records transport output outside research documents. Per-stage
`production_steps` supplies the default route; the raw fields below remain the
wire contract for integration and recovery.

Every stage packet has `stage`, `producer`, and a nonempty unique `artifacts`
array. All referenced paths are current run-relative files and must be declared
in that array (domain science checks may also use earlier accepted artifacts).
No symlinks, traversal, NaN, duplicate JSON keys or controller-owned paths are
accepted. Fields beginning with `_` are controller-injected and forbidden in
submitted packets. The controller binds exact bytes and pins its code and
stage knowledge for the run. A changed implementation cannot silently continue
an old run.

`context` returns knowledge text, not just links. Read it, perform the stage
with authorized host tools, then `commit --packet packet.json --operation-id ID`.
Reuse an operation ID only for identical arguments. Stages are ordered; a
recorded packet is a technical handoff, not a scientific pass or a waiver.

## Intake

`materials` lists `{id,path,sha256,availability,read_spans}`. Availability is
`readable` or `metadata_only`; the latter requires `limitation` and cannot
claim content spans. `read_spans` is a list of two-integer character intervals,
for example `[[0,100],[100,250]]` for a 250-character file. Readable spans cover
the whole text, with inclusive starts and exclusive ends. Actually use
`read-material --path FILE --start N --end N` to receive exact character spans
and read receipts in the same ledger. Each call returns at most 100000
characters; chunk longer files. The intake packet requires a nonempty top-level
`evidence` array of quote records, separate from each option's evidence. They are
`{material_id,start,end,quote}` and must resolve exactly to read content.
This proves bytes were returned, not that a model understood them.

`feasibility` contains `status`, `selected_option`, `options`, `resources`, and
`experiments`. Compare at least two options or supply `single_option_reason`.
Each option has `id`, `implementation`, `selection_reason` and span `evidence`.
Resources state `authorization` and `cost_basis`. Each experiment states
`claim`, `metric`, `comparison`, `protocol`, `completion`, and `failure_route`.
A conditional/blocked route cannot silently become an executable plan.
`status: conditional` requires a nonempty `prerequisites` array; `blocked`
requires nonempty `blocking_questions`. State the actual unresolved conditions.
Review/survey scope may explicitly declare an experiment inapplicable in the
approved Brief; this does not admit unimplemented statistics or remove other
required artifacts.

## Venue, template and plan

The full comparison, default **8–15** matched accepted samples, recent-two-edition
or journal-year policy, exact per-figure/table/panel records and official page
budget schema are in [venue-packets.md](venue-packets.md). They are validated,
not satisfied by mentioning a local source file. A small software fixture has an
explicit sampling exception; real studies disclose any count/window deviation.

Use the main controller's deterministic tools, without importing helpers or
hand-reimplementing their arithmetic:

```sh
python3 scripts/research.py --run RUN tool --name venue-compare --packet venue.json
python3 scripts/research.py --run RUN tool --name venue-design-template --packet venue.json --out venue/design-template.json
python3 scripts/research.py --run RUN tool --name venue-calibrate --packet venue.json --publish
python3 scripts/research.py --run RUN tool --name plan-snapshot --packet plan.json --out plan/snapshot.json
python3 scripts/research.py --run RUN tool --name plan-audit --packet review.json --out review/plan-audit.json
python3 scripts/research.py --run RUN tool --name citation-check --packet literature.json
```

Design templates prefill only observed numbers, captions, pages and panels.
The host actually reads the rendered samples and fills question/claim/evidence,
design takeaway and limitations. Blank semantic fields never count as an analysis.
Calibration publishes measurements, venue comparison/profile and plan-check JSON
to the packet's declared output fields, refusing a conflicting existing version.

A target already explicitly chosen is reused with the actual answer source.
For undecided venues, prepare a
`contract_status: draft` Brief with `venue: undecided`, `deliverable_plan: []`.
Intake and venue preparation need no fictitious final approval. Compare actual
sourced candidates, issue `ask-user --kind venue --packet venue.json`, present
the question to the user and wait for the real selection. Relay that answer with
`confirm-venue --packet venue.json --actor HUMAN --comparison-sha256 HASH
--response venue-answer.json --operation-id ID`. The hash comes from
`venue-compare`; delegated authorization is not a venue answer.
After accepting calibration, `set-contract --brief concrete.json --operation-id ID`
returns the final visible Brief and digest. It preserves scope/resources and
history. Relay the actual exact approval using `approve` before plan/experiments.

The plan packet contains `figure_table_plan`, `plan_snapshot` and `work_orders`.
Before plan acceptance, issue `ask-user --kind budget --packet proposal.json`,
show the uncapped default, optional caps and the concrete resource proposal,
then wait for the actual reply and `confirm-budget`. Default selection also
requires an answer. Every budget increase follows the same route. Source-bound
answer records and their exact fields are in [user-input.md](user-input.md).
For empirical scope it also contains `experiment_design_sha256`; create the
design and evaluator before plan acceptance. Approved non-empirical synthesis
omits that empirical adapter and retains its explicit experiment N/A reason.
`planning.plan_snapshot(path)` captures original plan
bytes and items. The frozen promise set cannot be retroactively replaced by a
plan matching the finished draft.

A WorkOrder is `{id,producer,requirement_ids,actions}`. Every approved ID must
be covered. Registered actions have exact fields:

- `{type:'validate_packet',packet:PATH}`.
- `{type:'render_statistics',input:RESULTS,table:PATH,figure:PATH}`.
- `{type:'compile_tex',source:TEX,pdf:PATH}`.
- `{type:'copy_artifact',source:PATH,target:PATH}`.
- `{type:'export_image',source:PATH,target:PATH,options:OBJECT}`.
- `{type:'prepare_tex',source:PATH,target:PATH,options:OBJECT}`.

Production options and capabilities are specified in `production.md`. They
come from the current venue/task, with no fixed dimensions, font, paper count
or revision filename. Included TeX dependencies enter effect input hashes.

`execute --work-id ID --operation-id ID` performs these bounded local calls.
Existing outputs cannot be overwritten. No generated JS, Python or shell is
executed. The host may run authorized research outside this executor and submit
its actual evidence. A completed WorkOrder is not item fulfillment.

For same-contract revisions, rewind the affected stage and use
`revise-orders --packet new-orders.json --reason TEXT --operation-id ID` to
append new versioned IDs and output paths. Original orders and scientific
promises remain frozen. `supersedes` may name a previous order, preserving its
failure/attempt history. Duplicate IDs, unknown obligations and unregistered
actions are refused.

## Literature, experiment, figures and write

Use `science-packets.md`. The default numerical adapter supports paired
independent cases, within-case seed means and mean/SD/SE or exploratory normal
95% CI. Unsupported data/units/statistics fail explicitly; do not force real
research into the software fixture. Figure knowledge covers broader design
choices, while automated plotting is limited to the admitted adapter.

The write packet uses one source: `manuscript == manuscript_tex`, plus
`manuscript_pdf`, `assertions`, and `bibliography`. Bind `template_source` and
`template_sha256` to the selected official template. The adapter checks literal
TeX documentclass/options and any explicitly required packages. When an official
template is unavailable, the template check requires an exact rule quote and
explicit format decision; it cannot silently substitute article format.
Independent reviewers still check official attribution and semantic fidelity.

## Binding, independent review and finalization

The default entry uses `bind-all` with the actual binding array, then
`prepare-review --out PATH`. Its request includes current gate guidance,
allowed verdicts and an unjudged response template containing the approved IDs
and subjects. The independent host fills judgments rather than copying hash
maps. `assemble-review --packet PATH` derives the eight completion roles from
accepted stages, recomputes R6 and records the full snapshot. Actual report and
revision paths remain explicit inputs; no initial-draft or pass is invented.
`commit` still performs final review-stage acceptance.

After write acceptance, submit `bind` packets with `requirement_id`, `work_id`,
`assets`, `evidence`, `manuscript`, and `location`. Location is a literal marker
in the accepted source. Figures and tables must be literally inserted via
includegraphics/input/include in that source and cannot share a primary asset
with a different promised visual. All bytes must already be validated.
The returned `binding_sha256` is the review's exact subject.

A review packet names `reviewer`, uses the exact accepted `manuscript_pdf`, and
contains `requirement_checks` for every approved ID, exactly once. Each check
has `requirement_id`, `binding_sha256`, `evidence_hashes` (the binding's entire
path→hash map), `verdict:'fulfilled'`, and a substantive `rationale`.
`gates` contains G1–G5 and R6 with `verdict:'pass'` and declared artifact
`evidence`. G2 inapplicability must exactly match explicit approved review
scope. A missing/unsupported requirement cannot be waived by a score or a
limitation. The actual host must supply independent review; declaring a name
in JSON does not prove that execution occurred.

R6 also requires `figure_table_plan`, the original `plan_snapshot`, and
`r6_reviews:[{id,verdict,reason,evidence_locations}]` for every original visual.
A missing/changed promise cannot be called fulfilled. Deviation judgments must
include a `disclosure_quote` actually in the PDF and cannot weaken the approved
item contract. G1–G5 semantic review remains distinct from R6 plan mechanics.

`finalize --operation-id ID` revalidates all stages, bindings, effects and exact
reviews, rebuilds TeX from the exported asset copy, checks reproduced PDF text,
and writes a manifest, editable source, PDF, producing software/evidence and ZIP.
`verify-export --manifest delivery/ID/manifest.json` detects changed bytes.
This is internal artifact verification, not an external acceptance or submission.
After a real final checkpoint judgment, `deliver --decision PATH --operation-id
ID` performs checkpoint application and finalization together. Export verification
also checks ZIP file membership, CRC and byte hashes; the producing runtime
contains its required upstream protocols.

## Rounds and recovery

Call `begin-round` **before** work, including source changes. It captures the
original plan and supported text engineering assets. It is required for post-plan
artifact commits and typed actions. Rewind captures a before-revision baseline.
An explicit `--session-id` names an actual host session. It cannot reset the task
budget, usage or creation time; the controller cannot prove host identity.
Thirty minutes since a checkpoint is a continuation/review prompt, not a task
cap. Large-file workload, stagnant rounds and refinement scores are advisory.
They cannot require a stop, split a valid manuscript or bypass scientific gates.
The one public `budget` is disclosed and answered during planning; known usage
and held reservations derive from actual resource receipts. See
[budget.md](budget.md) for scope and explicit deadline/closeout behavior.
`last_seen_at` and `last_progress_at` remain distinct. A revised finding has a new
ID and `revises` link; it does not inflate the count of distinct findings.


`prepare-round --evidence PATH ... [--current-plan PATH]` returns the before-round snapshot, actual plan diff, workload and a state/rules/evidence-bound request. Repeated `--evidence` flags accumulate their paths too. Zero evidence is allowed only for a genuine no-progress judgment.
`apply-round --request JSON --decision JSON --operation-id ID` accepts a host
judgment with `request_sha256`, `progress` (`verified_progress`, `no_progress`,
`regression`), `reason`, `status` (`active`, `needs_input`, `stopped`), optional
`finding` and `direction`. Waiting/stopped judgments name `blockers`.
A finding has `claim`, `validation`, and `limitations`; a structural pivot has
`id`, `change`, and `reason`. Stagnation advises a different direction while
available authorized work continues. `needs_input` requires concrete external
blockers with affected `requirement_ids`, covering all obligations and no known
executable local WorkOrders for a global wait. `stopped` additionally requires
an actual source-bound user stop or program-verified explicit budget exhaustion.
Internal implementation failures remain active work. Unchanged evidence is not new progress; duplicate findings
cannot inflate it. This judgment validates structure/binding, not scientific
truth. A round cannot grant final completion.

Missing terminal action receipts are uncertain and block replay. `reconcile`
requires an evidence JSON naming `attempt`, `order_sha256`, `pending_action`,
`outcome`, `reason`, and `artifact_hashes`. Completed dispositions must account
for all actions and expected output bytes; unrelated evidence is rejected.
Inputs must still match their captured version. `not_started` is refused when
any action/output exists. Failed work retains its effects and requires a known
successful retry or explicitly superseding work before delivery. Reconciliation
is an actual evidence review, not a timeout or permission to retry blindly.

`rewind --stage STAGE --reason TEXT --operation-id ID` keeps invalidated
receipts in history and clears downstream binding/review. It does not reset
attempts or change the Brief. Old-format runs cannot be resumed here.

## Fresh contexts and Git exploration

`fork-trigger --config JSON` evaluates upstream schedule/early/plateau limits.
`fork-plan --config JSON` creates files-only branch contexts and distinct Advisor
assignment slots. `fork-create --repo CLEAN_REPO --plan JSON --destination DIR`
creates actual detached worktrees at the best pinned commit. Keep the paper run
outside this checkout. The host launches the actual fresh-context agents.
`fork-select --plan JSON --candidates JSON --incumbent-metric NUMBER` verifies
branch observation bindings and chooses the working-metric winner. Supplying
`--promote-repo CLEAN_REPO` permits fast-forward promotion only when source HEAD
still equals the incumbent, every diff path is in Scope and no controller-owned
state is touched. Protected incumbent tags are per commit so later rounds work.
These operations do not grant scientific acceptance or external publication.


Final completion requires actual source, PDF, bibliography, citation plan,
citation verification, independent review, revision identity and a version
snapshot covering the actual files, as specified in science-packets.md.
`draft-preview --pdf manuscript/main.pdf --out previews/v1.pdf` produces a
watermarked non-delivered copy for sharing incomplete work. It never changes the
accepted source/PDF or turns a partial plan into fulfillment.
If the ledger is lost, init refuses to reset historical/export records.
`recover [--manifest delivery/ID/manifest.json]` restores only the latest durable controller checkpoint with pinned code and integrity checks. Optional exports are verified but never selected as an older recovery authority. Missing latest recovery evidence is a real blocker, not permission to reset counters.


## Complete prose guidance and judgment files

At `write` and `review`, `context` loads the complete user-local academic-prose
principles and case index. `context --writing-case E07 --writing-case E06`
returns the complete selected examples (at most two) with their source hashes;
these are teaching examples, not facts for the current paper. They are copied
with provenance, not reduced to an index pointing at an external installation.

Store the final host round decision JSON under `control/round-decisions/` (or
outside the run root). That reserved judgment-input channel is excluded from
workload snapshots and cannot be accepted as a research artifact. Actual R6
reports are created before `prepare-round`, included in its evidence list, and
bind `round_id`, `original_plan_sha256`, `current_plan_sha256` and `verdict`.
Public hashes alone do not authorize a request: apply compares the exact latest
controller-issued preparation and recomputes current document hashes. A heartbeat
updates last_seen without invalidating unchanged research evidence.

The one controller writes a latest durable checkpoint before the live control
projection. A mismatch blocks work. `recover` uses only that latest checkpoint,
retaining post-export rounds, operations and pending effects; an older export
can be verified but never used to rewind history. No separate research engine
or independently maintained ledger is introduced. This recovery copy has one writer and the same business state as the live projection.
