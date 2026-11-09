# Reusable production operations

Use the installed `paper/scripts/research.py` or repository `scripts/research.py`.
The controller owns workflow state. These operations own bounded file transforms,
recording and packet assembly. They do not choose research facts or create reviews.

JSON input flags accept absolute paths, run-relative paths from any working
directory, or an unambiguous existing cwd-relative path. Ambiguous paths fail.
Paths inside packets are always run-relative. Outputs must use new versions;
an identical artifact-tool retry may reuse existing bytes. Artifact tools cannot
write `.paper/`, `delivery/`, Git state or round-decision files.

## Select the current stage's operation

| Stage | Default operation | Host-owned judgment |
| --- | --- | --- |
| Intake and venue | `tool source-fetch`, `tool environment` | Source authority, suitability, resources and missing metadata |
| Venue choice | `venue-compare`, `ask-user --kind venue`, `confirm-venue --response` | Ask the real user, wait and retain the actual venue choice |
| Venue | `venue-calibrate --publish`, including its declared `reading_index` | Actual per-visual reading and source corrections |
| Plan budget and increases | `ask-user --kind budget`, `confirm-budget --response` | Show defaults, current usage and the concrete proposal; wait for the actual answer |
| Plan and experiment | `tool experiment-template` and `lib.experiments.ExperimentRecorder` | Scientific operation, scoring, exposure and protocol |
| Figures | `render_statistics`, `export_image` WorkOrder actions | Figure choice, units, interpretation and venue policy |
| Write | `prepare_tex`, `compile_tex` WorkOrder actions; `tool pdf-inspect` | Source policy, argument, captions and actual PDF viewing |
| Review | `bind-all`, `prepare-review`, `assemble-review` | Real independent findings, gates and each approved item |
| Revision | `rewind --preserve`, `tool revision-record`, new WorkOrders | Which claims need correction and why |
| Delivery | `prepare-round`, actual decision, `deliver` | The actual round judgment; submission is separate |

`tool --name NAME --packet INPUT.json [--out REPORT.json]` works on a current
initialized run. `artifact` has the same packet interface for the state-free
production tools below. Use `artifact` for copying/replaying historical inputs
without initializing, resetting or upgrading their old controller state. It
cannot approve a stage. `artifact review-request` consumes a saved `context`;
the normal current-run review uses `prepare-review` instead.

The question commands prepare packets, not human answers. Follow
[user-input.md](user-input.md) for the actual host question and source-bound
reply. Before an explicitly registered research resource call, reserve its
verifiable maximum with `budget-reserve`; afterwards preserve and settle the
actual receipt with `budget-settle`, including failed/unknown/overrun use.
[budget.md](budget.md) specifies this single task budget and its accounting scope.

## Acquire and inspect inputs

`source-fetch`: `sources:[{url,path,kind,record?,sha256?,max_bytes?}]`.
Kinds are `pdf`, `xml`, `json`, `zip`, `text` and `html`; URLs must be HTTPS.
The default provenance path is `path + '.source.json'`. Reuse verifies URL,
kind and content hash. A failed later source retains earlier verified files.
For changed sources, use new paths. A PDF header or successful download does
not prove relevance, acceptance or a complete reading. Incomplete APIs need an
explicit official alternative and documented parsing; do not guess a target
column or invent metadata.

`environment`: `{python?:EXECUTABLE,dependencies:[DISTRIBUTION,...]}`.
It returns versions and `available` or `missing_dependencies`. Resolve missing
dependencies before freezing an executable plan; the tool does not install them.

`reading-index`: `{samples:[{id,pdf,design?,analysis?,visual_read?,title?}],target}`.
Supply the actual per-visual records, either inline `design` or an `analysis`
JSON path. Counts and links derive from those records, with no fixed paper or
panel count. Each panel renders its question, claim, evidence, design takeaway
and limitations. Supplied locators, PDF page links and panel-measurement
corrections remain visible; a panel count alone is not a reading record. Links are relative so the index survives moving the run.
The normal venue packet declares `reading_index` in `artifacts`; calibration
`--publish` writes it and venue acceptance recomputes it. The existing 8–15
calibration policy still applies to actual venue selection.

`pdf-inspect`: `{source,directory,dpi?:120,pages?:[ONE_BASED_PAGE,...]}`.
Poppler renders all pages by default, with annotations visible. The output
report binds source bytes and rendered PNGs. Installed PyMuPDF also supplies
font spans and link rectangles/borders; absence is reported as unavailable.
The host must actually view the PNGs. Counts, text extraction, font flags or
the report cannot certify a visual reading. Inspect emphasis against numbers
and source typography before flagging an apparent ranking error. A changed
PDF requires an appropriately scoped new review.

## Record experiments without losing failures

Create the actual design first. Run `tool --name experiment-template` with
`{design,target}` to generate the stable recording envelope. Fill only the
task's `measure(case_id,seed,condition)` function before freezing the evaluator.
Use the generated import path, or set `AUTORESEARCH_RUNTIME` when moving it.
For a Python adapter, `lib.client.invoke` calls the public CLI without duplicating
path resolution or transport logging.

```python
with ExperimentRecorder(run, design=design_path) as records:
    for case_id, seed, condition in records.grid:
        with records.attempt(case_id, seed, condition) as complete:
            value, evidence = measure(case_id, seed, condition)
            complete(value, evidence)
    records.finish(outcome="inconclusive")
```

The grid and denominator derive from the frozen design. An exception records
the attempted operational failure and unstarted slots separately, then
propagates the error. Such records block scientific statistics. Do not translate
an installation failure into zero performance. Prior raw records cannot be
overwritten; use versioned result/raw paths for an authorized rerun.
An explicit `MeasurementFailure(reason, status="failed"|"stopped")` is reserved
for failures/stops defined by the frozen measurement protocol; those stay in
the scientific denominator with its frozen penalty. Ordinary exceptions never
receive that classification automatically.
The SDK supports the same paired-case numerical adapter as `science-packets.md`;
it does not make that statistical protocol universal.

`statistics`: `{input:RESULTS,table:PATH,figure:PATH}` is available through
both `tool` and state-free `artifact`; it uses the same validated paired-case
renderer as the WorkOrder action. This produces evidence, not stage acceptance.
The current numerical adapter supports exactly baseline/candidate paired cases.
Preflight the actual full condition matrix before freezing a research plan;
a three-arm or larger study needs a compatible validated adapter, not removal
or relabeling of required arms to fit the recorder.

## Prepare figures, text and bibliography

Registered WorkOrder actions are:

```json
{"type":"export_image","source":"figures/source.tiff","target":"figures/journal/figure.tif","options":{"mode":"RGB","dpi":300,"min_dpi":300}}
{"type":"prepare_tex","source":"manuscript/authored.tex","target":"manuscript/reading.tex","options":{"link_borders":"hidden"}}
```

These options are caller-selected examples, not journal defaults. Image options
are `mode:preserve|RGB|L`, `dpi` (number or pair), `min_dpi`, `width_pixels`,
`height_pixels`, `max_bytes`, and optional `background`. The exporter preserves
pixel dimensions, normalizes actual DPI metadata and verifies encoded pixels.
Nonopaque alpha cannot be discarded without an explicit background. Fonts are
not inferred from raster images.

`prepare_tex` options are `graphics:keep|omit`, `inline_inputs:boolean`,
`link_borders:template|hidden`, `flush_before:[LITERAL_MARKER,...]`, and
`captions_after_first_reference:boolean`. Caption relocation requires omitted
graphics and explicit paragraph breaks. Use an official-rule-backed variant
policy; no journal is assumed to forbid figures. Literal source transforms
refuse unsupported dynamic/verbatim constructs. Compilation and actual viewing
remain required. Included source hashes are recorded for interrupted-effect
reconciliation. Pure `image-export` and `tex-prepare` tools accept the same
`source`, `target`, `options` fields without `type`.

`bibliography`: `{target,entries:[{key,kind?,title,year,authors?,jats?,...}]}`.
Supply canonical author strings or `{family,given}` objects. A publisher JATS
path provides explicit author components without guessing compound surnames.
Public fields include `journal`, `booktitle`, `howpublished`, `doi`, `url`,
`volume`, `number` and `pages`. Plain metadata strings are escaped as text;
prewritten TeX is not interpreted. Internal data belongs in `metadata` and is
not copied into BibTeX. Source verification and claim
support are separate. Citation checking understands explicit BibTeX comma
ordering, so a plan and verification may retain the same verified source-name
wording without rewriting it merely to match bibliography formatting.
Year-only catalog dates may use
`published_at:"2011"`; overlapping question dates cannot be called prospective.

## Assemble actual reviews and delivery

After write acceptance, give `bind-all --packet bindings.json --operation-id ID`
an array of bindings. Requirements, WorkOrders, assets and literal insertion
locations come from the actual task. Valid earlier suboperations survive a
failed later item and can be retried with the same unchanged inputs.

`prepare-review --out review/request.json` verifies current evidence and
publishes the bounded response contract. The independent reviewer returns its
actual identity, request digest, reports, closed findings, G1–G5/R6 judgments,
one semantic check per approved ID, and one R6 judgment per original visual.
The reviewer returns each subject's `binding_sha256`; the program copies the
full authoritative hash map. Do not manufacture review execution.

`assemble-review --packet assembly.json` consumes:

```json
{"request":"review/request.json","judgment":"review/judgment.json","target":"review/packet.json","completion":{"revision":["review/revision.json"],"snapshot":"review/snapshot.json","r6_report":"review/r6-mechanical.json"}}
```

The assembler derives current source closure, PDF, bibliography, citation plan,
citation verification and declared submission variant from accepted stages.
Optional completion fields are `extra_source`, `extra_pdf`, `review`, and
`extra_artifacts`. It computes original-plan checks and the complete version
snapshot. Missing items, changed evidence, open findings or unresolved gates
stop assembly. It assembles judgments; only `commit` accepts the review stage.

Before modifying accepted evidence, use `rewind --preserve PATH ...` to copy
the affected accepted baseline. `revision-record` consumes
`{target,changes:[{before,after,reason}]}` and calculates baseline hashes. An
initial manuscript instead declares `{target,initial_draft:true,changes:[]}`.
Scientific corrections still need actual source comparison and independent
review; a revision record does not judge their validity.

After committing review, call `prepare-round --evidence PATH ...`. The program
saves its exact request in `control/round-decisions/`; transport logs go outside
the run. Read the request, then write only the actual judgment in that reserved
directory. Use `deliver --decision PATH --operation-id ID`: it applies the
issued checkpoint, finalizes, rebuilds from exported assets and verifies the
ZIP's complete file set, CRC and bytes. Raw `apply-round` and `finalize` remain
available. An exported runtime includes the upstream protocols it needs.
