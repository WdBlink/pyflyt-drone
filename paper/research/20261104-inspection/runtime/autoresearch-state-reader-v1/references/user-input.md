# User-answer checkpoints

Venue selection, the initial concrete plan budget and every budget increase are
**请用户回答** steps. `ask-user` prepares a question; it does not ask the human
or supply an answer. The host presents the returned question through its actual
user-input interface (Claude Code `AskUserQuestion`; Codex's available user-input
tool), then waits for a real reply. An async question remains pending until that
reply arrives. Continue only independent, already authorized preparation while
waiting. Silence, elapsed time, a recommended/preselected option and general
execution delegation cannot complete these steps.

If the user has already explicitly answered the same concrete choice, reuse that
actual instruction with its exact source and quote. Do not ask the same question
again merely to create a new record. A prior broad request to finish the research
does not answer which venue or whether to accept an undisclosed budget.

## Venue choice

After intake, build the sourced comparison described in `venue-packets.md`.
Run `tool --name venue-compare --packet venue.json`, then:

```text
ask-user --kind venue --packet venue.json
```

Show the candidate journals/conferences, year/track, official requirements,
research fit, relevant costs/constraints, unknowns and recommendation. Ask which
candidate the user chooses; a recommendation is not approval. The question binds
the actual source hashes and candidate IDs. A different suggested venue needs
its own source comparison before acceptance. After the answer, set the venue
packet's selected ID and venue to that choice, preserving the compared candidates,
and recompute `venue-compare` for the final packet. Apply:

```text
confirm-venue --packet venue.json --actor HUMAN --comparison-sha256 HASH --response venue-answer.json --operation-id ID
```

Calibrate the selected venue using the normal 8–15 matched accepted papers and
actual reading of every measured figure/table/panel. A count or generated
reading template does not complete that reading. Existing explicit target choices
still use their actual response source; `--authorization` cannot replace it.

## Plan and budget choice

Prepare the concrete scientific plan and resource explanation before accepting
the plan stage. Read `budget.md`. The proposal JSON is:

```json
{"budget":{},"plan":"planning/figure-table-plan.json","reason":"Resource needs of this concrete plan","estimate":"planning/resource-estimate.md"}
```

`estimate` is optional; when available it should explain the planned operations,
their verifiable maxima, price basis and unknowns. Never invent a price or imply
that a native host's session tokens are metered by this controller. Run:

```text
ask-user --kind budget --packet budget-proposal.json
```

Present these facts with the question:

- Default `budget: {}` sets no implicit task total cap and grants no new paid or
  compute authorization. Thirty minutes is a checkpoint prompt, not a hidden cap.
- Optional limits are research `calls`, `tokens`, `cost` with amount/currency and
  total wall-clock `minutes`. Time starts at run creation and includes waiting
  and all host sessions; it is not a fresh session allowance.
- Show the concrete plan, current limits, known consumption, held reservations,
  unknown/unreported usage, proposed limits and reason for any change.
- Ask the user to retain the default, give custom limits or decline this proposal.
  Declining retains the existing budget and requires a revised feasible plan.

Apply `confirm-budget --response budget-answer.json --operation-id ID` only
after the reply. No answer means plan acceptance and new research resource
reservations remain blocked. A new plan version or changed contract needs a
current answer. Every increase, removal of a cap or currency change uses the same
question and actual reply, with accumulated consumption visible. Do not overwrite
the Brief or reset a session to make additional resources appear approved.

## Answer records

Save the actual host/user transcript text under a versioned run-relative path;
preserve it unchanged. An answer JSON has exactly these fields:

```json
{"question_sha256":"HASH_FROM_ASK_USER","answer":{"candidate_id":"ID_FROM_COMPARISON"},"human_response":{"path":"inputs/user-choice.txt","sha256":"EXACT_SOURCE_HASH","quote":"Exact actual user reply"}}
```

For budget, `answer` is `{"choice":"default"}`, or
`{"choice":"custom","budget":{"calls":20}}`, or `{"choice":"decline"}`.
Only include units the user actually chose. An uncapped default also requires
an actual answer. The controller verifies source bytes, the quote, current
question/plan/ledger hashes and permitted choices; it cannot prove who authored
text or interpret its intent. The host must truthfully translate the actual
reply. Never write a simulated reply to advance a real research run. Software
fixtures label their scripted answers explicitly and prove only software behavior.

`context` and `status` expose unanswered questions. The controller preserves
answers and field amendments in its one state/history and includes the actual
response sources in delivery. These focused answers do not waive the existing
final Brief approval, scientific obligations or independent review gates.
