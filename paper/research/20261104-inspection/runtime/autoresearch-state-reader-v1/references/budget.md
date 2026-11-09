# Task budget and checkpoints

The only public resource-limit field is the Brief's `budget`. Its default is
`{}`: no implicit task cap. This default grants no new paid-service, compute or
publication permission. The concrete plan and actual user resource authorization
still govern execution. The plan-budget question must receive an actual answer,
including when the user chooses the default. See [user-input.md](user-input.md).

```json
{"budget":{"calls":1200,"tokens":5000000,"cost":{"amount":"20.00","currency":"USD"},"minutes":240}}
```

Every field is optional. `calls` and `tokens` are nonnegative integer caps for
explicitly registered research resource operations. Zero permits no use of that
meter. `cost` has a nonnegative plain decimal string and a three-letter uppercase
currency. The controller does not price services or convert currencies.
`minutes` is a positive, explicit task wall-time cap from run creation, including
preparation, waiting and every host session. Omission imposes no time cap.

`round_limits`, `resources.limits` and `resources.budget` are rejected for new
runs. Resources describe hardware, access and authorization; they are not another
limit configuration. The approved effective budget lives in the controller's
Brief. Editing an input file does not amend that frozen state.

## Confirm or change a budget

Prepare the local scientific and resource plan before committing the plan stage.
Use `ask-user --kind budget --packet proposal.json`. A proposal is
`{budget,plan,reason,estimate?}`. `plan` is the actual run-relative figure/table
plan JSON; `estimate` optionally names a readable resource explanation. The
question shows defaults, current limits, known usage, reservations, proposed
limits, units and the versioned plan. It must be asked through the real host
user-input tool. Apply the source-bound answer with `confirm-budget --response
answer.json --operation-id ID`.

The user may retain the uncapped default, set caps or decline the proposal.
Declining keeps the existing budget. A new plan version or changed contract needs
a current question. Every increase, removal of a cap or changed currency follows
this same explicit question/answer route. A recommended or preselected choice,
broad delegation, silence and elapsed waiting time are not answers.

## Record explicit research resources

The controller has one `usage_ledger` in `.paper/`, with normal operation IDs,
locking and recovery. Known consumption and still-held reservations are derived
from its entries, not rewritten by the model. Session counters are diagnostics
and cannot clear usage. Native host turns, subagents, local CLI calls and their
tokens/fees are not inferred or estimated by this ledger.

Before a host is about to issue a research resource operation, call:

```text
budget-reserve --packet reservation.json --operation-id REQUEST_ID
```

The packet is `{usage:{calls?,tokens?,cost?},purpose}`. Supply a verifiable
maximum for every capped resource. Admission sums known usage and in-flight
reservations. It requires an approved, user-confirmed frozen plan and a recorded
work-round baseline. A reservation does not launch a request or grant service
access. Unknown maximum usage cannot support a hard cap.

After the actual operation, preserve its original receipt and call:

```text
budget-settle --reservation REQUEST_ID --packet actual.json --operation-id ID
```

The packet is `{receipt:PATH,status:'known'|'unknown'|'not_started',usage:{...}}`.
Failed requests still include their actual resource use. The recorded values
must come from the actual receipt, not a hypothetical response. The program
checks source integrity and arithmetic; the host remains responsible for the
truthful translation of external usage.

Actual overruns are recorded even if they exceed the reservation or cap. They
block further resource admission, not accounting. Without `--reservation`, the
command records already-occurred unreserved use for audit; it authorizes nothing.
Identical operation IDs replay without double charging; changed inputs refuse.
One preserved receipt path cannot be charged to two ledger entries.

Missing capped meters or a mismatched currency keep an entry `unknown`.
Known partial values remain visible; its remaining reservation stays held.
Another resource request waits for reconciliation. Independent local work can
continue. Later usage evidence uses a new receipt version and a new operation ID
on the same unresolved reservation. `not_started` releases a reservation only
with actual no-dispatch evidence and no consumption. Missing uncapped meters
remain explicitly unreported, rather than becoming measured zero values.
If a newly added cap needs a meter missing from historical receipts or in-flight
reservations, admission waits for that actual usage to be reconciled. A later receipt may fill missing
meters; it cannot rewrite previously reported quantities or erase actual use.

## Checkpoint and exit

Thirty minutes since the last applied checkpoint produces a `checkpoint.due`
hint. It does not stop work. Document size, refinement scores and stale-round
counts likewise request review or a different approach. They are not budgets or
scientific acceptance. The host saves the checkpoint and continues available
authorized work; the package does not start a background scheduler.

`active` records progress or a checkpoint. Internal implementation work stays
active. `needs_input` requires structured external blockers with `kind`, `reason`
and affected `requirement_ids`. Allowed kinds are `input`, `authorization`,
`external_service` and `budget`. Partial blockers cannot put all other work into
a global wait, and executable local WorkOrders cannot be hidden by a blocker.
These checks constrain declarations; they do not prove external facts or human
intent.

`stopped` additionally requires `stop.reason` of `user_request` with a
source-bound `human_response`, or `budget_exhausted` verified by the program.
A checkpoint, unsupported adapter, low review gain or missing implementation is
not a terminal cause. A resource-only cap blocks its requests; global budget
termination also needs all remaining work blocked. An explicit total wall-time
cap can stop new work directly, including local preparation before plan acceptance.
`resume-task --response SOURCE_JSON
--operation-id ID` records a real user resumption without resetting usage or task
creation time. Budget amendments require their own current user answer.

Checkpoint application, usage reconciliation and finalization remain available
for closeout after a time cap. Financial usage uncertainty is disclosed and its
reservation retained. It does not replace the existing refusal of unresolved
scientific effects or failed evidence gates. Final delivery still requires every
original stage, requirement, actual independent review and rebuild/ZIP check.

## Historical runs

New controller state is format 2. Older run state and its implementation pin stay
unchanged. Use the old runtime to read or resume those records; the new CLI does
not silently rewrite their budget or approval history. An explicitly reviewed
migration or a new run must preserve the actual old evidence and resource use.
