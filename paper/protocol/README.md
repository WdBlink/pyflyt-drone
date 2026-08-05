# Frozen paper protocols

These files are the machine-readable execution authority for the
privileged-to-perceptual fixed-wing study:

- [`method-matrix.json`](method-matrix.json) freezes method identities, matched
  budgets, recurrent/export semantics, five independent mechanisms,
  leave-one-out cells, targeted interactions, selectors, and config-diff
  allowlists.
- [`evaluation-protocol.json`](evaluation-protocol.json) freezes the synchronized
  onboard RGB-D/proprioception claim, actor/evaluator separation, causal target
  association and corruption, training-only curriculum, paired confirmation,
  claim vocabulary, evidence lifecycle, and timing gate.
- [`loop-guard.json`](loop-guard.json) freezes known OPC units, protected reviews,
  exact fix closure, and experiment/final evidence requirements.
- [`literature-matrix.md`](literature-matrix.md) is a conservative derivation of
  the bounded primary-source search; it does not establish novelty or SOTA.

Validate the contracts before implementation or heartbeat completion:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/pyflyt/bin/python scripts/validate_paper_protocol.py contracts
/opt/homebrew/Caskroom/miniforge/base/envs/pyflyt/bin/python scripts/validate_paper_protocol.py loop-complete --session <session> --unit <id> --artifacts <paths...>
```

Null freeze hashes and an unsealed episode count are deliberate pre-pilot
boundaries. Comparative execution cannot start until their typed receipts are
created, content-addressed, and independently reviewed.
