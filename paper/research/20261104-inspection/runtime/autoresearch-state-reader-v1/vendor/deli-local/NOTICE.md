# Local Deli adaptation provenance

These three scripts and their tests are user-authored local additions described
in the installed `deli-autoresearch/PROVENANCE.md`. They are copied byte-for-byte;
the source paths and SHA-256 values are recorded in `manifest.json`.

They are not copies of Deli's third-party protocol pages. No upstream Deli
license is inferred and no Deli HTML or upstream original text is included here.
The integration imports measurement, profile, plan-audit and item-validation
functions only. The original CLIs are retained for regression evidence, and
are not invoked to create a second project state or approval ledger.

`python3 run-tests.py` runs the unchanged helper tests and explicitly skips the
four `CheckpointGateTests` that import the old `checkpoint.py`. That script was
not copied: the new runtime has one controller ledger instead. The new runtime
checks its own evidence binding, completion and recovery. Running bare discovery
inside this vendor directory is not its declared test command because those four
historical integration tests require the intentionally absent old ledger.
