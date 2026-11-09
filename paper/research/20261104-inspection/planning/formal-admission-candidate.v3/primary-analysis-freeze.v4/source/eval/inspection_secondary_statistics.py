"""Descriptive SR intervals with repeats nested inside independent roots."""

from __future__ import annotations

from statistics import mean

import numpy as np

from eval.inspection_recording import CONDITION_REPEATS
from eval.inspection_statistics import CONDITIONS, MAIN_SHAPE, RootTable, SEEDS


def sr_root_table(metadata, episodes, *, require_main_shape=True):
    """All planned rows must be completed; validation of flight receipts is upstream."""
    roots = [r["case_id"] for r in metadata]
    expected = {(root, condition, repeat) for root in roots
                for condition, count in CONDITION_REPEATS.items() for repeat in range(count)}
    scores = {}
    for row in episodes:
        identity = row["case_id"], row["condition"], row["repeat_ordinal"]
        if (identity not in expected or identity in scores or type(row["repeat_ordinal"]) is not int
                or row.get("status") != "completed" or type(row.get("SR")) is not int or row["SR"] not in (0, 1)):
            raise ValueError("unique completed planned binary SR records required; no operational-failure imputation")
        scores[identity] = row["SR"]
    if set(scores) != expected:
        raise ValueError("full planned episode denominator is required")
    means = {root: {c: mean(scores[root, c, repeat] for repeat in range(count))
                    for c, count in CONDITION_REPEATS.items()} for root in roots}
    return RootTable.parse(metadata, means, require_main_shape=require_main_shape)


def sr_cluster_intervals(table, *, draws=99999, batch_size=256):
    if type(draws) is not int or draws < 2 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive explicit bootstrap settings required")
    domains = {}
    for domain in sorted({g for g, _ in table.groups}):
        levels = sorted({level for g, level in table.groups if g == domain})
        if levels != sorted(MAIN_SHAPE[domain]):
            raise ValueError("complete level strata required within each domain")
        cells = [table.values[[i for i, g in enumerate(table.groups) if g == (domain, level)]] for level in levels]
        if any(len(cell) < 2 for cell in cells):
            raise ValueError("at least two independent roots per stratum required")
        rngs = [np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEEDS[domain], level, 1]))) for level in levels]
        boot = np.zeros((draws, len(CONDITIONS)))
        for start in range(0, draws, batch_size):
            end = min(draws, start+batch_size)
            for values, rng in zip(cells, rngs):
                indices = rng.integers(0, len(values), size=(end-start, len(values)))
                # One root draw carries all conditions and their repeat means.
                boot[start:end] += values[indices].mean(axis=1)/len(cells)
        macro = sum(cell.mean(axis=0) for cell in cells)/len(cells)
        domains[domain] = {"independent_roots": sum(map(len, cells)),
            "level_macro_root_mean_SR": dict(zip(CONDITIONS, map(float, macro))),
            "ci95_root_cluster_percentile_nominal": {c: np.quantile(boot[:, i], [.025, .975], method="linear").tolist()
                                                      for i, c in enumerate(CONDITIONS)},
            "stratum_counts": dict(zip(map(str, levels), map(len, cells)))}
    return {"role": "secondary_descriptive_SR_not_new_confirmatory_tests", "domains": domains,
            "draws": draws, "sampling_unit": "independent root; repetitions first averaged within root",
            "interval_scope": "nominal percentile bootstrap, not finite-sample guaranteed coverage; no flattened-repeat Wilson",
            "receipt_validation": "required upstream; unique metadata alone does not prove root independence"}
