"""Prespecified descriptive parent-cluster summaries; no new hypotheses."""
from __future__ import annotations

import hashlib
import math
from statistics import mean

import numpy as np

from eval.inspection_recording import CONDITION_REPEATS
from eval.inspection_statistics import SEEDS

FIELDS = ("C", "S", "D", "Q", "H", "E", "Y", "SR")
ARMS = {
    "ablation": ("periodic", "no-cross-aircraft-memory", "no-progressive-confirmation-guidance"),
    "latency": ("delay-00", "delay-02", "delay-05", "delay-10", "delay-20"),
}
SALTS = {"clean": 101, "ablation": 102, "latency": 103}
DRAWS = 99999


def intervention_methods(track):
    if track == "clean":
        return tuple(f"{side}:{condition}" for side in ("event", "clean") for condition in CONDITION_REPEATS)
    if track in ARMS:
        return ("primary", *ARMS[track])
    raise ValueError("only the three prespecified intervention tracks are analyzed")


def intervention_contrasts(track):
    if track == "clean":
        return tuple(("event-minus-clean:" + condition, "event:" + condition, "clean:" + condition)
                     for condition in CONDITION_REPEATS)
    if track == "ablation":
        return tuple((arm + "-minus-primary", arm, "primary") for arm in ARMS[track])
    if track == "latency":
        return tuple((arm + "-minus-fresh-zero", arm, "delay-00") for arm in ARMS[track][1:]) + (
            ("fresh-zero-minus-primary-stability", "delay-00", "primary"),)
    raise ValueError("only the three prespecified intervention tracks are analyzed")


def analyze_interventions(track, episodes):
    """One fixed root draw carries every arm, component and original/common score."""
    if np.__version__ != "1.26.4":
        raise ValueError("the frozen descriptive bootstrap requires NumPy1.26.4")
    methods = intervention_methods(track)
    scopes = ("original", "common_calendar") if track == "clean" else ("original",)
    roots = tuple(f"main-{domain}-L3-{i:02d}" for domain in ("ID", "OOD") for i in range(6))
    repeats = {method: CONDITION_REPEATS[method.split(":", 1)[1]] if track == "clean" else 3
               for method in methods}
    expected = {(root, method, repeat) for root in roots for method in methods for repeat in range(repeats[method])}
    values = {}
    for row in episodes:
        identity = row["case_id"], row["method"], row["repeat_ordinal"]
        if (identity not in expected or identity in values or type(row["repeat_ordinal"]) is not int
                or row.get("status") != "completed" or row.get("domain") != identity[0].split("-")[1]):
            raise ValueError("complete unique original parent x method x repeat records required")
        cells = {}
        for scope in scopes:
            score = row["score" if scope == "original" else "common_score"]
            if (set(score) != set(FIELDS) or any(type(score[key]) not in (int, float)
                    or not math.isfinite(score[key]) or not 0 <= score[key] <= 1 for key in FIELDS)
                    or any(type(score[key]) is not int or score[key] not in (0, 1) for key in ("H", "E", "SR"))):
                raise ValueError("eight finite admitted L3 components required; no missing-flight imputation")
            cells[scope] = score
        values[identity] = cells
    if set(values) != expected:
        raise ValueError("all planned intervention and primary comparator episodes are required")
    columns = [(scope, method, field) for scope in scopes for method in methods for field in FIELDS]
    positions = {column: i for i, column in enumerate(columns)}
    root_means = {root: {scope: {method: {field: mean(values[root, method, repeat][scope][field]
        for repeat in range(repeats[method])) for field in FIELDS} for method in methods} for scope in scopes}
        for root in roots}
    matrices = {domain: np.array([[root_means[root][scope][method][field] for scope, method, field in columns]
        for root in roots if root.split("-")[1] == domain], dtype=np.float64) for domain in ("ID", "OOD")}
    bootstraps, streams = {}, {}
    for domain, matrix in matrices.items():
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEEDS[domain], 3, SALTS[track]])))
        indices = rng.integers(0, 6, size=(DRAWS, 6), dtype=np.int64, endpoint=False)
        streams[domain] = {"seed_sequence": [SEEDS[domain], 3, SALTS[track]], "bit_generator": "PCG64",
            "root_order": [root for root in roots if root.split("-")[1] == domain],
            "shape": [DRAWS, 6], "dtype": "little-endian int64",
            "index_bytes_sha256": hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()}
        bootstrap = np.empty((DRAWS, len(columns)), dtype=np.float64)
        for start in range(0, DRAWS, 256):
            end = min(start + 256, DRAWS)
            bootstrap[start:end] = matrix[indices[start:end]].mean(axis=1)
        bootstraps[domain] = bootstrap

    def summarize(point, bootstrap):
        output = {}
        for scope in scopes:
            per_method = {}
            for method in methods:
                per_method[method] = {field: {"mean": float(point[positions[scope, method, field]]),
                    "ci95_root_cluster_percentile_nominal": np.quantile(
                        bootstrap[:, positions[scope, method, field]], [.025, .975], method="linear").tolist()}
                    for field in FIELDS}
            contrasts = {}
            if scope != "original" or track != "clean":
                for name, left, right in intervention_contrasts(track):
                    contrasts[name] = {"minuend": left, "subtrahend": right, "components": {}}
                    for field in FIELDS:
                        a, b = positions[scope, left, field], positions[scope, right, field]
                        contrasts[name]["components"][field] = {"effect": float(point[a] - point[b]),
                            "ci95_root_cluster_percentile_nominal": np.quantile(
                                bootstrap[:, a] - bootstrap[:, b], [.025, .975], method="linear").tolist()}
            output[scope] = {"methods": per_method, "contrasts": contrasts}
        return output

    domains = {domain: {"independent_parent_roots": 6,
        "scopes": summarize(matrix.mean(axis=0), bootstraps[domain])} for domain, matrix in matrices.items()}
    combined = summarize((matrices["ID"].mean(axis=0) + matrices["OOD"].mean(axis=0)) / 2,
                         (bootstraps["ID"] + bootstraps["OOD"]) / 2)
    return {"role": "prespecified_secondary_descriptive_no_new_confirmatory_tests", "track": track,
        "independent_parent_roots": 12, "new_independent_roots": 0,
        "additional_actual_episode_denominator": {"clean": 216, "ablation": 108, "latency": 180}[track],
        "referenced_primary_episode_denominator": 216 if track == "clean" else 36,
        "domains": domains, "combined_equal_domain_scopes": combined, "root_means": root_means,
        "bootstrap": {"draws": DRAWS, "numpy": np.__version__, "quantile": "linear", "streams": streams,
            "sampling_unit": "one original parent vector; repeats averaged first, all arms and scopes carried together",
            "interval_scope": "nominal descriptive percentile only; six parents per domain, no finite-sample guarantee"},
        "p_values": "not_computed", "confirmatory_family": "unchanged_main_global6",
        "independence": "independent computational streams do not establish independence of real root outputs"}
