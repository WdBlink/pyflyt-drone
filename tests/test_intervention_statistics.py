"""Literal scalar fixtures, not generated mission roots or experimental data."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "eval/inspection_intervention_statistics.py"
spec = importlib.util.spec_from_file_location("intervention_statistics_candidate", path)
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def episodes(track):
    rows = []
    for domain in ("ID", "OOD"):
        for i in range(6):
            for j, method in enumerate(stats.intervention_methods(track)):
                count = stats.CONDITION_REPEATS[method.split(":")[1]] if track == "clean" else 3
                for repeat in range(count):
                    y = .2 + i * .01 + j * .005 + repeat * .002 + (domain == "OOD") * .1
                    score = {key: y for key in stats.FIELDS}
                    score.update(H=1, E=1, SR=0)
                    row = {"case_id": f"main-{domain}-L3-{i:02d}", "domain": domain,
                        "method": method, "repeat_ordinal": repeat, "status": "completed", "score": score}
                    if track == "clean":
                        row["common_score"] = {**score, "Y": y + .1, "Q": y + .1}
                    rows.append(row)
    return rows


@pytest.mark.parametrize("track", ["clean", "ablation", "latency"])
def test_fixed_twelve_parent_repeat_means_and_shared_streams(track):
    rows = episodes(track)
    result = stats.analyze_interventions(track, rows)
    assert result["new_independent_roots"] == 0 and result["p_values"] == "not_computed"
    assert result["bootstrap"]["draws"] == 99999
    assert result["additional_actual_episode_denominator"] + result["referenced_primary_episode_denominator"] == len(rows)
    methods = stats.intervention_methods(track)
    method = next((name for name in methods if name.startswith("event:B4-M1")), methods[0])
    root = result["root_means"]["main-ID-L3-00"]["original"][method]
    assert root["Y"] == pytest.approx(np.mean([r["score"]["Y"] for r in rows
        if r["case_id"] == "main-ID-L3-00" and r["method"] == method]))
    scope = "common_calendar" if track == "clean" else "original"
    for domain in ("ID", "OOD"):
        stream = result["bootstrap"]["streams"][domain]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([stats.SEEDS[domain], 3, stats.SALTS[track]])))
        indices = rng.integers(0, 6, size=(99999, 6), dtype=np.int64, endpoint=False)
        assert stream["index_bytes_sha256"] == hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()
        assert stream["root_order"] == [f"main-{domain}-L3-{i:02d}" for i in range(6)]
        for contrast in result["domains"][domain]["scopes"][scope]["contrasts"].values():
            summary = contrast["components"]["Y"]
            # Every method has the same root slope: paired draws cancel it.
            assert summary["ci95_root_cluster_percentile_nominal"] == pytest.approx([summary["effect"]] * 2)
    id_value = result["domains"]["ID"]["scopes"][scope]["methods"][method]["Y"]["mean"]
    ood_value = result["domains"]["OOD"]["scopes"][scope]["methods"][method]["Y"]["mean"]
    assert result["combined_equal_domain_scopes"][scope]["methods"][method]["Y"]["mean"] == pytest.approx((id_value + ood_value) / 2)
    if track == "latency":
        contrasts = result["domains"]["ID"]["scopes"][scope]["contrasts"]
        assert contrasts["delay-20-minus-fresh-zero"]["subtrahend"] == "delay-00"
        assert contrasts["fresh-zero-minus-primary-stability"]["subtrahend"] == "primary"
    if track == "clean":
        assert result["domains"]["ID"]["scopes"]["original"]["contrasts"] == {}


@pytest.mark.parametrize("fault", ["missing", "duplicate", "bool_repeat", "wrong_parent", "nan", "missing_D", "binary", "unprojected"])
def test_incomplete_or_invalid_intervention_vectors_rejected(fault):
    rows = episodes("ablation")
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "bool_repeat":
        rows[0]["repeat_ordinal"] = False
    elif fault == "wrong_parent":
        rows[0]["case_id"] = "main-ID-L3-06"
    elif fault == "nan":
        rows[0]["score"]["Y"] = float("nan")
    elif fault == "missing_D":
        rows[0]["score"]["D"] = None
    elif fault == "binary":
        rows[0]["score"]["H"] = .5
    else:
        rows[0]["score"]["dynamic_entities"] = []
    with pytest.raises(ValueError):
        stats.analyze_interventions("ablation", rows)
