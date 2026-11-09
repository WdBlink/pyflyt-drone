"""Independent CPU-only audit and bounded-iid null diagnostics; never flight data."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[6]
RESEARCH = REPO / "paper/research/20261104-inspection"
sys.path.insert(0, str(REPO))
from eval.inspection_recording import CONDITION_REPEATS, episode_schedule
from eval.inspection_statistics import (
    CONDITIONS, MAIN_SHAPE, RootTable, analyze_roots,
    macro_difference_and_se, studentized_reference,
)

SOURCE_PATHS = (
    "eval/inspection_statistics.py", "eval/inspection_recording.py",
    "eval/preflight_inspection_statistics.py", "eval/preflight_inspection_recording.py",
    "tests/test_inspection_statistics.py",
)
NAMES = ("architecture-M1", "architecture-M2", "classic-M1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return dict(count=int(k), denominator=int(n), rate=p,
                mc95_wilson=[max(0., center-radius), min(1., center+radius)])


def metadata():
    return [dict(case_id=f"review-synthetic-{g}-L{l}-{i:02d}",
                 lineage_group=f"review-synthetic-{g}-L{l}-{i:02d}", domain=g, level=l)
            for g, levels in MAIN_SHAPE.items() for l, n in levels.items() for i in range(n)]


def manual_quantile(values, q):
    a = sorted(values)
    pos = (len(a)-1)*q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    return a[lo] + (pos-lo)*(a[hi]-a[lo])


def manual_holm(p):
    out = {}; previous = 0.
    for i, key in enumerate(sorted(p, key=lambda k: (p[k], k))):
        previous = max(previous, (len(p)-i)*p[key])
        out[key] = min(1., previous)
    return out


def independent_scalar_reference(table, draws=97):
    """Recompute each bootstrap from scalar root differences, fsum, and stdev.

    Does not call production estimator/studentization/quantiles/Holm. Uses the
    declared RNG only to reproduce the same index experiment.
    """
    result = {}; p = {}
    pairs = (("B4-M1", "B3-M1"), ("B4-M2", "B3-M2"), ("B4-M1", "B1"))
    for domain, seed in (("ID", 2026110401), ("OOD", 2026110402)):
        levels = sorted({l for g, l in table.groups if g == domain})
        cells = [table.values[[i for i, g in enumerate(table.groups) if g == (domain, l)]] for l in levels]
        streams = [np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, l]))) for l in levels]
        indices = [rng.integers(0, len(a), size=(draws, len(a))) for a, rng in zip(cells, streams)]
        k = len(cells)
        for name, (a, b) in zip(NAMES, pairs):
            x = [[float(row[CONDITIONS.index(a)] - row[CONDITIONS.index(b)]) for row in cell] for cell in cells]
            means = [math.fsum(s)/len(s) for s in x]
            effect = math.fsum(means)/k
            se = math.sqrt(math.fsum(statistics.variance(s)/(k*k*len(s)) for s in x))
            star = []; exceed = 0; degenerate = 0
            for draw in range(draws):
                sampled = [[s[int(i)] for i in idx[draw]] for s, idx in zip(x, indices)]
                delta = math.fsum(math.fsum(s)/len(s) for s in sampled)/k
                sd = math.sqrt(math.fsum(statistics.variance(s)/(k*k*len(s)) for s in sampled))
                star.append(delta)
                if sd <= 1e-12:
                    degenerate += 1; exceed += 1
                elif abs((delta-effect)/sd) >= abs(effect/se):
                    exceed += 1
            raw_p = 1. if se <= 1e-12 or degenerate/draws > .01 else (1+exceed)/(draws+1)
            key = domain + "." + name
            p[key] = raw_p
            result[key] = dict(effect=effect, root_se=se, p_raw=raw_p,
                               ci95_pointwise=[manual_quantile(star, .025), manual_quantile(star, .975)],
                               exceedances=exceed, degenerate_draws=degenerate)
    adjusted = manual_holm(p)
    for key, value in result.items(): value["p_holm_global6"] = adjusted[key]
    return result


def audit_existing():
    record = {}
    for part, expected in (("full-shape", 1296), ("holdout-shape", 216)):
        root = RESEARCH / "qualification/adapter-preflight.v5" / part
        order = json.loads((root/"episode-order.json").read_text())
        schedule_sha = digest(root/"episode-order.json")
        reproduced = episode_schedule(list(reversed(order["cases"])), order["seed"])
        # The reviewer runtime is separately frozen; version metadata must not
        # be confused with an actual schedule difference.
        assert {k:v for k,v in reproduced.items() if k != "numpy_version"} == {k:v for k,v in order.items() if k != "numpy_version"}
        receipts = [json.loads(p.read_text()) for p in sorted((root/"order-receipts").glob("*.json"))]
        assert len(receipts) == len(order["slots"]) == expected
        previous = None
        for slot, receipt in zip(order["slots"], receipts):
            assert all(receipt[k] == v for k, v in slot.items())
            assert receipt["schedule_sha256"] == schedule_sha
            case, condition, repeat = slot["case_id"], slot["condition"], slot["repeat_ordinal"]
            start = json.loads((root/f"starts/{case}-{condition}-{repeat}.json").read_text())
            raw = json.loads((root/receipt["raw"]).read_text())
            assert start["order_index"] == slot["order_index"] == raw["evidence"]["order_index"]
            assert start["schedule_sha256"] == raw["evidence"]["schedule_sha256"] == schedule_sha
            assert (raw["case_id"], raw["seed"], raw["condition"]) == (case, repeat, condition)
            assert raw["status"] == "completed" and raw["attempted"] is True
            at = datetime.fromisoformat(start["started_at"])
            assert previous is None or at >= previous
            previous = datetime.fromisoformat(raw["finished_at"])
        blocks = [order["slots"][i:i+18] for i in range(0, expected, 18)]
        for block in blocks:
            assert len({s["case_id"] for s in block}) == 1
            assert Counter(s["condition"] for s in block) == Counter(CONDITION_REPEATS)
        completion = json.loads((root/"recording-completion-index.json").read_text())
        assert completion["episode_denominator"] == expected and completion["schedule_sha256"] == schedule_sha
        for group, h in completion["results_sha256"].items(): assert digest(root/group/"results.json") == h
        outer = json.loads((root/"root-means/results.json").read_text())
        for row in outer["runs"]:
            raw = json.loads((root/row["raw"]).read_text()); evidence = raw["evidence"]
            values = []
            for binding in evidence["inner_attempts"]:
                assert digest(root/binding["path"]) == binding["sha256"]
                values.append(json.loads((root/binding["path"]).read_text())["value"])
            assert len(values) == CONDITION_REPEATS[row["condition"]]
            assert row["value"] == raw["value"] == statistics.mean(values)
            assert evidence["schedule_sha256"] == schedule_sha
            assert digest(root/evidence["inner_results"]) == evidence["inner_results_sha256"]
        record[part] = dict(episode_slots=expected, root_blocks=len(blocks), outer_rows=len(outer["runs"]),
                            receipt_start_raw_identity_and_time_order=True, frozen_schedule_reproduced=True,
                            original_numpy=order["numpy_version"], reviewer_numpy=np.__version__,
                            aggregation_bindings_valid=True, first_root_conditions=[s["condition"] for s in blocks[0]])
    return record


def hand_and_invariance_checks():
    delta, se = macro_difference_and_se([np.array([[0.], [.2]]), np.array([[.4], [.6], [.8], [.2]])])
    assert math.isclose(float(delta[0]), .3) and math.isclose(float(se[0]), math.sqrt(1/150))
    tie = studentized_reference([.25], [.25], np.array([[-.25], [.25], [.75]]), np.array([[.5], [.25], [.5]]))[0]
    assert tie["exceedances"] == 2 and tie["p_raw"] == .75
    boundary = {}
    for count in (1, 2):
        ss = np.ones((100, 1)); ss[:count] = 0
        value = studentized_reference([.3], [.1], np.full((100, 1), .3), ss)[0]
        assert value["degenerate_draws"] == count and value["exceedances"] == count
        assert value["p_raw"] == (2/101 if count == 1 else 1.)
        boundary[str(count)] = value
    for value in (0., 1e-13, 1e-12):
        actual = studentized_reference([.2], [value], np.full((100,1), .2), np.ones((100,1)))[0]
        assert actual["p_raw"] == 1. and actual["T_obs"] is None
    meta = metadata(); rng = np.random.Generator(np.random.PCG64(470125))
    means = {r["case_id"]: {c: float(.5 + rng.uniform(-.2, .2)) for c in CONDITIONS} for r in meta}
    table = RootTable.parse(meta, means)
    production = analyze_roots(table, "B1", draws=97, batch_size=13)
    reference = independent_scalar_reference(table, draws=97)
    max_abs = 0.
    for domain in ("ID", "OOD"):
        for name in NAMES:
            a = production["domains"][domain]["contrasts"][name]; b = reference[domain+"."+name]
            for key in ("effect", "root_se", "p_raw", "p_holm_global6"):
                max_abs = max(max_abs, abs(a[key]-b[key]))
                assert math.isclose(a[key], b[key], abs_tol=2e-13), (domain, name, key, a[key], b[key])
            assert np.allclose(a["ci95_pointwise"], b["ci95_pointwise"], atol=2e-13, rtol=0)
            assert a["exceedances"] == b["exceedances"]
    reverse = RootTable.parse(list(reversed(meta)), {k:dict(reversed(list(v.items()))) for k,v in reversed(list(means.items()))})
    assert analyze_roots(reverse, "B1", draws=97, batch_size=31) == production
    identical = {r["case_id"]: {c: float(.2 + .5*rng.random()) for c in CONDITIONS} for r in meta}
    for row in identical.values(): row["B4-M1"] = row["B3-M1"]
    paired = analyze_roots(RootTable.parse(meta, identical), "B1", draws=97)
    for domain in ("ID", "OOD"):
        x = paired["domains"][domain]["contrasts"]["architecture-M1"]
        assert x["root_se"] == 0 and x["ci95_pointwise"] == [0.,0.] and x["p_raw"] == 1.
    return dict(hand_macro_effect=.3, hand_macro_se=math.sqrt(1/150), tie_tail=tie,
                degenerate_draw_boundary=boundary, observed_se_epsilon_boundary_conservative=True,
                independent_97draws_max_scalar_error=max_abs,
                input_and_batch_invariance=True, identical_paired_methods_have_zero_difference_interval=True,
                independent_reference=reference)


def generate_null(scenario, rng, meta):
    means = {}; diagnostics = dict(rare_root_counts={"ID":0, "OOD":0})
    for row in meta:
        g = row["domain"]
        if scenario in ("symmetric_uniform_null", "shared_zero_gate_null"):
            base = float(rng.uniform(.35,.65))
            gate = 1 if scenario == "symmetric_uniform_null" else int(rng.random() < .65)
            score = {c:gate*base for c in CONDITIONS}
            for c, repeats in CONDITION_REPEATS.items():
                if repeats == 3:
                    score[c] = float(gate*(base + rng.uniform(-.2,.2,3).mean()))
        elif scenario == "skew_episode_null":
            score = {c:.95 for c in CONDITIONS}
            for c in ("B4-M1", "B4-M2"):
                score[c] = float((rng.random(3)<.95).mean())
        elif scenario == "rare_root_loss_null":
            # For each B4: .02*0 + .98*(.95 + .95*.02/.98) = .95.
            # Small independent zero-mean repeat noise on the common branch
            # keeps observed SE positive even when no rare loss was seen.
            score = {c:.95 for c in CONDITIONS}
            for c in ("B4-M1", "B4-M2"):
                rare = rng.random() < .02
                diagnostics["rare_root_counts"][g] += int(rare)
                score[c] = 0. if rare else float(.95+.95*.02/.98+rng.uniform(-.002,.002,3).mean())
        else:
            raise ValueError(scenario)
        assert all(0 <= x <= 1 for x in score.values())
        means[row["case_id"]] = score
    return means, diagnostics


def stress(replications, draws):
    scenarios = ("symmetric_uniform_null", "skew_episode_null", "rare_root_loss_null", "shared_zero_gate_null")
    meta = metadata(); summaries = {}; records = []
    for si, scenario in enumerate(scenarios):
        counters = {f"{g}.{n}": Counter() for g in ("ID","OOD") for n in NAMES}
        families = Counter(); fallback = Counter()
        for replication in range(replications):
            data_seed = [20261105, 8101, si, replication]
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(data_seed)))
            means, diag = generate_null(scenario, rng, meta)
            result = analyze_roots(RootTable.parse(meta, means), "B1", draws=draws, batch_size=128)
            item = dict(scenario=scenario, replication=replication, data_seed=data_seed, diagnostics=diag, tests={})
            rejected = []; raw_rejected = []; bound_p = {}
            for g in ("ID", "OOD"):
                for n in NAMES:
                    key = f"{g}.{n}"; c = result["domains"][g]["contrasts"][n]
                    covered = c["ci95_pointwise"][0] <= 0 <= c["ci95_pointwise"][1]
                    counters[key]["raw05"] += c["p_raw"] <= .05
                    counters[key]["holm05"] += c["reject_nominal_alpha05"]
                    counters[key]["coverage"] += covered
                    counters[key][c["status"]] += 1
                    if c["reject_nominal_alpha05"]: rejected.append(key)
                    if c["p_raw"] <= .05: raw_rejected.append(key)
                    bound_p[key] = c["hoeffding_p_robustness_only"]
                    item["tests"][key] = {k:c[k] for k in ("effect","root_se","p_raw","p_holm_global6","ci95_pointwise","status","exceedances","degenerate_draws")}
            adjusted_bound = manual_holm(bound_p)
            families["global_holm6_false_rejection"] += bool(rejected)
            families["any_unadjusted05_false_rejection"] += bool(raw_rejected)
            families["hoeffding_global_holm6_false_rejection"] += any(v <= .05 for v in adjusted_bound.values())
            item["global_holm6_false_rejection"] = bool(rejected)
            records.append(item)
        summaries[scenario] = dict(
            replications=replications, bootstrap_draws=draws, true_macro_effects_all_zero=True,
            family_metrics={k:wilson(v,replications) for k,v in families.items()},
            per_hypothesis={key:{"raw_false_positive05":wilson(c["raw05"],replications),
                                  "holm6_false_positive05":wilson(c["holm05"],replications),
                                  "pointwise_ci95_coverage":wilson(c["coverage"],replications),
                                  "status_counts":{s:c[s] for s in ("nominal_asymptotic","degenerate_or_near_degenerate","bootstrap_reference_unstable")}}
                            for key,c in counters.items()})
        print(json.dumps(dict(completed=scenario, **summaries[scenario]["family_metrics"])), flush=True)
    return summaries, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--draws", type=int, default=999)
    args = parser.parse_args()
    out = Path(__file__).parent / "results.v1"
    out.mkdir(exist_ok=False)
    before = {p:digest(REPO/p) for p in SOURCE_PATHS}
    write(out/"source-snapshot.json", {p:(REPO/p).read_text() for p in SOURCE_PATHS})
    write(out/"source-sha256.json", before)
    write(out/"existing-v5-audit.json", audit_existing())
    write(out/"independent-numeric-checks.json", hand_and_invariance_checks())
    summaries, records = stress(args.replications, args.draws)
    with (out/"stress-replicates.jsonl").open("x") as stream:
        for record in records: stream.write(json.dumps(record, allow_nan=False)+"\n")
    after = {p:digest(REPO/p) for p in SOURCE_PATHS}
    assert before == after, "production source changed during review; refuse to bind mixed versions"
    summary = dict(role="independent_synthetic_statistics_review_not_flight_or_method_results",
                   reviewed_at=datetime.now(timezone.utc).isoformat(), python=sys.version,
                   executable=sys.executable, numpy=np.__version__, source_sha256=before,
                   review_script_sha256=digest(Path(__file__)), sources_unchanged=True,
                   stress_configuration=dict(replications=args.replications, draws=args.draws,
                       bootstrap_seeds_fixed_to_production=True, independent_data_seed_scheme=[20261105,8101,"scenario_index","replication_index"],
                       sampling_unit="independent root, full condition vector", true_null="all six macro mean differences zero"),
                   scenarios=summaries,
                   limitations=["fixed finite synthetic scenarios; not universal calibration", "200-run Monte Carlo error quantified by Wilson intervals",
                       "999 bootstrap draws approximate the intended tail; no exact-test guarantee", "no physical, model/API, test or real holdout data"])
    write(out/"report.json", summary)


if __name__ == "__main__": main()
