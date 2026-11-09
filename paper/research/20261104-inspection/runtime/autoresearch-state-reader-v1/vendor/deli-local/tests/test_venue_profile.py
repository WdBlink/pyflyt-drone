"""Venue calibration measurement tests. Fixtures are synthetic, not real papers."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/venue_profile.py"
spec = importlib.util.spec_from_file_location("venue_profile", SCRIPT)
vp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vp)


def page(*blocks):
    return "\n".join(blocks)


SAMPLE_TEXT = "\f".join([
    page("A Fast Method for Cooperative Navigation",
         "Abstract. We present a method.",
         "Figure 1: System overview of the proposed architecture.",
         "(a) front-end  (b) back-end  (c) planner"),
    page("1 Introduction",
         "Our method is shown in Figure 1 and builds on prior work.",
         "Table 1: Main results on three benchmarks. Mean +- std over 5 seeds.",
         "Figure 2: Qualitative comparison against the baseline."),
    page("5 Conclusion", "We conclude here.", "References", "[1] A. Author. Title. 2020."),
])


class MeasurementTests(unittest.TestCase):
    def test_counts_figures_tables_and_pages(self):
        record = vp.measure_text(SAMPLE_TEXT, source="fixture")
        self.assertEqual(record["counts"]["figures"], 2)
        self.assertEqual(record["counts"]["tables"], 1)
        self.assertEqual(record["pages_total"], 3)
        self.assertEqual(record["pages_main_body"], 3)

    def test_main_body_stops_at_references_heading(self):
        text = "Figure 1: A.\nBody text.\nReferences\n[1] x.\nFigure 2: appendix figure."
        record = vp.measure_text(text)
        self.assertEqual(record["pages_main_body"], 1)
        self.assertEqual(record["counts"]["figures"], 2)
    def test_panels_counted_from_caption_markers(self):
        record = vp.measure_text(SAMPLE_TEXT)
        first = next(item for item in record["figures"] if item["number"] == 1)
        self.assertEqual(first["panel_count"], 3)
        second = next(item for item in record["figures"] if item["number"] == 2)
        self.assertEqual(second["panel_count"], 0)

    def test_subfigure_panels_before_parent_caption_are_found(self):
        # LaTeX emits '(a) caption' lines before the parent caption, with
        # blank lines in between; the real PDF case this models.
        text = ("(a) Front-end module.\n"
                "\n"
                "(b) Back-end module.\n"
                "\n"
                "Figure 1: System overview with two submodules.\n")
        record = vp.measure_text(text)
        first = next(item for item in record["figures"] if item["number"] == 1)
        self.assertEqual(first["panels"], ["a", "b"])

    def test_figure_without_subfigures_reports_no_panels(self):
        text = "Figure 1: A single panel figure with no subfigure labels.\n"
        record = vp.measure_text(text)
        self.assertEqual(record["counts"]["figure_panels"], 0)

    def test_inline_reference_is_not_counted_as_figure(self):
        record = vp.measure_text(SAMPLE_TEXT)
        numbers = [item["number"] for item in record["figures"]]
        self.assertEqual(numbers, [1, 2])
        self.assertEqual(len(numbers), len(set(numbers)))

    def test_roman_and_uppercase_labels(self):
        text = ("TABLE I: First table with results.\n"
                "Figure 3: A figure.\n"
                "Table II: Second table.\n")
        record = vp.measure_text(text)
        self.assertEqual(record["counts"]["tables"], 2)
        self.assertEqual(sorted(item["number"] for item in record["tables"]), [1, 2])
        self.assertEqual(record["counts"]["figures"], 1)

    def test_noncontiguous_numbering_warns(self):
        text = "Figure 1: one.\nFigure 5: five.\n"
        record = vp.measure_text(text)
        self.assertTrue(any("not contiguous" in item for item in record["warnings"]))

    def test_missing_references_warns(self):
        record = vp.measure_text("Figure 1: one.\n")
        self.assertTrue(any("no References heading" in item for item in record["warnings"]))

    def test_appendix_pointer_is_not_a_missing_body_caption(self):
        # "Figure 5 of Appendix E" names an appendix figure. Treating it as a
        # body caption we failed to detect would report a false undercount.
        text = ("Figure 1: The only body figure.\n"
                "We show an example in Figure 5 of Appendix E.\n")
        record = vp.measure_text(text)
        self.assertFalse(any("undercount" in item for item in record["warnings"]))

    def test_genuine_body_undercount_still_warns(self):
        text = ("Figure 1: one.\n"
                "As shown in Figure 3, the trend continues.\n")
        record = vp.measure_text(text)
        self.assertTrue(any("undercount" in item for item in record["warnings"]))
        self.assertTrue(any("3" in item for item in record["warnings"]))

    def test_result_table_hint(self):
        text = ("Table 1: Ablation study of each component.\n"
                "Table 2: Notation for all symbols.\n")
        record = vp.measure_text(text)
        hint = {item["number"]: item["result_hint"] for item in record["tables"]}
        self.assertTrue(hint[1])
        self.assertFalse(hint[2])


class AggregateTests(unittest.TestCase):
    def records(self, figures, tables, pages):
        return [{"source": f"s{index}.pdf", "venue_label": "V", "year": 2026,
                 "acceptance_evidence": "http://example.invalid/proof",
                 "pages_main_body": page, "pages_total": page,
                 "counts": {"figures": figure, "tables": table,
                            "figure_panels": figure * 2, "result_tables": table},
                 "figures": [], "tables": [], "warnings": []}
                for index, (figure, table, page) in enumerate(zip(figures, tables, pages))]

    def test_budget_band_uses_median_and_interquartile_range(self):
        profile = vp.build_profile(self.records([4, 5, 6, 7], [2, 2, 3, 3], [8, 8, 9, 9]),
                                   "V", "d")
        self.assertEqual(profile["budget"]["figures"]["target"], 6)
        self.assertEqual(profile["budget"]["figures"]["min"], 4)
        self.assertEqual(profile["budget"]["figures"]["max"], 7)

    def test_budget_band_keeps_typical_counts_and_drops_outliers(self):
        # The band constrains our own plan, so it must not exclude a count that
        # a quarter of the sampled papers actually use; a lone outlier is the
        # only value it should leave out.
        profile = vp.build_profile(self.records([3, 3, 3, 11], [2, 2, 2, 2], [8] * 4),
                                   "V", "d")
        band = profile["budget"]["figures"]
        self.assertEqual(band["min"], 3)
        self.assertLess(band["max"], 11)

    def test_unverified_samples_are_listed(self):
        records = self.records([4], [2], [8])
        records[0].pop("acceptance_evidence")
        profile = vp.build_profile(records, "V", "d")
        self.assertEqual(len(profile["unverified_samples"]), 1)

    def test_mixed_page_bases_are_flagged(self):
        records = self.records([4, 4], [2, 2], [8, 8])
        records[0]["pages_basis"] = "references_heading"
        records[1]["pages_basis"] = "total_pages_fallback"
        profile = vp.build_profile(records, "V", "d")
        self.assertTrue(profile["pages_basis_mixed"])
        self.assertTrue(any("page-count bases" in item for item in profile["warnings"]))

    def test_uniform_page_basis_is_not_flagged(self):
        records = self.records([4, 4], [2, 2], [8, 8])
        for record in records:
            record["pages_basis"] = "references_heading"
        profile = vp.build_profile(records, "V", "d")
        self.assertFalse(profile["pages_basis_mixed"])
        self.assertEqual(profile["pages_basis"], "references_heading")

    def test_design_patterns_aggregate_types_and_placements(self):
        records = self.records([2], [1], [8])
        design = {"papers": {"S01": {
            "figures": [{"number": 1, "type": "method_overview", "claim": "c",
                         "placement": "body"}],
            "tables": [{"number": 1, "type": "main_results", "claim": "c",
                        "placement": "body"}]}}}
        profile = vp.build_profile(records, "V", "d", design)
        patterns = profile["design_patterns"]
        self.assertEqual(patterns["figure_types"]["method_overview"]["papers"], 1)
        self.assertEqual(patterns["table_types"]["main_results"]["papers"], 1)
        self.assertEqual(patterns["placements"]["body"], 2)


class PlanCheckTests(unittest.TestCase):
    def profile(self, figure_share=True):
        records = [{"source": "s.pdf", "venue_label": "V", "year": 2026,
                    "acceptance_evidence": "proof", "pages_main_body": 8, "pages_total": 8,
                    "counts": {"figures": 5, "tables": 3, "figure_panels": 10,
                               "result_tables": 2}, "figures": [], "tables": [], "warnings": []}]
        design = None
        if figure_share:
            design = {"papers": {"S01": {
                "figures": [{"number": 1, "type": "qualitative_comparison", "claim": "c",
                             "placement": "body"}],
                "tables": [{"number": 1, "type": "main_results", "claim": "c",
                            "placement": "body"}]}}}
        return vp.build_profile(records, "V", "d", design)

    def item(self, **kwargs):
        base = {"id": "F1", "kind": "figure", "type": "method_overview",
                "question": "How does the data flow between modules?",
                "claim": "The data flow is unidirectional and composable",
                "data": "architecture diagram", "section": "§3", "priority": "must"}
        base.update(kwargs)
        return base

    def test_valid_plan_passes(self):
        plan = {"items": [self.item(),
                          self.item(id="T1", kind="table", type="main_results",
                                     question="Does the method beat the baselines?",
                                     data="results.json:main", section="§5")]}
        report = vp.check_plan(self.profile(), plan)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("qualitative_comparison" in item for item in report["errors"]))

    def test_missing_field_is_error(self):
        plan = {"items": [{"id": "F1", "kind": "figure", "type": "method_overview"}]}
        report = vp.check_plan(self.profile(figure_share=False), plan)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("missing required field" in item for item in report["errors"]))

    def test_question_equal_claim_is_error(self):
        plan = {"items": [self.item(question="same", claim="same")]}
        report = vp.check_plan(self.profile(figure_share=False), plan)
        self.assertTrue(any("must differ" in item for item in report["errors"]))

    def test_duplicate_id_is_error(self):
        plan = {"items": [self.item(), self.item()]}
        report = vp.check_plan(self.profile(figure_share=False), plan)
        self.assertTrue(any("duplicate id" in item for item in report["errors"]))

    def test_unknown_type_is_error(self):
        plan = {"items": [self.item(type="sparkline")]}
        report = vp.check_plan(self.profile(figure_share=False), plan)
        self.assertTrue(any("unknown figure type" in item for item in report["errors"]))

    def test_out_of_band_count_needs_reason(self):
        plan = {"items": [self.item(id=f"F{n}") for n in range(1, 12)]}
        report = vp.check_plan(self.profile(figure_share=False), plan)
        self.assertTrue(any("deviation_reason" in item for item in report["errors"]))

    def test_out_of_band_count_with_reason_is_warning(self):
        plan = {"items": [self.item(id=f"F{n}") for n in range(1, 12)],
                "deviation_reason": "The method needs one extra ablation visual"}
        report = vp.check_plan(self.profile(figure_share=False), plan)
        self.assertEqual(report["status"], "warn")
        self.assertFalse(any("deviation_reason" in item for item in report["errors"]))

    def test_dominant_venue_pattern_must_be_covered(self):
        plan = {"items": [self.item()]}
        report = vp.check_plan(self.profile(figure_share=True), plan)
        self.assertTrue(any("qualitative_comparison" in item for item in report["errors"]))

    def test_pending_data_on_must_item_warns(self):
        plan = {"items": [self.item(data="TODO: run experiment")]}
        report = vp.check_plan(self.profile(figure_share=False), plan)
        self.assertTrue(any("pending data" in item for item in report["warnings"]))


class RealizeCheckTests(unittest.TestCase):
    def plan(self, figures=1, tables=1, data="results.json:main"):
        return {"items": [self.item(id=f"F{n}", kind="figure", data=data)
                          for n in range(1, figures + 1)]
                + [self.item(id=f"T{n}", kind="table", data=data)
                   for n in range(1, tables + 1)]}

    def item(self, **kwargs):
        base = {"id": "F1", "kind": "figure", "type": "method_overview",
                "question": "How does the data flow?", "claim": "It is unidirectional",
                "data": "architecture diagram", "section": "§3", "priority": "must"}
        base.update(kwargs)
        return base

    def test_matching_counts_pass(self):
        measured = {"counts": {"figures": 1, "tables": 1}, "pages_main_body": 8,
                    "pages_total": 8, "warnings": []}
        report = vp.check_realize(self.plan(), measured)
        self.assertEqual(report["status"], "pass")

    def test_count_mismatch_is_error(self):
        measured = {"counts": {"figures": 3, "tables": 0}, "pages_main_body": 8,
                    "pages_total": 8, "warnings": []}
        report = vp.check_realize(self.plan(), measured)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("figures" in item for item in report["errors"]))

    def test_pending_must_data_is_error(self):
        measured = {"counts": {"figures": 1, "tables": 1}, "pages_main_body": 8,
                    "pages_total": 8, "warnings": []}
        report = vp.check_realize(self.plan(data="TODO"), measured)
        self.assertTrue(any("pending data" in item for item in report["errors"]))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="deli-venue-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), *args],
                              capture_output=True, text=True, cwd=str(self.root))

    def test_measure_from_text_file(self):
        text = self.root / "p1.txt"
        text.write_text(SAMPLE_TEXT, encoding="utf-8")
        out = self.root / "p1.json"
        result = self.run_cli("measure", "--text", str(text), "--out", str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(record["counts"]["figures"], 2)
        self.assertEqual(record["counts"]["tables"], 1)

    def test_missing_file_exits_two(self):
        result = self.run_cli("measure", "--text", str(self.root / "nope.txt"),
                              "--out", str(self.root / "out.json"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("venue_profile:", result.stderr)

    def test_measure_requires_exactly_one_input(self):
        result = self.run_cli("measure", "--out", str(self.root / "out.json"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("exactly one", result.stderr)

    def test_profile_and_plan_check_roundtrip(self):
        text = self.root / "p1.txt"
        text.write_text(SAMPLE_TEXT, encoding="utf-8")
        measured = self.root / "p1.json"
        self.assertEqual(self.run_cli("measure", "--text", str(text), "--out", str(measured)).returncode, 0)
        measurements = self.root / "measurements.json"
        record = json.loads(measured.read_text(encoding="utf-8"))
        record["sample_id"] = "S01"
        measurements.write_text(json.dumps({"samples": [record]}), encoding="utf-8")
        profile = self.root / "profile.json"
        result = self.run_cli("profile", "--measurements", str(measurements),
                              "--venue", "V", "--direction", "d", "--out", str(profile))
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"items": [{
            "id": "F1", "kind": "figure", "type": "method_overview",
            "question": "How does the data flow between modules?",
            "claim": "The data flow is unidirectional and composable",
            "data": "architecture diagram", "section": "§3", "priority": "must"}]}),
            encoding="utf-8")
        check = self.run_cli("check-plan", "--profile", str(profile), "--plan", str(plan))
        self.assertEqual(check.returncode, 0, check.stderr)
        report = json.loads(check.stdout)
        self.assertIn(report["status"], ("pass", "warn", "fail"))


if __name__ == "__main__":
    unittest.main()
