"""Tests for the plan-compliance audit.

The behaviour that matters here is that a deviation from the plan is
detected, and that a plan edited after the fact is reported rather than
silently compared against the version that already matches the draft.
"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/plan_audit.py"
spec = importlib.util.spec_from_file_location("plan_audit", SCRIPT)
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)


def plan_item(identifier, kind, claim="c", data="d", priority="must"):
    return {"id": identifier, "kind": kind, "type": "planned", "question": "q",
            "claim": claim, "data": data, "section": "S", "priority": priority}


class ContractBaselineTests(unittest.TestCase):
    """The frozen contract is the baseline, and its caption mapping is three-state."""

    def contract(self, items):
        return {"version": 1, "contract_hash": "h", "items": items}

    def baseline(self, contract):
        baseline = dict(contract)
        baseline["items"] = pa.contract_items(contract)
        return baseline

    def test_contract_items_map_the_promise_to_its_caption(self):
        contract = self.contract([{"id": "T4", "kind": "table", "title": "t",
                                   "purpose": "p" * 12, "claim": "c",
                                   "acceptance_criteria": ["x"],
                                   "draft_label": "Table 1"}])
        items = pa.contract_items(contract)
        self.assertEqual(items[0]["delivered_as"], "Table 1")

    def test_a_promise_with_no_caption_is_not_matched_by_id(self):
        # T1's promise is the main-results table, which was never delivered.
        # Table 1 in the draft carries T4's promise, so matching T1 by number
        # would credit the wrong table.
        contract = self.contract([{"id": "T1", "kind": "table", "title": "t",
                                   "purpose": "p" * 12, "claim": "c",
                                   "acceptance_criteria": ["x"],
                                   "draft_label": ""}])
        results, _, _, _ = pa.audit(self.baseline(contract), None, None,
                                     "Table 1. Caption text.", set(), {1}, None, "hash")
        self.assertEqual(results[0]["verdict"], "undisclosed_deviation")
        self.assertIn("no draft caption", results[0]["delivered"])

    def test_a_rotated_promise_is_matched_to_its_own_caption(self):
        contract = self.contract([{"id": "T4", "kind": "table", "title": "t",
                                   "purpose": "p" * 12, "claim": "c",
                                   "acceptance_criteria": ["x"],
                                   "draft_label": "Table 1"}])
        results, _, _, _ = pa.audit(self.baseline(contract), None, None,
                                     "Table 1. Caption text.", set(), {1}, None, "hash")
        self.assertEqual(results[0]["verdict"], "fulfilled")

    def test_an_undecided_promise_is_still_matched_by_id(self):
        # No draft_label key at all: the host has not decided yet, so the
        # caption is matched the ordinary way.
        contract = self.contract([{"id": "T1", "kind": "table", "title": "t",
                                   "purpose": "p" * 12, "claim": "c",
                                   "acceptance_criteria": ["x"]}])
        baseline = self.baseline(contract)
        self.assertNotIn("delivered_as", baseline["items"][0])
        results, _, _, _ = pa.audit(baseline, None, None,
                                     "Table 1. Caption text.", set(), {1}, None, "hash")
        self.assertEqual(results[0]["verdict"], "fulfilled")


class CaptionDetectionTests(unittest.TestCase):
    def test_caption_at_end_of_line_is_found(self):
        # A two-column layout sets the caption at the end of a text line.
        text = ("... exercises.                                    "
                "Figure 2. Effect of the reference intervention, and how its sign "
                "depends on the branch. (a) Interpolated spectral radius.")
        figures, tables = pa.draft_labels(text)
        self.assertEqual(figures, {2})
        self.assertEqual(tables, set())

    def test_body_reference_is_not_a_caption(self):
        text = "Figure 2(b) is the operative result of this section: the same"
        figures, _ = pa.draft_labels(text)
        self.assertEqual(figures, set())

    def test_table_caption_is_found(self):
        text = "Table 1. Claim-by-claim check of the analysis against the recorded"
        _, tables = pa.draft_labels(text)
        self.assertEqual(tables, {1})

    def test_inline_reference_with_capital_is_not_a_caption(self):
        # "Figure 3 The design..." would match without the capital requirement.
        text = "as shown in Figure 3 the trend continues for all seeds"
        figures, _ = pa.draft_labels(text)
        self.assertEqual(figures, set())


class AuditTests(unittest.TestCase):
    def build(self, items, pending=None):
        return {"venue": "V", "items": items, "pending_visuals": pending or []}

    def test_delivered_item_is_fulfilled(self):
        plan = self.build([plan_item("F1", "figure")])
        results, rewrites, _, _ = pa.audit(
            plan, None, None, "Figure 1. Something. Text.", {1}, set(), None, "hash")
        self.assertEqual(results[0]["verdict"], "fulfilled")
        self.assertEqual(rewrites, [])

    def test_missing_item_is_undisclosed_deviation(self):
        plan = self.build([plan_item("F1", "figure"), plan_item("F2", "figure")])
        results, _, _, _ = pa.audit(
            plan, None, None, "Figure 1. Something. Text.", {1}, set(), None, "hash")
        verdicts = {r["id"]: r["verdict"] for r in results}
        self.assertEqual(verdicts["F1"], "fulfilled")
        self.assertEqual(verdicts["F2"], "undisclosed_deviation")

    def test_pending_data_downgrades_a_present_item(self):
        plan = self.build([plan_item("F1", "figure", data="TODO: run it")])
        results, _, _, _ = pa.audit(
            plan, None, None, "Figure 1. Something. Text.", {1}, set(), None, "hash")
        self.assertEqual(results[0]["verdict"], "disclosed_deviation")

    def test_item_without_a_number_goes_to_the_reviewer(self):
        # Without a number the item cannot be matched to a caption, so the
        # script must not report it as a failed deliverable.
        plan = self.build([{"id": "qualitative", "kind": "figure", "claim": "c",
                           "data": "d", "priority": "must"}])
        results, _, _, _ = pa.audit(
            plan, None, None, "Figure 1. Something. Text.", {1}, set(), None, "hash")
        self.assertEqual(results[0]["verdict"], "requires_reviewer")
        self.assertIn("no figure/table number", results[0]["delivered"])

    def test_deferred_item_goes_to_the_reviewer(self):
        plan = self.build([plan_item("F1", "figure")],
                          pending=[plan_item("F4", "figure",
                                             data="results.json:e2")])
        results, _, _, _ = pa.audit(
            plan, None, None, "Figure 1. Something. Text.", {1}, set(), None, "hash")
        deferred = [r for r in results if r["id"] == "F4"][0]
        self.assertEqual(deferred["verdict"], "requires_reviewer")
        self.assertIn("disclosed", deferred["reviewer_question"])

    def test_plan_rewrite_is_reported(self):
        plan = self.build([plan_item("F1", "figure")])
        snapshot = {"sha256": "different", "items": [plan_item("F1", "figure"),
                                                     plan_item("F9", "figure")]}
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(snapshot, handle)
            snapshot_path = handle.name
        try:
            _, rewrites, _, _ = pa.audit(
                plan, None, snapshot_path, "Figure 1. S. Text.", {1}, set(),
                None, "current")
        finally:
            Path(snapshot_path).unlink()
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(rewrites[0]["id"], "F9")
        self.assertEqual(rewrites[0]["change"], "removed")

    def test_identical_plan_reports_no_rewrite(self):
        plan = self.build([plan_item("F1", "figure")])
        snapshot = {"sha256": "same", "items": [plan_item("F1", "figure")]}
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(snapshot, handle)
            snapshot_path = handle.name
        try:
            _, rewrites, _, _ = pa.audit(
                plan, None, snapshot_path, "Figure 1. S. Text.", {1}, set(),
                None, "same")
        finally:
            Path(snapshot_path).unlink()
        self.assertEqual(rewrites, [])


class NumericPromiseTests(unittest.TestCase):
    def test_value_present_in_draft_is_fulfilled(self):
        plan = {"venue": "V", "items": [
            {"id": "F2", "kind": "figure", "claim": "c",
             "data": "the 1296 grid points and 54 critical points", "priority": "must"}]}
        results = pa.audit_numbers(plan, "we scanned 1296 points, 54 of them critical")
        values = {r["value"]: r["found_in_draft"] for r in results}
        self.assertTrue(values["1296"])
        self.assertTrue(values["54"])

    def test_value_absent_from_draft_is_flagged(self):
        plan = {"venue": "V", "items": [
            {"id": "T1", "kind": "table", "claim": "c",
             "data": "results from 8000 Monte Carlo paths", "priority": "must"}]}
        results = pa.audit_numbers(plan, "the draft says nothing about paths")
        entry = [r for r in results if r["value"] == "8000"][0]
        self.assertFalse(entry["found_in_draft"])
        self.assertEqual(entry["verdict"], "undisclosed_deviation")

    def test_scientific_notation_is_matched(self):
        plan = {"venue": "V", "items": [
            {"id": "F1", "kind": "figure", "claim": "c",
             "data": "policy MSE of 1.37e-05", "priority": "must"}]}
        results = pa.audit_numbers(plan, "the policy MSE is 1.37 x 10^-5")
        self.assertEqual(len(results), 1)
        entry = results[0]
        self.assertEqual(entry["value"], "1.37e-05")
        self.assertTrue(entry["found_in_draft"])

    def test_bare_small_integers_are_not_treated_as_promises(self):
        plan = {"venue": "V", "items": [
            {"id": "F1", "kind": "figure", "claim": "c",
             "data": "a single grid point", "priority": "must"}]}
        self.assertEqual(pa.audit_numbers(plan, "text"), [])

    def test_same_value_stated_twice_is_reported_once_per_item(self):
        plan = {"venue": "V", "items": [
            {"id": "F1", "kind": "figure", "claim": "c",
             "data": "1296 points", "priority": "must"}]}
        results = pa.audit_numbers(plan, "1296")
        self.assertEqual(len(results), 1)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="deli-plan-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), *args],
                              capture_output=True, text=True, cwd=str(self.root))

    def test_snapshot_then_audit(self):
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps({
            "venue": "V",
            "items": [plan_item("F1", "figure"), plan_item("F2", "figure")],
        }), encoding="utf-8")
        snapshot = self.root / "snap.json"
        result = self.run_cli("snapshot", "--plan", str(plan_path), "--out", str(snapshot))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(snapshot.exists())

        draft = self.root / "draft.txt"
        draft.write_text("Figure 1. Only one. Text here.", encoding="utf-8")
        out = self.root / "audit.json"
        result = self.run_cli("audit", "--plan", str(plan_path),
                              "--plan-snapshot", str(snapshot),
                              "--draft", str(plan_path),
                              "--draft-text", str(draft), "--out", str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(report["counts"]["fulfilled"], 1)
        self.assertEqual(report["counts"]["undisclosed_deviation"], 1)

    def test_missing_plan_file_exits_two(self):
        result = self.run_cli("snapshot", "--plan", str(self.root / "nope.json"),
                              "--out", str(self.root / "s.json"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("plan_audit:", result.stderr)


class OriginalPromiseSetTests(unittest.TestCase):
    """A plan narrowed after the fact must still be scored against what it promised.

    The failure mode: a plan rewritten to match the draft scores a perfect
    ratio against its own items, while the promise set it was written against
    shows that most visuals were never delivered. The plan may record the
    original promise set; when it does, that set is the honest baseline.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(SCRIPT), *args],
                              capture_output=True, text=True, cwd=str(self.root))

    def build(self, original_items):
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps({
            "venue": "V",
            "items": [plan_item("F1", "figure")],
            "pending_visuals": [plan_item("F4", "figure")],
            "original_promise_set": {"note": "before the narrowing", "items": original_items},
        }), encoding="utf-8")
        draft = self.root / "draft.txt"
        draft.write_text("Figure 1. Delivered. More text.\nTable 1. Also delivered.",
                         encoding="utf-8")
        snapshot = self.root / "snap.json"
        self.run_cli("snapshot", "--plan", str(plan_path), "--out", str(snapshot))
        out = self.root / "audit.json"
        result = self.run_cli("audit", "--plan", str(plan_path),
                              "--plan-snapshot", str(snapshot),
                              "--draft", str(plan_path),
                              "--draft-text", str(draft), "--out", str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(out.read_text(encoding="utf-8"))

    def test_original_promise_set_is_scored_separately(self):
        report = self.build([
            {"id": "F1", "kind": "figure", "claim": "c", "delivered_as": "Figure 1"},
            {"id": "F2", "kind": "figure", "claim": "c", "delivered_as": None},
        ])
        base = report["original_promise_set"]
        self.assertIsNotNone(base)
        self.assertEqual(base["promised"], 2)
        self.assertEqual(base["delivered"], 1)
        self.assertEqual(base["commitment_kept"], 0.5)
        # The narrowed plan's own ratio is reported too, and stays provisional.
        self.assertEqual(report["commitment_kept"], 1.0)
        self.assertTrue(report["commitment_kept_is_provisional"])

    def test_rotated_id_does_not_match_the_wrong_table(self):
        # T1 (main results) was never delivered; T4 (theory check) is the draft's
        # Table 1. Matching by the rotated id would credit T1 and blame T4.
        report = self.build([
            {"id": "T1", "kind": "table", "claim": "c", "delivered_as": None},
            {"id": "T4", "kind": "table", "claim": "c", "delivered_as": "Table 1"},
        ])
        rows = {r["id"]: r for r in report["original_promise_set"]["items"]}
        self.assertEqual(rows["T1"]["verdict"], "not_delivered")
        self.assertEqual(rows["T4"]["verdict"], "fulfilled")
        self.assertEqual(report["original_promise_set"]["commitment_kept"], 0.5)

    def test_absent_original_promise_set_leaves_the_baseline_null(self):
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps({
            "venue": "V", "items": [plan_item("F1", "figure")],
        }), encoding="utf-8")
        draft = self.root / "draft.txt"
        draft.write_text("Figure 1. Only one.", encoding="utf-8")
        snapshot = self.root / "snap.json"
        self.run_cli("snapshot", "--plan", str(plan_path), "--out", str(snapshot))
        out = self.root / "audit.json"
        result = self.run_cli("audit", "--plan", str(plan_path),
                              "--plan-snapshot", str(snapshot),
                              "--draft", str(plan_path),
                              "--draft-text", str(draft), "--out", str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(out.read_text(encoding="utf-8"))
        self.assertIsNone(report["original_promise_set"])


if __name__ == "__main__":
    unittest.main()
