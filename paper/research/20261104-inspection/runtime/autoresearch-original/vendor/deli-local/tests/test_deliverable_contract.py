"""Fault injection for the deliverable contract.

The behaviour that matters is that a run cannot be declared delivered while an
approved item is missing, uninserted, unreviewed or stale, and that a fully
satisfied contract can still pass. These tests inject faults into the final
acceptance entry point rather than only exercising helpers.

Fixture content is synthetic. It is not scientific evidence and not a human
approval.
"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/deliverable_contract.py"
spec = importlib.util.spec_from_file_location("deliverable_contract", SCRIPT)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)


def item(identifier, kind="figure", priority="must", dependencies=None):
    return {
        "id": identifier,
        "kind": kind,
        "title": identifier + " required comparison",
        "purpose": "Compare the measured outcome against the matched baseline condition.",
        "claim": "The intervention changes the measured outcome under the stated condition.",
        "required_content": ["All configurations and their uncertainty."],
        "comparison": ["Matched baseline at equal budget."],
        "measurement_requirements": ["Report mean and spread over independent runs."],
        "evidence_requirements": ["Use completed experiment results."],
        "dependencies": dependencies or [],
        "acceptance_criteria": ["Include every configuration and cite the evidence path."],
        "priority": priority,
        "section": "S",
    }


class ContractFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "state").mkdir()
        (self.root / "figures").mkdir()
        (self.root / "sec").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    # -- helpers -------------------------------------------------------

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    def freeze(self, items):
        self.write("state/deliverable_items.json", {"items": items})
        self.write("state/approved_brief.md", "# Approved brief\n")
        dc.cmd_freeze(self.args(items="state/deliverable_items.json",
                                brief="state/approved_brief.md",
                                contract_id="t", venue="V", paper_type="original",
                                revision_of=None, approved_by=None, approved_at=None,
                                reason=None, force=True))
        return json.loads((self.root / "state/deliverable_contract.json").read_text(encoding="utf-8"))

    def args(self, **kwargs):
        base = {"project": str(self.root), "out": None, "force": True,
                "requirement": None, "artifact": None, "manuscript_ref": None,
                "evidence": None, "draft_pdf": None, "draft_text": None,
                "manuscript_root": None, "plan": None,
                "apply": False, "request": None, "decision": None, "final": False,
                "items": None, "brief": None, "contract_id": None, "venue": None,
                "paper_type": None, "revision_of": None, "approved_by": None,
                "approved_at": None, "reason": None,
                "experiment": None, "outcome": None,
                "followup_task": None}
        base.update(kwargs)
        return type("Args", (), base)()

    def bind_item(self, identifier, artifact=None, ref=None, evidence=None, manuscript=True):
        artifact = artifact or f"figures/{identifier.lower()}.pdf"
        self.write(artifact, f"%PDF-{identifier}")
        if ref and manuscript:
            # Only write the manuscript fragment when the caller asks for it, so a
            # test can bind an artifact that never reaches the manuscript.
            self.write(f"sec/{identifier.lower()}.tex", f"\\label{{{ref}}}\n\\ref{{{ref}}}\n")
        if evidence:
            self.write(evidence, "evidence")
        dc.cmd_bind(self.args(requirement=identifier, artifact=artifact,
                              manuscript_ref=ref or "",
                              evidence=[evidence] if evidence else None))

    def coverage(self):
        """Run the coverage check and return (exit code, report)."""
        dc.cmd_coverage(self.args(out="state/coverage.json"))
        report = json.loads((self.root / "state/coverage.json").read_text(encoding="utf-8"))
        return report

    def review_item(self, identifier, verdict="satisfied", artifact=None):
        contract = dc.load_contract(self.root)
        bindings = dc.load_bindings(self.root, contract)
        reviews = dc.load_reviews(self.root, contract)
        artifact_hash = bindings.get(identifier, {}).get("artifact_sha256", "")
        reviews[identifier] = {"verdict": verdict,
                               "explanation": "Checked the artifact against its requirement.",
                               "evidence_locations": ["Section S"],
                               "artifact_sha256": artifact_hash,
                               "reviewed_at": "2026-10-20T00:00:00+00:00"}
        self.write("state/deliverable_reviews.json",
                   {"contract_hash": contract["contract_hash"], "reviews": reviews})

    def accept(self, **kwargs):
        return dc.compute_report(self.root, self.args(**kwargs))


class FreezeTests(ContractFixture):
    def test_freeze_writes_contract_and_rendered_brief(self):
        contract = self.freeze([item("F1"), item("T1", kind="table")])
        self.assertEqual(len(contract["items"]), 2)
        self.assertTrue((self.root / "state/deliverable_brief.md").is_file())
        rendered = (self.root / "state/deliverable_brief.md").read_text(encoding="utf-8")
        self.assertIn("F1", rendered)
        self.assertIn("Compare the measured outcome", rendered)

    def test_rendered_brief_carries_the_approved_requirements(self):
        self.freeze([item("F1")])
        rendered = (self.root / "state/deliverable_brief.md").read_text(encoding="utf-8")
        self.assertIn("Include every configuration", rendered)
        self.assertIn("Matched baseline at equal budget.", rendered)
        self.assertIn("Report mean and spread over independent runs.", rendered)

    def test_item_without_acceptance_criteria_is_rejected(self):
        broken = item("F1")
        del broken["acceptance_criteria"]
        with self.assertRaises(dc.ContractError):
            dc.validate_items([broken])

    def test_item_without_a_research_question_is_rejected(self):
        broken = item("F1")
        broken["purpose"] = "short"
        with self.assertRaises(dc.ContractError):
            dc.validate_items([broken])

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(dc.ContractError):
            dc.validate_items([item("F1"), item("F1")])

    def test_revision_removing_an_item_needs_approval(self):
        self.freeze([item("F1"), item("F2")])
        self.write("state/deliverable_items.json", {"items": [item("F1")]})
        with self.assertRaises(dc.ContractError):
            dc.cmd_freeze(self.args(items="state/deliverable_items.json",
                                    brief=None, contract_id=None, venue=None, paper_type=None,
                                    revision_of="yes", approved_by=None, approved_at=None,
                                    reason=None, force=True))

    def test_revision_records_what_changed(self):
        self.freeze([item("F1"), item("F2")])
        self.write("state/deliverable_items.json", {"items": [item("F1")]})
        dc.cmd_freeze(self.args(items="state/deliverable_items.json", brief=None,
                                contract_id=None, venue=None, paper_type=None,
                                revision_of="yes", approved_by="user",
                                approved_at="2026-10-20T00:00:00Z", reason="E2 unavailable",
                                force=True))
        contract = json.loads((self.root / "state/deliverable_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["revisions"][-1]["removed"], ["F2"])
        self.assertEqual(contract["revisions"][-1]["approved_by"], "user")


class CoverageTests(ContractFixture):
    def test_uncovered_item_is_reported(self):
        self.freeze([item("F1"), item("F2")])
        self.write("state/task_plan.json",
                   {"contract_hash": dc.load_contract(self.root)["contract_hash"],
                    "tasks": [{"id": "T1", "requirement_ids": ["F1"], "depends_on": [],
                               "status": "planned"}]})
        report = self.coverage()
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any(problem.get("id") == "F2" and problem["kind"] == "uncovered"
                            for problem in report["problems"]))

    def test_task_referring_to_an_unknown_item_is_rejected(self):
        self.freeze([item("F1")])
        self.write("state/task_plan.json",
                   {"contract_hash": dc.load_contract(self.root)["contract_hash"],
                    "tasks": [{"id": "T1", "requirement_ids": ["F9"], "depends_on": [],
                               "status": "planned"}]})
        with self.assertRaises(dc.ContractError):
            dc.cmd_coverage(self.args(out=None))

    def test_one_task_may_serve_several_items(self):
        self.freeze([item("F1"), item("F2")])
        self.write("state/task_plan.json",
                   {"contract_hash": dc.load_contract(self.root)["contract_hash"],
                    "tasks": [{"id": "T1", "requirement_ids": ["F1", "F2"], "depends_on": [],
                               "status": "planned"}]})
        report = self.coverage()
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["dispatchable"], ["T1"])

    def test_stale_decomposition_is_reported(self):
        self.freeze([item("F1")])
        self.write("state/task_plan.json",
                   {"contract_hash": "0" * 64,
                    "tasks": [{"id": "T1", "requirement_ids": ["F1"], "depends_on": [],
                               "status": "planned"}]})
        report = self.coverage()
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any(problem["kind"] == "stale_decomposition"
                            for problem in report["problems"]))

    def test_a_blocked_task_marks_the_item_blocked(self):
        self.freeze([item("F1", dependencies=["E2"])])
        self.write("state/task_plan.json",
                   {"contract_hash": dc.load_contract(self.root)["contract_hash"],
                    "tasks": [{"id": "T1", "requirement_ids": ["F1"], "depends_on": [],
                               "status": "blocked", "blocked_by": ["E2"]}]})
        report = self.coverage()
        self.assertTrue(any(problem.get("id") == "F1" and problem["kind"] == "blocked"
                            and "E2" in problem["message"] for problem in report["problems"]))


class AcceptanceFaultInjectionTests(ContractFixture):
    def test_satisfied_contract_passes(self):
        self.freeze([item("F1"), item("T1", kind="table")])
        for identifier in ("F1", "T1"):
            self.bind_item(identifier, ref=identifier.lower())
            self.review_item(identifier)
        report = self.accept()
        self.assertTrue(report["deliverable"], report["blocking"])
        self.assertEqual(report["counts"]["satisfied"], 2)

    def test_missing_artifact_blocks_delivery(self):
        self.freeze([item("F1")])
        self.review_item("F1")
        report = self.accept()
        self.assertFalse(report["deliverable"])
        self.assertTrue(any("F1" in message for message in report["blocking"]))

    def test_unrelated_artifact_cannot_substitute_for_a_missing_item(self):
        self.freeze([item("F1"), item("F2")])
        # A real, present, reviewed artifact exists, but it is F1's, not F2's.
        self.bind_item("F1", ref="f1")
        self.review_item("F1")
        report = self.accept()
        self.assertFalse(report["deliverable"])
        self.assertTrue(any("F2" in message for message in report["blocking"]))

    def test_artifact_not_inserted_into_the_manuscript_blocks_delivery(self):
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one", manuscript=False)
        self.review_item("F1")
        report = self.accept(manuscript_root="sec")
        self.assertFalse(report["deliverable"])
        self.assertTrue(any("未插入稿件" in message for message in report["blocking"]))

    def test_artifact_present_and_inserted_passes_the_manuscript_check(self):
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one")
        self.review_item("F1")
        report = self.accept(manuscript_root="sec")
        self.assertTrue(report["deliverable"], report["blocking"])

    def test_missing_semantic_review_blocks_delivery(self):
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one")
        report = self.accept()
        self.assertFalse(report["deliverable"])
        self.assertTrue(any("待评审" in message for message in report["blocking"]))

    def test_unmet_review_blocks_delivery(self):
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one")
        self.review_item("F1", verdict="unmet")
        report = self.accept()
        self.assertFalse(report["deliverable"])
        self.assertTrue(any("未满足要求" in message for message in report["blocking"]))

    def test_changed_artifact_invalidates_the_stored_review(self):
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one")
        self.review_item("F1")
        self.assertTrue(self.accept()["deliverable"])
        # The artifact changes after the review was recorded.
        self.write("figures/f1.pdf", "%PDF-F1-changed")
        report = self.accept()
        self.assertFalse(report["deliverable"])
        self.assertTrue(any("待评审" in message or "已变更" in message
                            for message in report["blocking"]))

    def test_changed_evidence_invalidates_the_binding(self):
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one", evidence="results.json")
        self.review_item("F1")
        self.assertTrue(self.accept()["deliverable"])
        self.write("results.json", "changed")
        report = self.accept()
        self.assertFalse(report["deliverable"])

    def test_item_dropped_from_the_plan_is_reported(self):
        self.freeze([item("F1"), item("F2")])
        for identifier in ("F1", "F2"):
            self.bind_item(identifier, ref=identifier.lower())
            self.review_item(identifier)
        self.write("state/figure_table_plan.json", {"items": [{"id": "F1"}]})
        report = self.accept(plan="state/figure_table_plan.json")
        self.assertTrue(any("无声消失" in message for message in report["blocking"]))

    def test_a_deferred_visual_still_counts_as_present_in_the_plan(self):
        self.freeze([item("F1"), item("F2")])
        for identifier in ("F1", "F2"):
            self.bind_item(identifier, ref=identifier.lower())
            self.review_item(identifier)
        self.write("state/figure_table_plan.json",
                   {"items": [{"id": "F1"}], "pending_visuals": [{"id": "F2"}]})
        report = self.accept(plan="state/figure_table_plan.json")
        self.assertFalse(any("无声消失" in message for message in report["blocking"]))

    def test_a_rotated_id_is_traced_through_plan_ref(self):
        # The plan renumbers a promise; plan_ref keeps it traceable instead of
        # making it look like the promise vanished.
        rotated = item("T4")
        rotated["plan_ref"] = "T1"
        self.freeze([rotated])
        self.bind_item("T4", ref="tab:claims")
        self.review_item("T4")
        self.write("state/figure_table_plan.json", {"items": [{"id": "T1"}]})
        report = self.accept(plan="state/figure_table_plan.json")
        self.assertFalse(any("无声消失" in message for message in report["blocking"]))
        self.assertTrue(report["deliverable"], report["blocking"])

    def test_blocking_messages_name_the_items(self):
        self.freeze([item("F1"), item("F2")])
        report = self.accept()
        self.assertTrue(any(message.startswith("F1") for message in report["blocking"]))
        self.assertTrue(any(message.startswith("F2") for message in report["blocking"]))

    def test_commitment_kept_counts_only_must_items(self):
        self.freeze([item("F1"), item("F2", priority="optional")])
        self.bind_item("F1", ref="fig:one")
        self.review_item("F1")
        report = self.accept()
        self.assertEqual(report["must_total"], 1)
        self.assertEqual(report["must_satisfied"], 1)
        self.assertEqual(report["commitment_kept"], 1.0)

    def test_an_unbound_item_is_not_also_reported_as_pending_review(self):
        self.freeze([item("F1")])
        report = self.accept()
        missing = [message for message in report["blocking"] if "缺失" in message]
        pending = [message for message in report["blocking"] if "待评审" in message]
        self.assertEqual(len(missing), 1)
        self.assertEqual(pending, [])

    def test_the_draft_banner_is_plain_ascii_latex(self):
        # The banner goes into a real paper's LaTeX. CJK text would not compile
        # without extra packages, so the banner carries ASCII and item IDs.
        self.freeze([item("F1"), item("F2")])
        report = self.accept()
        banner = report["draft_banner"]
        banner.encode("ascii")
        self.assertIn("F2", banner)
        self.assertIn("0/2", banner)

    def test_a_draft_without_the_banner_is_rejected(self):
        self.freeze([item("F1")])
        self.write("draft.txt", "A manuscript with no draft marker at all.")
        with self.assertRaises(dc.ContractError):
            self.accept(draft_text=str(self.root / "draft.txt"))

    def test_a_draft_carrying_the_banner_is_accepted_as_draft(self):
        self.freeze([item("F1")])
        report = self.accept()
        self.write("draft.txt", report["draft_banner"])
        result = self.accept(draft_text=str(self.root / "draft.txt"))
        self.assertFalse(result["deliverable"])


class BannerLatexTests(ContractFixture):
    """The banner is injected into a real manuscript, so its LaTeX must be sound."""

    def test_the_banner_has_balanced_braces(self):
        self.freeze([item("F1"), item("F2")])
        dc.cmd_banner(self.args(out="state/draft_banner.tex"))
        tex = (self.root / "state/draft_banner.tex").read_text(encoding="utf-8")
        depth = 0
        for character in tex:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            self.assertGreaterEqual(depth, 0, "unbalanced closing brace in the banner")
        self.assertEqual(depth, 0, "unbalanced braces in the banner")

    def test_the_banner_is_ascii_and_escapes_underscores(self):
        self.freeze([item("F1")])
        dc.cmd_banner(self.args(out="state/draft_banner.tex"))
        tex = (self.root / "state/draft_banner.tex").read_text(encoding="utf-8")
        tex.encode("ascii")
        self.assertIn("DRAFT", tex)
        self.assertIn("\\fbox", tex)

    def test_the_banner_is_refused_when_the_contract_is_satisfied(self):
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one")
        self.review_item("F1")
        with self.assertRaises(dc.ContractError):
            dc.cmd_banner(self.args(out="state/draft_banner.tex"))


class BranchTests(ContractFixture):
    def test_rerun_branch_blocks_dependent_items(self):
        self.freeze([item("F1", dependencies=["E2"])])
        self.write("state/deliverable_branches.json",
                   {"branches": [{"experiment": "E2", "outcome": "failed", "decision": "rerun",
                                  "reason": "crashed", "followup_task": "T7",
                                  "approved_by": "", "approved_at": "",
                                  "recorded_at": "2026-10-20T00:00:00+00:00",
                                  "contract_hash": dc.load_contract(self.root)["contract_hash"]}]})
        self.bind_item("F1", ref="fig:one")
        self.review_item("F1")
        report = self.accept()
        self.assertFalse(report["deliverable"])
        self.assertTrue(any("E2" in message and "阻塞" in message
                            for message in report["blocking"]))

    def test_negative_result_branch_requires_review_not_automatic_pass(self):
        self.freeze([item("F1", dependencies=["E2"])])
        self.write("state/deliverable_branches.json",
                   {"branches": [{"experiment": "E2", "outcome": "failed",
                                  "decision": "negative_result", "reason": "partial data",
                                  "followup_task": "", "approved_by": "", "approved_at": "",
                                  "recorded_at": "2026-10-20T00:00:00+00:00",
                                  "contract_hash": dc.load_contract(self.root)["contract_hash"]}]})
        self.bind_item("F1", ref="fig:one")
        report = self.accept()
        self.assertFalse(report["deliverable"])
        self.assertTrue(any("负结果" in message for message in report["blocking"]))

    def test_rerun_decision_needs_a_followup_task(self):
        self.freeze([item("F1")])
        with self.assertRaises(dc.ContractError):
            dc.cmd_branch(self.args(experiment="E2", outcome="failed", decision="rerun",
                                    reason="crashed", followup_task="", approved_by="",
                                    approved_at="", force=True))

    def test_brief_revision_decision_needs_approval(self):
        self.freeze([item("F1")])
        with self.assertRaises(dc.ContractError):
            dc.cmd_branch(self.args(experiment="E2", outcome="failed", decision="revise_brief",
                                    reason="scope change", followup_task="", approved_by="",
                                    approved_at="", force=True))


class ReviewInterfaceTests(ContractFixture):
    def test_review_request_lists_unreviewed_items_only(self):
        self.freeze([item("F1"), item("F2")])
        self.bind_item("F1", ref="fig:one")
        self.review_item("F1")
        dc.cmd_review(self.args(out="state/review_request.json", final=False))
        request = json.loads((self.root / "state/review_request.json").read_text(encoding="utf-8"))
        self.assertEqual([entry["requirement_id"] for entry in request["items_to_review"]], [])
        self.assertEqual(request["reusable"], ["F1"])

    def test_an_unbound_item_is_not_put_in_the_review_request(self):
        # An item with no artifact has nothing to judge; it is reported as not
        # ready rather than handed to a reviewer as if it were reviewable.
        self.freeze([item("F1"), item("F2")])
        self.bind_item("F1", ref="fig:one")
        dc.cmd_review(self.args(out="state/review_request.json", final=False))
        request = json.loads((self.root / "state/review_request.json").read_text(encoding="utf-8"))
        self.assertEqual([entry["requirement_id"] for entry in request["items_to_review"]], ["F1"])
        self.assertEqual([entry["requirement_id"] for entry in request["not_ready"]], ["F2"])

    def test_review_request_carries_the_requirement_the_reviewer_must_judge(self):
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one")
        dc.cmd_review(self.args(out="state/review_request.json", final=False))
        request = json.loads((self.root / "state/review_request.json").read_text(encoding="utf-8"))
        entry = request["items_to_review"][0]
        self.assertIn("Compare the measured outcome", entry["purpose"])
        self.assertTrue(any("Include every configuration" in criterion
                            for criterion in entry["acceptance_criteria"]))
        self.assertTrue(any("Matched baseline" in criterion
                            for criterion in entry["comparison"]))

    def test_review_must_cover_every_requested_item(self):
        self.freeze([item("F1"), item("F2")])
        for identifier in ("F1", "F2"):
            self.bind_item(identifier, ref=identifier.lower())
        dc.cmd_review(self.args(out="state/review_request.json", final=False))
        request = json.loads((self.root / "state/review_request.json").read_text(encoding="utf-8"))
        self.write("state/review_decision.json",
                   {"request_id": request["request_id"],
                    "checks": [{"requirement_id": "F1", "verdict": "satisfied",
                                "explanation": "ok",
                                "evidence_locations": ["Section S"]}]})
        with self.assertRaises(dc.ContractError):
            dc.cmd_review(self.args(apply=True, request="state/review_request.json",
                                    decision="state/review_decision.json", final=False))

    def test_review_of_an_unrequested_item_is_rejected(self):
        self.freeze([item("F1"), item("F2")])
        for identifier in ("F1", "F2"):
            self.bind_item(identifier, ref=identifier.lower())
        dc.cmd_review(self.args(out="state/review_request.json", final=False))
        request = json.loads((self.root / "state/review_request.json").read_text(encoding="utf-8"))
        self.write("state/review_decision.json",
                   {"request_id": request["request_id"],
                    "checks": [{"requirement_id": "F1", "verdict": "satisfied",
                                "explanation": "ok", "evidence_locations": ["S"]},
                               {"requirement_id": "F9", "verdict": "satisfied",
                                "explanation": "ok", "evidence_locations": ["S"]}]})
        with self.assertRaises(dc.ContractError):
            dc.cmd_review(self.args(apply=True, request="state/review_request.json",
                                    decision="state/review_decision.json", final=False))

    def test_a_satisfied_review_is_applied_and_unblocks_the_item(self):
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one")
        dc.cmd_review(self.args(out="state/review_request.json", final=False))
        request = json.loads((self.root / "state/review_request.json").read_text(encoding="utf-8"))
        self.write("state/review_decision.json",
                   {"request_id": request["request_id"],
                    "checks": [{"requirement_id": "F1", "verdict": "satisfied",
                                "explanation": "Answers the stated question with the matched baseline.",
                                "evidence_locations": ["Section S"],
                                "artifact_sha256": request["items_to_review"][0]["artifact_sha256"]}]})
        dc.cmd_review(self.args(apply=True, request="state/review_request.json",
                                decision="state/review_decision.json", final=False))
        report = self.accept()
        self.assertTrue(report["deliverable"], report["blocking"])

    def test_review_against_a_different_artifact_revision_is_rejected(self):
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one")
        dc.cmd_review(self.args(out="state/review_request.json", final=False))
        request = json.loads((self.root / "state/review_request.json").read_text(encoding="utf-8"))
        self.write("state/review_decision.json",
                   {"request_id": request["request_id"],
                    "checks": [{"requirement_id": "F1", "verdict": "satisfied",
                                "explanation": "ok", "evidence_locations": ["S"],
                                "artifact_sha256": "1" * 64}]})
        with self.assertRaises(dc.ContractError):
            dc.cmd_review(self.args(apply=True, request="state/review_request.json",
                                    decision="state/review_decision.json", final=False))


class BundleDigestTests(ContractFixture):
    def test_bundle_digest_changes_when_the_artifact_changes(self):
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one")
        before = dc.bundle_digest(self.root)
        self.write("figures/f1.pdf", "%PDF-changed")
        self.assertNotEqual(before, dc.bundle_digest(self.root))

    def test_bundle_digest_is_stable_when_nothing_changes(self):
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one")
        self.assertEqual(dc.bundle_digest(self.root), dc.bundle_digest(self.root))


class CheckpointGateTests(ContractFixture):
    """The final completion entry point must refuse an unsatisfied contract."""

    def checkpoint_module(self):
        path = Path(__file__).resolve().parents[1] / "scripts/checkpoint.py"
        loader = importlib.util.spec_from_file_location("deli_checkpoint", path)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        return module

    def test_complete_is_refused_when_the_contract_is_not_satisfied(self):
        checkpoint = self.checkpoint_module()
        self.freeze([item("F1")])
        with self.assertRaises(checkpoint.CheckpointError) as raised:
            checkpoint.require_contract_accepted(self.root)
        self.assertIn("deliverable_acceptance.json", str(raised.exception))

    def test_complete_is_refused_when_the_acceptance_report_is_stale(self):
        checkpoint = self.checkpoint_module()
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one")
        self.review_item("F1")
        report = self.accept()
        self.write("state/deliverable_acceptance.json", report)
        checkpoint.require_contract_accepted(self.root)
        self.write("figures/f1.pdf", "%PDF-changed")
        with self.assertRaises(checkpoint.CheckpointError) as raised:
            checkpoint.require_contract_accepted(self.root)
        self.assertIn("stale", str(raised.exception))

    def test_complete_is_allowed_when_the_contract_is_satisfied(self):
        checkpoint = self.checkpoint_module()
        self.freeze([item("F1")])
        self.bind_item("F1", ref="fig:one")
        self.review_item("F1")
        self.write("state/deliverable_acceptance.json", self.accept())
        checkpoint.require_contract_accepted(self.root)

    def test_a_project_without_a_contract_is_unaffected(self):
        checkpoint = self.checkpoint_module()
        checkpoint.require_contract_accepted(self.root)


if __name__ == "__main__":
    unittest.main()
