import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "lib" / "science.py"
spec = importlib.util.spec_from_file_location("paper_science", MODULE)
science = importlib.util.module_from_spec(spec)
spec.loader.exec_module(science)


class ScienceChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.design = {
            "experiment_id": "E1", "claim_id": "C1", "hypothesis": "Lower descriptive case error",
            "independent_variable": "method", "dependent_variable": "error",
            "control_variables": ["same evaluator", "same inputs"],
            "metric": {"name": "error", "unit": "m", "direction": "lower"},
            "statistical_unit": "case", "cases": ["a", "b", "c"], "seeds": [0, 1],
            "conditions": ["baseline", "candidate"], "failure_policy": "penalty", "failure_value": 10,
            "statistics": {"estimator": "mean", "interval": "sd", "paired": True},
            "data_roles": [{"id": "dev", "role": "development", "used_for_selection": True}],
            "planned_at": "2026-11-03T00:00:00Z",
        }
        self.put("experiment/design.json", self.design)
        self.design_hash = self.hash("experiment/design.json")
        self.results = {"design": "experiment/design.json", "design_sha256": self.design_hash,
                        "started_at": "2026-11-03T01:00:00Z", "outcome": "inconclusive",
                        "reported_denominator": 12, "runs": []}
        values = {"baseline": {"a": [2, 4], "b": [4, 6], "c": [6, 8]},
                  "candidate": {"a": [1, 3], "b": [2, 4], "c": [7, 9]}}
        for condition in self.design["conditions"]:
            for case in self.design["cases"]:
                for seed in self.design["seeds"]:
                    row = {"id": f"{case}-{condition}-{seed}", "case_id": case, "seed": seed,
                           "condition": condition, "status": "completed", "value": values[condition][case][seed],
                           "raw": f"experiment/raw/{case}-{condition}-{seed}.json"}
                    self.put(row["raw"], {key: row[key] for key in ("case_id", "seed", "condition", "status", "value")})
                    self.results["runs"].append(row)
        self.put("experiment/results.json", self.results)
        self.put("experiment/summary.md", "A descriptive paired case comparison with one adverse case.\n")
        self.plan = {"experiment_design_sha256": self.design_hash}

    def put(self, relative, value):
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value, indent=2) if not isinstance(value, str) else value)

    def hash(self, relative):
        return hashlib.sha256((self.root / relative).read_bytes()).hexdigest()

    def paths(self):
        return [path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()]

    def experiment_packet(self):
        return {"design": "experiment/design.json", "results": "experiment/results.json",
                "summary": "experiment/summary.md", "artifacts": self.paths(), "_plan": self.plan}

    def figure_packet(self, suffix="svg"):
        rendered = science.render_statistics(self.root, "experiment/results.json", "figures/results.tex",
                                             "figures/results." + suffix)
        packet = {"design": "experiment/design.json", "results": "experiment/results.json",
                  "statistics": rendered["statistics_path"], "table": rendered["table"],
                  "figure": rendered["figure"], "trace": rendered["trace"], "caption": rendered["caption"],
                  "_plan": self.plan, "artifacts": self.paths()}
        return packet, rendered

    def writing_packet(self):
        figures, rendered = self.figure_packet()
        literature, _ = self.catalog_packet()
        assertion = {"id": "A1", "claim_id": "C1", "text": "The candidate mean error was 4.333333 m.",
                     "statistics": rendered["statistics_path"], "measure": "candidate.mean",
                     "value": 13 / 3, "unit": "m", "denominator": 3,
                     "condition": "all-planned-cases", "strength": "descriptive", "kind": "numeric"}
        document = {"assertions": [assertion],
                    "limitations": [{"kind": "scope", "text": "These results describe the planned cases only."},
                                    {"kind": "small_sample", "text": "Three independent cases limit generalization."},
                                    {"kind": "adverse", "text": "The candidate regressed on case c."}],
                    "prose_review": {"problem": "Measured case error", "choice": "Paired methods",
                                     "finding": "Lower mean, one adverse case", "scope": "Three cases"},
                    "revisions": []}
        self.put("paper/assertions.json", document)
        self.put("paper/main.tex", assertion["text"] + "\n" + "\n".join(item["text"] for item in document["limitations"]))
        self.put("paper/references.bib", "@misc{fixture, title={Synthetic software fixture}}")
        packet = {"manuscript": "paper/main.tex", "assertions": "paper/assertions.json",
                  "bibliography": literature["bibliography"], "artifacts": self.paths(),
                  "section_packets": [{"id": "S1", "path": "paper/main.tex", "purpose": "Report the actual comparison",
                                       "claim_ids": ["C1"], "citation_keys": [], "evidence": [rendered["statistics_path"]],
                                       "baseline": None, "line_limit_exception": None}],
                  "_plan": self.plan, "_history": {"figures": figures, "literature": literature}}
        return packet, document

    def test_real_records_are_paired_at_case_level(self):
        self.assertEqual([], science.validate("experiment", self.experiment_packet(), self.root))
        packet, result = self.figure_packet()
        stats = result["statistics"]
        self.assertEqual(3, stats["n_cases"])
        self.assertEqual(2, stats["seeds_per_case"])
        self.assertEqual([3, 5, 7], stats["case_values"]["baseline"])
        self.assertAlmostEqual(2.0, stats["conditions"]["baseline"]["sd"])
        self.assertAlmostEqual(13 / 3, stats["conditions"]["candidate"]["mean"])
        self.assertAlmostEqual(-2 / 3, stats["delta"]["mean"])
        self.assertEqual(["c"], stats["adverse_case_ids"])
        self.assertEqual([], science.validate("figures", packet, self.root))

    def test_svg_and_tex_contain_data_and_uncertainty_labels(self):
        packet, rendered = self.figure_packet()
        self.assertIn("5.000000", (self.root / packet["table"]).read_text())
        self.assertIn("SD", rendered["caption"])
        self.assertIn("Denominator=12", rendered["caption"])
        self.assertIn("paired cases", rendered["caption"])
        self.assertNotIn("independent", rendered["caption"])
        self.assertIn("sample SD (ddof=1)", rendered["caption"])
        self.assertIn("statistics sha256", (self.root / packet["figure"]).read_text())
        self.assertEqual(rendered, science.render_statistics(self.root, "experiment/results.json",
                                                            "figures/results.tex", "figures/results.svg"))

    def test_pdf_is_recomputed_and_deterministic_when_matplotlib_is_available(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("PDF requires optional Matplotlib")
        packet, _ = self.figure_packet("pdf")
        self.assertTrue((self.root / packet["figure"]).read_bytes().startswith(b"%PDF-"))
        self.assertEqual([], science.validate("figures", packet, self.root))
        damaged = (self.root / packet["figure"]).read_bytes() + b"tampered chart"
        (self.root / packet["figure"]).write_bytes(damaged)
        trace = science.strict_json(self.root, packet["trace"])
        trace["figure_sha256"] = self.hash(packet["figure"])
        self.put(packet["trace"], trace)
        self.assertIn("figure differs", science.validate("figures", packet, self.root)[0])

    def test_frozen_design_cannot_change_after_results(self):
        self.design["failure_value"] = 99
        self.put("experiment/design.json", self.design)
        self.results["design_sha256"] = self.hash("experiment/design.json")
        self.put("experiment/results.json", self.results)
        self.assertIn("frozen plan", science.validate("experiment", self.experiment_packet(), self.root)[0])

    def test_raw_record_must_match_reported_measurement(self):
        self.results["runs"][0]["value"] = 999
        self.put("experiment/results.json", self.results)
        self.assertIn("raw measurement differs", science.validate("experiment", self.experiment_packet(), self.root)[0])

    def test_missing_run_does_not_become_success_only_subset(self):
        self.results["runs"].pop()
        self.put("experiment/results.json", self.results)
        self.assertIn("all planned runs", science.validate("experiment", self.experiment_packet(), self.root)[0])

    def test_failed_and_stopped_runs_count_with_penalty_and_summary(self):
        for row, status in zip(self.results["runs"][:2], ("failed", "stopped")):
            row.update(status=status, value=None, reason="controlled fixture measurement failure", failure_kind="measurement")
            self.put(row["raw"], {key: row[key] for key in ("case_id", "seed", "condition", "status", "value", "reason", "failure_kind")})
        self.put("experiment/results.json", self.results)
        self.assertIn("summary omits", science.validate("experiment", self.experiment_packet(), self.root)[0])
        self.put("experiment/summary.md", "Known attempts: a-baseline-0 failed; a-baseline-1 stopped. Neither is discarded.")
        self.assertEqual([], science.validate("experiment", self.experiment_packet(), self.root))
        _, result = self.figure_packet()
        self.assertEqual(12, result["statistics"]["planned_runs"])
        self.assertEqual(1, result["statistics"]["failed_runs"])
        self.assertEqual(1, result["statistics"]["stopped_runs"])
        self.assertEqual(10, result["statistics"]["case_values"]["baseline"][0])

    def test_uncertain_external_outcome_requires_reconciliation(self):
        self.results["runs"][0].update(status="unknown", value=None, reason="receipt missing")
        self.put("experiment/results.json", self.results)
        self.assertIn("requires reconciliation", science.validate("experiment", self.experiment_packet(), self.root)[0])

    def test_operation_failure_cannot_become_scientific_negative(self):
        row = self.results["runs"][0]
        row.update(status="failed", value=None, reason="process did not launch", failure_kind="operation")
        self.put(row["raw"], row)
        self.put("experiment/results.json", self.results)
        self.assertIn("cannot become scientific negative", science.validate("experiment", self.experiment_packet(), self.root)[0])

    def test_wrong_denominator_is_blocked(self):
        self.results["reported_denominator"] = 11
        self.put("experiment/results.json", self.results)
        self.assertIn("denominator mismatch", science.validate("experiment", self.experiment_packet(), self.root)[0])

    def test_holdout_selection_and_seed_only_pairing_are_blocked(self):
        self.design["data_roles"][0] = {"id": "hidden", "role": "holdout", "used_for_selection": True}
        self.put("experiment/design.json", self.design)
        self.assertIn("cannot be used", science.validate("experiment", self.experiment_packet(), self.root)[0])
        self.design["data_roles"][0]["used_for_selection"] = False
        self.design["statistics"]["paired"] = False
        self.put("experiment/design.json", self.design)
        self.assertIn("paired case comparison", science.validate("experiment", self.experiment_packet(), self.root)[0])

    def test_copied_seed_names_do_not_make_other_cases_paired(self):
        self.results["runs"][-1]["case_id"] = "another-case"
        self.put("experiment/results.json", self.results)
        self.assertIn("unplanned run identity", science.validate("experiment", self.experiment_packet(), self.root)[0])

    def test_declared_uncertainty_method_is_not_silently_changed(self):
        self.design["statistics"]["interval"] = "aesthetic-error"
        self.put("experiment/design.json", self.design)
        self.assertIn("unsupported interval", science.validate("experiment", self.experiment_packet(), self.root)[0])

    def test_unknown_estimator_does_not_fall_back_to_mean(self):
        self.design["statistics"]["estimator"] = "best-success"
        self.put("experiment/design.json", self.design)
        self.assertIn("unsupported estimator", science.validate("experiment", self.experiment_packet(), self.root)[0])

    def test_plotting_values_caption_and_image_are_all_checked(self):
        packet, rendered = self.figure_packet()
        original = copy.deepcopy(rendered["statistics"])
        changed = copy.deepcopy(original)
        changed["conditions"]["candidate"]["mean"] = 1
        self.put(packet["statistics"], changed)
        self.assertIn("statistics differ", science.validate("figures", packet, self.root)[0])
        self.put(packet["statistics"], original)
        packet["caption"] = "Error bars are CI; only successful samples count."
        self.assertIn("caption changes", science.validate("figures", packet, self.root)[0])
        packet["caption"] = rendered["caption"]
        (self.root / packet["table"]).write_text("candidate & 1.000000")
        self.assertIn("table values differ", science.validate("figures", packet, self.root)[0])

    def test_evaluator_bytes_are_frozen(self):
        self.put("experiment/evaluator.txt", "metric=error")
        self.design["evaluator"] = {"path": "experiment/evaluator.txt", "sha256": self.hash("experiment/evaluator.txt")}
        self.put("experiment/design.json", self.design)
        self.design_hash = self.hash("experiment/design.json")
        self.plan["experiment_design_sha256"] = self.design_hash
        self.results["design_sha256"] = self.design_hash
        self.put("experiment/results.json", self.results)
        self.assertEqual([], science.validate("experiment", self.experiment_packet(), self.root))
        self.put("experiment/evaluator.txt", "metric=always-win")
        self.assertIn("evaluator bytes changed", science.validate("experiment", self.experiment_packet(), self.root)[0])

    def test_artifact_references_are_closed(self):
        packet = self.experiment_packet()
        packet["artifacts"].remove(self.results["runs"][0]["raw"])
        self.assertIn("unbound science evidence", science.validate("experiment", packet, self.root)[0])

    def test_symlink_cannot_escape_run_root(self):
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        (Path(outside.name) / "secret.json").write_text("{}")
        (self.root / "escape").symlink_to(outside.name, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            science.strict_json(self.root, "escape/secret.json")

    def test_rewriting_output_cannot_overwrite_raw_or_existing_different_artifact(self):
        with self.assertRaisesRegex(ValueError, "raw evidence"):
            science.render_statistics(self.root, "experiment/results.json", self.results["runs"][0]["raw"], "f.svg")
        self.put("occupied.tex", "unrelated user file")
        with self.assertRaisesRegex(ValueError, "different content"):
            science.render_statistics(self.root, "experiment/results.json", "occupied.tex", "f.svg")
        self.assertEqual("unrelated user file", (self.root / "occupied.tex").read_text())

    def test_initial_draft_with_evidence_bound_numbers_passes(self):
        packet, _ = self.writing_packet()
        self.assertEqual([], science.validate("write", packet, self.root))

    def test_causal_and_significance_assertions_are_not_approved_from_descriptive_data(self):
        packet, document = self.writing_packet()
        document["assertions"][0]["kind"] = "causal"
        self.put("paper/assertions.json", document)
        self.assertIn("causal interpretation", science.validate("write", packet, self.root)[0])
        document["assertions"][0].update(kind="numeric", strength="significant")
        self.put("paper/assertions.json", document)
        self.assertIn("conclusion strength", science.validate("write", packet, self.root)[0])

    def test_adverse_cases_cannot_disappear_from_manuscript(self):
        packet, document = self.writing_packet()
        document["limitations"] = [item for item in document["limitations"] if item["kind"] != "adverse"]
        self.put("paper/assertions.json", document)
        self.assertIn("adverse paired cases", science.validate("write", packet, self.root)[0])

    def test_writing_numbers_units_denominator_and_condition_are_bound(self):
        for key, wrong in (("value", 1.0), ("unit", "cm"), ("denominator", 6), ("condition", "successful-cases")):
            with self.subTest(key=key):
                # Build separate fixtures because the statistics renderer protects existing outputs.
                packet, document = self.writing_packet()
                document["assertions"][0][key] = wrong
                self.put("paper/assertions.json", document)
                self.assertTrue(science.validate("write", packet, self.root))

    def test_prose_must_contain_bound_number_not_only_a_correct_sidecar(self):
        packet, document = self.writing_packet()
        document["assertions"][0]["text"] = "The candidate mean error was 1.000000 m."
        self.put("paper/main.tex", (self.root / "paper/main.tex").read_text().replace("4.333333", "1.000000"))
        self.put("paper/assertions.json", document)
        self.assertIn("bound numerical value", science.validate("write", packet, self.root)[0])

    def test_revision_fidelity_checks_relationships(self):
        packet, document = self.writing_packet()
        facts = {"identity": "C1", "metric": "error", "data": ["a", "b", "c"], "baseline": "baseline",
                 "denominator": 3, "unit": "m", "interval": "sd", "strength": "descriptive"}
        revision = {"before": "An earlier sentence", "after": document["assertions"][0]["text"],
                    "reason": "Put the measured comparison first", "facts_before": facts,
                    "facts_after": {**facts, "denominator": 6}}
        document["revisions"] = [revision]
        self.put("paper/assertions.json", document)
        self.assertIn("factual relationships", science.validate("write", packet, self.root)[0])
        revision["facts_after"] = facts
        self.put("paper/assertions.json", document)
        self.assertEqual([], science.validate("write", packet, self.root))

    def catalog_packet(self):
        self.put("literature/source.txt", "Synthetic software source: a paired case example.")
        self.put("literature/metadata.txt", "Synthetic software source; Offline Fixture; 2026; Software fixture; fixture:S1. This is not a real publication.")
        catalog = {"question_formed_at": "2026-11-03T00:00:00Z",
                   "searches": [{"query": "paired case error", "method": "keyword", "result_ids": ["S1", "S2"]}],
                   "naive_baseline_ids": ["S1"], "hard_negative_ids": ["S2"],
                   "candidates": [
                       {"id": "S1", "title": "Synthetic software source", "url": "https://example.org/fixture",
                        "published_at": "2026-10-01T00:00:00Z", "exposure": "prospective",
                        "purpose": "Compare the independent statistical unit", "decision": "include",
                        "read_status": "read", "read_path": "literature/source.txt",
                        "verification": {"status": "primary_checked", "evidence": "literature/metadata.txt"},
                        "evidence_spans": [{"path": "literature/source.txt", "quote": "a paired case example"}]},
                       {"id": "S2", "title": "Synthetic hard negative", "url": "https://example.org/negative",
                        "published_at": "2026-10-01T00:00:00Z", "exposure": "prospective",
                        "purpose": "Topical hit with wrong statistical unit", "decision": "exclude",
                        "read_status": "registered", "exclusion_reason": "Incompatible unit"}]}
        self.put("literature/catalog.json", catalog)
        identity = {"title": "Synthetic software source", "authors": ["Offline Fixture"], "year": 2026,
                    "venue": "Software fixture", "identifier": "fixture:S1"}
        claim = {"id": "LC1", "text": "The source describes a paired case example.",
                 "evidence_spans": [{"path": "literature/source.txt", "quote": "a paired case example"}]}
        record = {"key": "fixture", "candidate_id": "S1", **identity, "url": catalog["candidates"][0]["url"],
                  "taxonomy_cell": "paired-case", "depth": "B", "status": "verified", "claims": [claim]}
        self.put("literature/citation_plan.jsonl", json.dumps(record) + "\n")
        verification = {"records": [{"key": "fixture", "status": "verified", "checked_at": "2026-11-03T02:00:00Z",
                                     "identity": identity, "source_sha256": self.hash("literature/source.txt"),
                                     "evidence": ["literature/metadata.txt"],
                                     "claim_checks": [{"claim_id": "LC1", "status": "supported", "evidence_spans": claim["evidence_spans"]}]}],
                        "batches": []}
        self.put("literature/citation_verification.json", verification)
        self.put("literature/references.bib", "@misc{fixture, title={Synthetic software source}, author={Offline Fixture}, year={2026}, howpublished={Software fixture}}")
        return {"catalog": "literature/catalog.json", "artifacts": self.paths(),
                "bibliography": "literature/references.bib", "citation_plan": "literature/citation_plan.jsonl",
                "citation_verification": "literature/citation_verification.json"}, catalog

    def test_literature_preserves_purpose_baseline_and_hard_negative(self):
        packet, _ = self.catalog_packet()
        self.assertEqual([], science.validate("literature", packet, self.root))

    def test_literature_unknown_spans_and_prospective_postdated_sources_are_blocked(self):
        packet, catalog = self.catalog_packet()
        catalog["candidates"][0]["evidence_spans"][0]["quote"] = "invented quote"
        self.put("literature/catalog.json", catalog)
        self.assertIn("span does not match", science.validate("literature", packet, self.root)[0])
        catalog["candidates"][0]["evidence_spans"][0]["quote"] = "a paired case example"
        catalog["candidates"][0]["published_at"] = "2026-11-04T00:00:00Z"
        self.put("literature/catalog.json", catalog)
        self.assertIn("prospective inspiration", science.validate("literature", packet, self.root)[0])
        catalog["candidates"][0]["exposure"] = "retrospective"
        self.put("literature/catalog.json", catalog)
        self.assertEqual([], science.validate("literature", packet, self.root))

    def test_registered_source_is_not_read_evidence(self):
        packet, catalog = self.catalog_packet()
        catalog["candidates"][0]["read_status"] = "registered"
        self.put("literature/catalog.json", catalog)
        self.assertIn("registered source", science.validate("literature", packet, self.root)[0])

    def test_json_duplicate_keys_and_nonfinite_values_fail(self):
        self.put("duplicate.json", '{"claim": 1, "claim": 2}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            science.strict_json(self.root, "duplicate.json")
        self.put("nonfinite.json", '{"measurement": NaN}')
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            science.strict_json(self.root, "nonfinite.json")


    def citation_record(self):
        return json.loads((self.root / "literature/citation_plan.jsonl").read_text())

    def test_citation_plan_and_verification_are_required(self):
        packet, _ = self.catalog_packet()
        packet.pop("citation_plan")
        self.assertIn("evidence path", science.validate("literature", packet, self.root)[0])

    def test_bibliographic_identity_and_claim_binding_are_checked(self):
        packet, _ = self.catalog_packet()
        self.put(packet["bibliography"], "@misc{fixture,title={Different source},author={Offline Fixture},year={2026}}")
        self.assertIn("BibTeX title identity", science.validate("literature", packet, self.root)[0])
        packet, _ = self.catalog_packet()
        verification = science.strict_json(self.root, packet["citation_verification"])
        verification["records"][0]["claim_checks"][0]["claim_id"] = "another-claim"
        self.put(packet["citation_verification"], verification)
        self.assertIn("claim binding", science.validate("literature", packet, self.root)[0])

    def test_source_verification_expires_when_source_bytes_change(self):
        packet, _ = self.catalog_packet()
        self.put("literature/source.txt", (self.root / "literature/source.txt").read_text() + " Edited source version.")
        self.assertIn("stale after source change", science.validate("literature", packet, self.root)[0])

    def test_preprint_cannot_be_upgraded_to_conference_without_official_evidence(self):
        packet, _ = self.catalog_packet()
        self.put(packet["bibliography"], "@inproceedings{fixture,title={Synthetic software source},author={Offline Fixture},year={2026},booktitle={Software fixture}}")
        self.assertIn("acceptance evidence", science.validate("literature", packet, self.root)[0])
        record = self.citation_record()
        record["acceptance"] = {"status": "accepted", "authority": "arxiv_self_report", "evidence": "literature/metadata.txt", "version": "camera-ready"}
        self.put(packet["citation_plan"], json.dumps(record) + "\n")
        self.assertIn("official proceedings", science.validate("literature", packet, self.root)[0])
        record["acceptance"]["authority"] = "official_proceedings"
        self.put(packet["citation_plan"], json.dumps(record) + "\n")
        self.assertEqual([], science.validate("literature", packet, self.root))

    def test_final_writing_rejects_unresolved_citation_even_below_twenty(self):
        packet, _ = self.writing_packet()
        record = self.citation_record()
        record["status"] = "read"
        self.put("literature/citation_plan.jsonl", json.dumps(record) + "\n")
        verification = science.strict_json(self.root, "literature/citation_verification.json")
        verification["records"][0]["status"] = "unresolved"
        self.put("literature/citation_verification.json", verification)
        self.assertIn("unresolved citation identity", science.validate("write", packet, self.root)[0])

    def test_lqs_unknown_inputs_are_not_invented_and_arithmetic_is_checked(self):
        packet, _ = self.catalog_packet()
        self.put("literature/lqs.md", "Software-test score convention; no inferred reputation or acceptance.")
        packet["artifacts"] = self.paths()
        record = self.citation_record()
        record["lqs"] = {"dimensions": {"recency": 8, "impact": None, "venue": 6, "institution": None, "acceptance": 3},
                         "total": 7, "convention": "literature/lqs.md"}
        self.put(packet["citation_plan"], json.dumps(record) + "\n")
        self.assertIn("fabricated total", science.validate("literature", packet, self.root)[0])
        record["lqs"]["total"] = None
        self.put(packet["citation_plan"], json.dumps(record) + "\n")
        self.assertEqual([], science.validate("literature", packet, self.root))
        record["lqs"]["dimensions"].update(impact=4, institution=5)
        record["lqs"]["total"] = 0
        self.put(packet["citation_plan"], json.dumps(record) + "\n")
        self.assertIn("LQS arithmetic", science.validate("literature", packet, self.root)[0])
        record["lqs"]["total"] = 8*.30 + 4*.25 + 6*.20 + 5*.10 + 3*.15
        self.put(packet["citation_plan"], json.dumps(record) + "\n")
        self.assertEqual([], science.validate("literature", packet, self.root))

    def many_citations(self, count):
        packet, catalog = self.catalog_packet()
        source = catalog["candidates"][0]
        original = self.citation_record()
        check = science.strict_json(self.root, packet["citation_verification"])["records"][0]
        records, checks, bibliography, identity_evidence = [], [], [], []
        for index in range(count):
            key, candidate_id = f"key{index:02}", f"source{index:02}"
            candidate = copy.deepcopy(source)
            candidate.update(id=candidate_id, title=f"Synthetic source {index}", url=f"https://example.invalid/{index}")
            catalog["candidates"].append(candidate)
            record = copy.deepcopy(original)
            record.update(key=key, candidate_id=candidate_id, title=candidate["title"], url=candidate["url"], identifier=f"fixture:{index}")
            record["claims"][0]["id"] = "LC" + str(index)
            records.append(record)
            checked = copy.deepcopy(check)
            checked.update(key=key, identity={name: record[name] for name in science.IDENTITY_FIELDS})
            checked["claim_checks"][0]["claim_id"] = record["claims"][0]["id"]
            checks.append(checked)
            bibliography.append(f"@misc{{{key},title={{{record['title']}}},author={{Offline Fixture}},year={{2026}},howpublished={{Software fixture}}}}")
            identity_evidence.append("; ".join(str(value) for value in [record["title"], "Offline Fixture", 2026, record["venue"], record["identifier"]]))
        self.put(packet["catalog"], catalog)
        self.put(packet["citation_plan"], "\n".join(json.dumps(record) for record in records) + "\n")
        self.put(packet["bibliography"], "\n".join(bibliography))
        self.put("literature/metadata.txt", "\n".join(identity_evidence))
        verification = {"records": checks, "batches": []}
        self.put(packet["citation_verification"], verification)
        return packet, records, verification

    def test_each_twenty_citations_needs_ordered_batch_record(self):
        packet, records, verification = self.many_citations(20)
        self.assertIn("twenty accumulated citations", science.validate("literature", packet, self.root)[0])
        self.put("literature/batch.txt", "Checked twenty synthetic source identities and bound claims.")
        packet["artifacts"] = self.paths()
        verification["batches"] = [{"keys": [record["key"] for record in records], "checked_at": "2026-11-03T03:00:00Z",
                                    "evidence": "literature/batch.txt"}]
        self.put(packet["citation_verification"], verification)
        self.assertEqual([], science.validate("literature", packet, self.root))
        verification["batches"][0]["keys"].reverse()
        self.put(packet["citation_verification"], verification)
        self.assertIn("skips or reorders", science.validate("literature", packet, self.root)[0])

    def test_final_tail_citation_requires_individual_identity_and_claim_verification(self):
        write, _ = self.writing_packet()
        literature, records, verification = self.many_citations(21)
        self.put("literature/batch.txt", "The first twenty synthetic citation identities and claims were checked.")
        verification["batches"] = [{"keys": [record["key"] for record in records[:20]],
                                    "checked_at": "2026-11-03T03:00:00Z", "evidence": "literature/batch.txt"}]
        self.put(literature["citation_verification"], verification)
        literature["artifacts"] = self.paths()
        write["artifacts"] = self.paths()
        write["_history"]["literature"] = literature
        # No second batch receipt is needed when every tail record was actually checked.
        self.assertEqual([], science.validate("write", write, self.root))
        records[-1]["status"] = "read"
        verification["records"][-1]["status"] = "unresolved"
        self.put(literature["citation_plan"], "\n".join(json.dumps(record) for record in records) + "\n")
        self.put(literature["citation_verification"], verification)
        self.assertEqual([], science.validate("literature", literature, self.root))
        self.assertIn("unresolved citation identity: key20", science.validate("write", write, self.root)[0])
        records[-1]["status"] = "verified"
        verification["records"][-1]["status"] = "verified"
        verification["records"][-1]["claim_checks"][0]["status"] = "unresolved"
        self.put(literature["citation_plan"], "\n".join(json.dumps(record) for record in records) + "\n")
        self.put(literature["citation_verification"], verification)
        self.assertIn("unsupported/unresolved citation claims", science.validate("write", write, self.root)[0])

    def test_generated_tex_limit_and_existing_large_source_exception(self):
        packet, _ = self.writing_packet()
        original = (self.root / "paper/main.tex").read_text()
        large = original + "\n" + "\n".join("% preserved source " + str(i) for i in range(305))
        self.put("paper/main.tex", large)
        self.assertIn("exceeds 300 lines", science.validate("write", packet, self.root)[0])
        self.put("revisions/main-baseline.tex", large)
        baseline = {"path": "revisions/main-baseline.tex", "sha256": self.hash("revisions/main-baseline.tex")}
        packet["section_packets"][0].update(baseline=baseline,
            line_limit_exception={"reason": "Preserve this existing large source without unrelated splitting", "baseline_sha256": baseline["sha256"]})
        packet["artifacts"] = self.paths()
        self.assertEqual([], science.validate("write", packet, self.root))
        self.put("revisions/main-baseline.tex", original)
        self.assertIn("baseline changed", science.validate("write", packet, self.root)[0])

    def test_included_sections_need_unique_ownership_and_claim_location(self):
        packet, _ = self.writing_packet()
        self.put("paper/section.tex", "Additional background material.")
        self.put("paper/main.tex", (self.root / "paper/main.tex").read_text() + "\n\\input{section}\n")
        packet["artifacts"] = self.paths()
        self.assertIn("lack section packets", science.validate("write", packet, self.root)[0])
        self.put("paper/main.tex", (self.root / "paper/main.tex").read_text().replace(r"\input{section}", r"\input section.tex"))
        self.assertIn("lack section packets", science.validate("write", packet, self.root)[0])
        extra = {"id": "S2", "path": "paper/section.tex", "purpose": "Provide background", "claim_ids": ["C1"],
                 "citation_keys": [], "evidence": ["literature/source.txt"], "baseline": None, "line_limit_exception": None}
        packet["section_packets"].append(extra)
        self.assertIn("not located", science.validate("write", packet, self.root)[0])
        extra["claim_ids"] = []
        self.assertEqual([], science.validate("write", packet, self.root))
        packet["section_packets"].append(copy.deepcopy(extra))
        self.assertIn("duplicate section", science.validate("write", packet, self.root)[0])

    def test_actual_citation_keys_cannot_be_unplanned_or_dropped(self):
        packet, _ = self.writing_packet()
        self.put("paper/main.tex", (self.root / "paper/main.tex").read_text() + "\n\\cite{unknown-source}\n")
        packet["section_packets"][0]["citation_keys"] = ["unknown-source"]
        self.assertIn("absent/dropped/unverified", science.validate("write", packet, self.root)[0])

    def review_packet(self):
        literature, _ = self.catalog_packet()
        statement = "The source describes a paired case example."
        document = {"assertions": [{"id": "A1", "claim_id": "LC1", "kind": "source_claim", "strength": "source_bounded",
                                    "text": statement, "citation_keys": ["fixture"]}],
                    "limitations": [{"kind": "scope", "text": "This synthesis covers only the verified source."}],
                    "prose_review": {"problem": "Classify evidence units", "choice": "Compare source methods",
                                     "finding": statement, "scope": "One verified synthetic source"}, "revisions": []}
        self.put("paper/assertions.json", document)
        self.put("paper/main.tex", statement + "\n\\cite{fixture}\n" + document["limitations"][0]["text"])
        packet = {"manuscript": "paper/main.tex", "assertions": "paper/assertions.json",
                  "bibliography": literature["bibliography"], "artifacts": self.paths(),
                  "section_packets": [{"id": "S1", "path": "paper/main.tex", "purpose": "Explain the bounded source claim",
                                       "claim_ids": ["LC1"], "citation_keys": ["fixture"], "evidence": ["literature/source.txt"],
                                       "baseline": None, "line_limit_exception": None}],
                  "_brief": {"paper_type": "survey", "not_applicable": {"experiment": "No empirical experiment in the approved synthesis"}},
                  "_history": {"literature": literature}, "_plan": {}}
        return packet, document

    def test_non_empirical_survey_has_a_source_bounded_writing_path(self):
        packet, _ = self.review_packet()
        self.assertEqual([], science.validate("write", packet, self.root))
        packet["_brief"] = {"paper_type": "original"}
        self.assertEqual([], science.validate("write", packet, self.root))

    def test_review_cannot_turn_unverified_source_into_a_general_claim(self):
        packet, document = self.review_packet()
        document["assertions"][0]["text"] = "All studies universally use the case as a unit."
        self.put("paper/assertions.json", document)
        self.put("paper/main.tex", document["assertions"][0]["text"] + "\n\\cite{fixture}\n" + document["limitations"][0]["text"])
        self.assertIn("planned claim", science.validate("write", packet, self.root)[0])

    def test_review_evidence_diagram_is_bound_to_verified_claims_and_raw_spans(self):
        write, _ = self.review_packet()
        self.put("figures/synthesis.svg", '<svg xmlns="http://www.w3.org/2000/svg"><text>Case-level synthesis</text></svg>')
        self.put("figures/synthesis.tex", "Fixture source & Case-level unit \\\\ ")
        trace = {"kind": "evidence_diagram", "figure_sha256": self.hash("figures/synthesis.svg"),
                 "table_sha256": self.hash("figures/synthesis.tex"),
                 "source_hashes": {"literature/source.txt": self.hash("literature/source.txt")},
                 "claim_ids": ["LC1"], "citation_keys": ["fixture"]}
        self.put("figures/synthesis.trace.json", trace)
        packet = {"mode": "evidence_diagram", "figure": "figures/synthesis.svg", "table": "figures/synthesis.tex",
                  "trace": "figures/synthesis.trace.json", "caption": "A bounded source-method synthesis, not numerical experiment results.",
                  "claim_ids": ["LC1"], "citation_keys": ["fixture"],
                  "evidence_spans": [{"path": "literature/source.txt", "quote": "a paired case example"}],
                  "artifacts": self.paths(), "_brief": write["_brief"], "_history": write["_history"]}
        self.assertEqual([], science.validate("figures", packet, self.root))
        self.put("figures/synthesis.svg", '<svg xmlns="http://www.w3.org/2000/svg"><text>Changed diagram</text></svg>')
        self.assertIn("stale/mismatched", science.validate("figures", packet, self.root)[0])

    def final_packet(self):
        write, _ = self.writing_packet()
        self.put("paper/main.pdf", "%PDF-1.4\nSynthetic unit test header; actual compilation is checked by the controller.\n")
        write["manuscript_pdf"] = "paper/main.pdf"
        self.put("review/report.md", "Unit-test reviewer evidence; full independent review is a controller check.")
        self.put("revision/history.json", {"status": "initial_draft", "changes": []})
        completion = {"source": ["paper/main.tex"], "pdf": ["paper/main.pdf"],
                      "bibliography": [write["bibliography"]],
                      "citation_plan": ["literature/citation_plan.jsonl"],
                      "citation_verification": ["literature/citation_verification.json"],
                      "review": ["review/report.md"], "revision": ["revision/history.json"],
                      "snapshot": ["snapshot/version.json"]}
        required = {path for role, paths in completion.items() if role != "snapshot" for path in paths}
        self.put("snapshot/version.json", {"files": {path: self.hash(path) for path in required}})
        history = {**write["_history"], "write": write,
                   "review": {"artifacts": ["review/report.md"]}}
        return {"completion": completion, "artifacts": self.paths(), "_history": history, "_plan": self.plan}

    def test_all_eight_completion_roles_and_snapshot_members_are_required(self):
        packet = self.final_packet()
        self.assertEqual([], science.validate_delivery(packet, self.root))
        packet["completion"].pop("citation_plan")
        self.assertIn("eight completion roles", science.validate_delivery(packet, self.root)[0])

    def test_snapshot_cannot_stand_in_for_its_missing_artifacts(self):
        packet = self.final_packet()
        packet["artifacts"].remove("revision/history.json")
        self.assertIn("unbound science evidence", science.validate_delivery(packet, self.root)[0])
        packet["artifacts"] = self.paths()
        self.put("review/report.md", "Edited after snapshot.")
        self.assertIn("stale artifact", science.validate_delivery(packet, self.root)[0])

    def test_snapshot_must_cover_all_non_snapshot_delivery_roles(self):
        packet = self.final_packet()
        snapshot = science.strict_json(self.root, "snapshot/version.json")
        snapshot["files"].pop("paper/main.pdf")
        self.put("snapshot/version.json", snapshot)
        self.assertIn("omits completion artifacts", science.validate_delivery(packet, self.root)[0])

    def test_initial_draft_does_not_need_fabricated_revision_history(self):
        packet = self.final_packet()
        self.put("revision/history.json", {"status": "revised", "changes": []})
        self.assertIn("revision history contradicts", science.validate_delivery(packet, self.root)[0])


    def test_year_only_sources_keep_precision_and_cannot_claim_prospective_overlap(self):
        packet, _ = self.catalog_packet()
        catalog = json.loads((self.root / 'literature/catalog.json').read_text())
        catalog['candidates'][0]['published_at'] = '2026'
        self.put('literature/catalog.json', catalog)
        self.assertIn('overlapping', science.validate('literature', packet, self.root)[0])
        catalog['candidates'][0]['exposure'] = 'retrospective'
        self.put('literature/catalog.json', catalog)
        self.assertEqual([], science.validate('literature', packet, self.root))
        self.assertEqual('2026', json.loads((self.root / 'literature/catalog.json').read_text())['candidates'][0]['published_at'])

    def test_explicit_bibtex_surname_order_matches_source_author_identity(self):
        packet, _ = self.catalog_packet()
        metadata = (self.root/'literature/metadata.txt').read_text().replace('Offline Fixture', 'Ada Van Example')
        self.put('literature/metadata.txt', metadata)
        plan = json.loads((self.root/'literature/citation_plan.jsonl').read_text())
        plan['authors'] = ['Ada Van Example']
        self.put('literature/citation_plan.jsonl', json.dumps(plan)+'\n')
        verified = json.loads((self.root/'literature/citation_verification.json').read_text())
        verified['records'][0]['identity']['authors'] = ['Ada Van Example']
        self.put('literature/citation_verification.json', verified)
        bib = (self.root/'literature/references.bib').read_text().replace('author={Offline Fixture}', 'author={Van Example, Ada}')
        self.put('literature/references.bib', bib)
        self.assertEqual([], science.validate('literature', packet, self.root))
        self.put('literature/references.bib', bib.replace('Van Example, Ada', 'Different, Ada'))
        self.assertIn('author identity mismatch', science.validate('literature', packet, self.root)[0])


if __name__ == "__main__":
    unittest.main()
