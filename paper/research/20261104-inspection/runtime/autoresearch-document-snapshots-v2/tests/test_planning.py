from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import planning, venue_tools


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def visual_plan():
    return {'items': [
        {'id': 'F1', 'kind': 'figure', 'type': 'quantitative_results',
         'question': 'Which method has the lower error?', 'claim': 'The candidate has lower measured error.',
         'data': 'Measured trials.', 'section': 'Results', 'priority': 'must'},
        {'id': 'T1', 'kind': 'table', 'type': 'main_results',
         'question': 'What are the observed trial outcomes?', 'claim': 'The table includes failures as well as successes.',
         'data': 'The complete trial ledger.', 'section': 'Results', 'priority': 'must'}]}


def make_pdf(root, name='sample', separator='.'):
    source = root / (name + '.tex')
    source.write_text(r'''\documentclass{article}
\begin{document}
Fixture study. This is software test data, not an accepted research paper.
Figure 1 and Table 1 summarize the measured fixture outcomes.

\noindent Figure 1. Candidate comparison under fixed test conditions.

\noindent Table 1. Main results including all recorded failures.

The comparison uses the full fixture ledger and is not scientific evidence.
\end{document}
'''.replace('Fixture study.', 'Fixture study ' + name + '.').replace('Figure 1. Candidate', 'Figure 1' + separator + ' Candidate').replace('Table 1. Main', 'Table 1' + separator + ' Main'))
    out = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', source.name],
                         cwd=root, capture_output=True, timeout=60)
    if out.returncode:
        raise AssertionError(out.stdout.decode(errors='replace')[-1500:])
    return root / (name + '.pdf')


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        text = 'The fixture supports a deterministic local CPU comparison.'
        (self.root / 'source.txt').write_text(text)
        self.quote = {'material_id': 'M1', 'start': 0, 'end': len(text), 'quote': text}
        self.packet = {'artifacts': ['source.txt'],
            'materials': [{'id': 'M1', 'path': 'source.txt', 'sha256': sha(self.root / 'source.txt'),
                           'availability': 'readable', 'read_spans': [[0, len(text)]]}],
            'evidence': [self.quote],
            'feasibility': {'status': 'feasible', 'selected_option': 'cpu',
                'options': [{'id': 'cpu', 'implementation': 'A deterministic standard-library fixture.',
                             'selection_reason': 'No external model or remote computer is required.', 'evidence': [self.quote]}],
                'single_option_reason': 'The software fixture has one fixed reference implementation.',
                'resources': {'authorization': 'Offline software test only.', 'cost_basis': 'No paid calls.'},
                'experiments': [{'claim': 'The fixture produces complete evidence.', 'metric': 'error in metres',
                    'comparison': 'candidate against the fixed baseline', 'protocol': 'frozen paired cases and seeds',
                    'completion': 'all declared cases have terminal records', 'failure_route': 'retain failed and unknown trials'}]}}

    def test_full_read_and_source_grounded_feasibility(self):
        self.assertEqual([], planning.validate('intake', self.packet, self.root))

    def test_registered_material_is_not_read_content(self):
        self.packet['materials'][0]['read_spans'] = [[0, 5]]
        self.assertIn('partially read', ';'.join(planning.validate('intake', self.packet, self.root)))

    def test_quote_must_match_actual_read_span(self):
        self.packet['evidence'][0] = {**self.quote, 'quote': 'invented evidence'}
        self.assertIn('quote does not match', ';'.join(planning.validate('intake', self.packet, self.root)))

    def test_metadata_only_has_limit_and_cannot_be_evidence(self):
        self.packet['materials'][0].update(availability='metadata_only', read_spans=[], limitation='Binary metadata only.')
        self.assertIn('readable material', ';'.join(planning.validate('intake', self.packet, self.root)))

    def test_omitting_experiments_needs_the_frozen_scope(self):
        self.packet['feasibility']['experiments'] = []
        self.packet['feasibility']['no_experiment_reason'] = 'Review-only comparison of published evidence.'
        self.assertIn('approved review/survey', ';'.join(planning.validate('intake', self.packet, self.root)))
        self.packet['_brief'] = {'paper_type': 'review', 'not_applicable': {'experiment': 'Review-only comparison of published evidence.'}}
        self.assertEqual([], planning.validate('intake', self.packet, self.root))

    def test_changed_material_and_unsupported_route_fail(self):
        (self.root / 'source.txt').write_text('changed')
        self.assertIn('changed', ';'.join(planning.validate('intake', self.packet, self.root)))
        self.packet['materials'][0]['sha256'] = sha(self.root / 'source.txt')
        self.packet['materials'][0]['read_spans'] = [[0, 7]]
        self.packet['evidence'] = [{'material_id': 'M1', 'start': 0, 'end': 7, 'quote': 'changed'}]
        self.packet['feasibility']['options'][0]['evidence'] = []
        self.packet['feasibility']['options'][0]['basis'] = 'assumption'
        self.assertIn('actually read evidence', ';'.join(planning.validate('intake', self.packet, self.root)))


class VenueSamplingPolicyTests(unittest.TestCase):
    def packet(self, kind='journal'):
        return {'sampling_policy': {'kind': kind, 'as_of': '2026-11-04'}}

    def records(self, count=8, year=2026):
        return [{'sample_id': 'S' + str(index), 'year': year, 'warnings': []} for index in range(count)]

    def test_eight_to_fifteen_is_the_actual_default(self):
        packet = self.packet()
        self.assertTrue(venue_tools.sampling_report(packet, self.records())['count_in_default_range'])
        for count in (7, 16):
            with self.subTest(count=count), self.assertRaisesRegex(planning.PlanningError, '8–15'):
                venue_tools.sampling_report(packet, self.records(count))
        packet['sampling_policy']['sample_count_deviation_reason'] = 'Only seven eligible same-direction accepted open papers were accessible in the recorded search scope.'
        result = venue_tools.sampling_report(packet, self.records(7))
        self.assertTrue(result['budget_is_reference_only'])
        packet['sampling_policy']['sample_count_deviation_reason'] = 'x'
        with self.assertRaisesRegex(planning.PlanningError, 'concrete reason'):
            venue_tools.sampling_report(packet, self.records(7))

    def test_old_journal_samples_need_a_recorded_reason(self):
        packet = self.packet()
        with self.assertRaisesRegex(planning.PlanningError, 'time-window'):
            venue_tools.sampling_report(packet, self.records(year=2022))
        packet['sampling_policy']['time_window_deviation_reason'] = 'The older papers are retained as explicitly labelled format comparisons because the exact subtopic has few recent accepted articles.'
        self.assertEqual(8, len(venue_tools.sampling_report(packet, self.records(year=2022))['outside_window_samples']))

    def test_exact_journal_dates_resolve_the_two_year_boundary(self):
        packet = self.packet(); records = self.records(year=2024)
        for record in records:
            record['publication_date'] = '2024-12-01'
        result = venue_tools.sampling_report(packet, records)
        self.assertEqual([], result['outside_window_samples'])
        self.assertEqual('2024-11-04', result['time_window']['start'])
        records[0]['publication_date'] = '2024-01-01'
        with self.assertRaisesRegex(planning.PlanningError, 'time-window'):
            venue_tools.sampling_report(packet, records)

    def test_conference_sampling_names_the_recent_editions(self):
        packet = self.packet('conference')
        with self.assertRaisesRegex(planning.PlanningError, 'recent completed editions'):
            venue_tools.sampling_report(packet, self.records())
        packet['sampling_policy']['recent_editions'] = [2026, 2024]
        self.assertEqual([], venue_tools.sampling_report(packet, self.records(year=2024))['outside_window_samples'])
        with self.assertRaisesRegex(planning.PlanningError, 'time-window'):
            venue_tools.sampling_report(packet, self.records(year=2025))

    def test_fixture_exception_is_explicit_and_has_no_real_survey_claim(self):
        packet = self.packet(); packet['software_fixture'] = True
        with self.assertRaisesRegex(planning.PlanningError, 'fixture needs'):
            venue_tools.sampling_report(packet, self.records(1))
        packet['sampling_policy']['fixture_exception_reason'] = 'Single synthetic PDF tests plumbing; it is not real venue sampling.'
        result = venue_tools.sampling_report(packet, self.records(1))
        self.assertTrue(result['software_fixture'])
        self.assertTrue(result['budget_is_reference_only'])

    def test_measurement_warnings_cannot_disappear(self):
        packet = self.packet(); records = self.records()
        records[0]['warnings'] = ['No References heading; total-page fallback.']
        with self.assertRaisesRegex(planning.PlanningError, 'warnings require'):
            venue_tools.sampling_report(packet, records)
        packet['sampling_policy']['measurement_warning_review'] = 'This sample has no References section, so its page count is not pooled with references-heading body pages.'
        self.assertEqual('S0', venue_tools.sampling_report(packet, records)['measurement_warnings'][0]['sample_id'])


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'template.tex').write_text(r'\documentclass[10pt,twocolumn]{article}')
        (self.root / 'main.tex').write_text(r'\documentclass[10pt, twocolumn]{article}' + '\n' + r'\usepackage[labelsep=period]{caption}')
        (self.root / 'rules.md').write_text('Use a two-column article format. No official template is available.')
        self.venue = {'template': {'status': 'available', 'path': 'template.tex', 'sha256': sha(self.root / 'template.tex')},
                      'official_rules': {'path': 'rules.md', 'sha256': sha(self.root / 'rules.md')}}
        self.packet = {'stage': 'write', 'manuscript_tex': 'main.tex', 'template_source': 'template.tex',
                       'template_sha256': self.venue['template']['sha256'], 'artifacts': ['main.tex', 'template.tex'],
                       '_history': {'venue': self.venue}}

    def test_literal_official_class_and_options_are_preserved(self):
        self.assertEqual([], planning.validate('write', self.packet, self.root))
        (self.root / 'main.tex').write_text(r'% \documentclass{wrong}' + '\n' + r'\documentclass[10pt, twocolumn]{article}')
        self.assertEqual([], planning.validate('write', self.packet, self.root))

    def test_changed_class_or_options_cannot_silently_replace_the_template(self):
        for declaration in (r'\documentclass[10pt,twocolumn]{report}', r'\documentclass[12pt]{article}'):
            with self.subTest(declaration=declaration):
                (self.root / 'main.tex').write_text(declaration)
                self.assertIn('documentclass/options differ', ';'.join(planning.validate('write', self.packet, self.root)))

    def test_template_version_and_required_packages_are_checked(self):
        self.packet['template_sha256'] = '0' * 64
        self.assertIn('hash differs', ';'.join(planning.validate('write', self.packet, self.root)))
        self.packet['template_sha256'] = self.venue['template']['sha256']
        self.venue['template']['required_packages'] = [{'name': 'caption', 'options': ['labelsep=period']}]
        self.assertEqual([], planning.validate('write', self.packet, self.root))
        self.venue['template']['required_packages'][0]['options'] = ['labelsep=colon']
        self.assertIn('package options', ';'.join(planning.validate('write', self.packet, self.root)))
        self.venue['template']['required_packages'] = ['graphicx']
        self.assertIn('omitted required', ';'.join(planning.validate('write', self.packet, self.root)))

    def test_no_official_template_requires_a_bound_format_decision(self):
        self.venue['template'] = {'status': 'not_available', 'reason': 'The author rules provide format requirements only.'}
        self.packet['template_source'] = 'rules.md'; self.packet['artifacts'].append('rules.md')
        self.packet['template_sha256'] = self.venue['official_rules']['sha256']
        self.assertIn('format check', ';'.join(planning.validate('write', self.packet, self.root)))
        self.packet['format_check'] = {'basis': 'official_requirement', 'reason': 'Literal format matches the frozen author rules.',
            'documentclass': 'article', 'options': ['10pt', 'twocolumn'],
            'evidence_quote': 'Use a two-column article format.'}
        self.assertEqual([], planning.validate('write', self.packet, self.root))
        self.packet['format_check']['evidence_quote'] = 'Invented rules'
        self.assertIn('quote is absent', ';'.join(planning.validate('write', self.packet, self.root)))


@unittest.skipUnless(shutil.which('pdflatex') and shutil.which('pdftotext') and shutil.which('pdfinfo'),
                     'Real PDF checks need pdflatex and Poppler')
class PlanningPdfTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        make_pdf(self.root)
        for name, text in [('rules.md', 'Fixture author rules: eight pages excluding references and appendix. This is synthetic.'),
                           ('template.tex', '\\documentclass{article}'),
                           ('accepted.md', 'Synthetic acceptance evidence, software fixture only.')]:
            (self.root / name).write_text(text)
        dump(self.root / 'visual-plan.json', visual_plan())
        self.packet = {'stage': 'venue', 'software_fixture': True, 'venue': 'Fixture venue',
            'direction': 'software verification', 'paper_type': 'original',
            'artifacts': ['sample.pdf', 'rules.md', 'template.tex', 'accepted.md', 'visual-plan.json',
                          'measurements.json', 'profile.json', 'plan-check.json'],
            'official_rules': self.evidence('rules.md', basis='official_requirement'),
            'template': self.evidence('template.tex', status='available'),
            'samples': [{'id': 'S1', 'pdf': 'sample.pdf', 'venue_label': 'Fixture venue', 'year': 2026,
                         'acceptance_evidence': self.evidence('accepted.md'),
                         'direction_match': 'Synthetic study of the same fixture question.',
                         'design': {'figures': [{'type': 'quantitative_results', 'placement': 'body'}],
                                    'tables': [{'type': 'main_results', 'placement': 'body'}]}}],
            'sampling_limitations': 'One synthetic PDF exercises measurement, not a real venue preference.',
            'figure_table_plan': 'visual-plan.json', 'measurements': 'measurements.json',
            'venue_profile': 'profile.json', 'venue_plan_check': 'plan-check.json'}
        self.packet['venue_comparison'] = 'venue-comparison.json'
        self.packet['artifacts'].append('venue-comparison.json')
        self.packet['_brief'] = {key: self.packet[key] for key in ('venue', 'direction', 'paper_type')}
        source = self.evidence('rules.md', quote='eight pages excluding references and appendix')
        self.packet['venue_selection'] = {'mode': 'existing_target', 'selected_id': 'V1',
            'criteria': ['scope', 'article type', 'format and access'], 'rationale': 'The fixture target was already chosen.',
            'candidates': [{'id': 'V1', 'venue': 'Fixture venue', 'kind': 'journal',
                'scope_fit': 'Synthetic comparison of the same software verification question.',
                'article_type_fit': 'A synthetic original article.', 'constraints': 'No real submission or external research.',
                'cost': {'status': 'unknown', 'basis': 'No real publication fee was investigated.'},
                'timing': {'status': 'unknown', 'basis': 'No real editorial schedule was investigated.'},
                'advantages': ['Exercises the chosen-target path.'], 'limitations': ['Not a real journal.'],
                'evidence': [source]}]}
        self.packet['sampling_policy'] = {'kind': 'journal', 'as_of': '2026-11-04',
            'fixture_exception_reason': 'One synthetic PDF exercises measurement only; not a venue estimate.',
            'measurement_warning_review': 'The fixture lacks References; total-page fallback is retained explicitly.'}
        self.packet['page_budget'] = {'target': 8, 'basis': 'official_requirement', 'limit_type': 'hard_limit',
            'includes_references': False, 'includes_appendix': False, 'source': source,
            'reason': 'Synthetic source text supplies this format rule; not a real author policy.'}
        design = venue_tools.design_template(self.packet, self.root)['papers']['S1']
        self.packet['samples'][0]['visual_read'] = {'pages': [1], 'method': 'rendered_pdf',
            'notes': 'The real compiled fixture PDF page is inspected for these test declarations.'}
        for plural, visual_type in (('figures', 'quantitative_results'), ('tables', 'main_results')):
            for row in design[plural]:
                row.update(type=visual_type, placement='body', question='Which outcomes does the fixture show?',
                    claim='The full fixture outcomes are reported.', evidence='Synthetic trial descriptions in the PDF.',
                    design_takeaway='Name the comparison conditions in the caption.',
                    limitations='This is a software fixture and carries no real scientific result.')
        self.packet['samples'][0]['design'] = {key: design[key] for key in ('figures', 'tables')}
        self.recompute()
        self.plan_packet = {'stage': 'plan', 'figure_table_plan': 'visual-plan.json',
            'plan_snapshot': 'snapshot.json', 'artifacts': ['visual-plan.json', 'snapshot.json'],
            '_history': {'venue': self.packet},
            '_brief': {'deliverable_plan': [
                {'id': item['id'], 'kind': item['kind'], 'title': item['claim'],
                 'purpose': item['question'], 'acceptance_criteria': ['Evidence-backed result'],
                 'evidence_requirements': ['Full trial ledger']} for item in visual_plan()['items']]}}
        dump(self.root / 'snapshot.json', planning.plan_snapshot(self.root / 'visual-plan.json'))

    def evidence(self, name, **extra):
        return {'path': name, 'sha256': sha(self.root / name), 'url': 'https://fixture.invalid/' + name, **extra}

    def recompute(self):
        computed = planning.calibrate(self.packet, self.root)
        for key in ('measurements', 'venue_profile', 'venue_plan_check', 'venue_comparison'):
            dump(self.root / self.packet[key], computed[key])

    def review_packet(self):
        return {'stage': 'review', 'figure_table_plan': 'visual-plan.json', 'plan_snapshot': 'snapshot.json',
            'manuscript_pdf': 'sample.pdf', 'artifacts': ['visual-plan.json', 'snapshot.json', 'sample.pdf'],
            '_history': {'plan': self.plan_packet},
            'r6_reviews': [{'id': key, 'verdict': 'fulfilled', 'reason': 'Verified the actual fixture caption and evidence.',
                            'evidence_locations': ['sample.pdf page 1']} for key in ('F1', 'T1')]}

    def test_actual_pdf_measurement_profile_budget_and_frozen_plan(self):
        self.assertEqual([], planning.validate('venue', self.packet, self.root))
        self.assertEqual([], planning.validate('plan', self.plan_packet, self.root))
        profile = json.loads((self.root / 'profile.json').read_text())
        self.assertEqual(1, profile['budget']['figures']['target'])
        self.assertEqual(1, profile['budget']['tables']['target'])
        self.assertEqual([], planning.validate('review', self.review_packet(), self.root))

    def test_standard_colon_captions_are_not_mistaken_for_missing_outputs(self):
        make_pdf(self.root, name='colon', separator=':')
        review = self.review_packet()
        review['manuscript_pdf'] = 'colon.pdf'; review['artifacts'].append('colon.pdf')
        self.assertEqual([], planning.validate('review', review, self.root))

    def test_fake_measurement_or_stale_official_template_fails(self):
        measurements = json.loads((self.root / 'measurements.json').read_text())
        measurements[0]['counts']['figures'] = 42
        dump(self.root / 'measurements.json', measurements)
        self.assertIn('recomputed', ';'.join(planning.validate('venue', self.packet, self.root)))
        self.recompute()
        (self.root / 'template.tex').write_text('changed template')
        self.assertIn('stale', ';'.join(planning.validate('venue', self.packet, self.root)))

    def test_missing_acceptance_evidence_or_direction_fails(self):
        self.packet['samples'][0]['acceptance_evidence']['sha256'] = '0' * 64
        self.assertIn('acceptance evidence', ';'.join(planning.validate('venue', self.packet, self.root)))
        self.packet['samples'][0]['acceptance_evidence'] = self.evidence('accepted.md')
        self.packet['samples'][0]['direction_match'] = ''
        self.assertIn('direction-match', ';'.join(planning.validate('venue', self.packet, self.root)))

    def test_undecided_journal_comparison_needs_actual_current_confirmation(self):
        selection = self.packet['venue_selection']
        selection['mode'] = 'candidate_comparison'
        selection['candidates'].append({**deepcopy(selection['candidates'][0]), 'id': 'V2', 'venue': 'Alternative synthetic venue',
                                         'advantages': ['Alternative source-based format comparison.']})
        self.packet['_brief']['venue'] = 'undecided'
        self.recompute()
        self.assertIn('actual controller-recorded', ';'.join(planning.validate('venue', self.packet, self.root)))
        report = venue_tools.comparison_report(self.packet, self.root)
        self.packet['_venue_confirmation'] = {'candidate_id': 'V1', 'venue': self.packet['venue'],
            'comparison_sha256': venue_tools.comparison_digest(report)}
        self.assertEqual([], planning.validate('venue', self.packet, self.root))
        self.packet['venue_selection']['rationale'] += ' Changed rationale.'
        self.recompute()
        self.assertIn('confirmation is stale', ';'.join(planning.validate('venue', self.packet, self.root)))

    def test_existing_target_does_not_allow_a_silent_venue_change(self):
        self.packet['_brief']['venue'] = 'Different target already approved'
        self.assertIn('exact previously chosen', ';'.join(planning.validate('venue', self.packet, self.root)))

    def test_design_template_is_measured_but_semantically_incomplete(self):
        template = venue_tools.design_template(self.packet, self.root)
        row = template['papers']['S1']['figures'][0]
        self.assertEqual(1, row['number'])
        self.assertIn('Figure 1.', row['caption'])
        self.assertEqual('', row['claim'])
        self.packet['samples'][0]['design']['figures'][0]['claim'] = ''
        with self.assertRaisesRegex(planning.PlanningError, 'missing claim'):
            planning.calibrate(self.packet, self.root)

    def test_per_visual_correspondence_cannot_be_replaced_by_equal_counts(self):
        row = self.packet['samples'][0]['design']['figures'][0]
        row['number'] = 2
        with self.assertRaisesRegex(planning.PlanningError, 'number differs'):
            planning.calibrate(self.packet, self.root)
        row['number'] = 1; row['caption'] = 'A different caption.'
        with self.assertRaisesRegex(planning.PlanningError, 'caption differs'):
            planning.calibrate(self.packet, self.root)

    def test_real_pdf_panel_measurement_prefills_every_panel(self):
        source = (self.root / 'sample.tex').read_text()
        source = source.replace(r'\noindent Figure 1.', r'\noindent(a) First case.\par' + '\n' + r'\noindent(b) Second case.\par' + '\n' + r'\noindent Figure 1.')
        (self.root / 'panels.tex').write_text(source)
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'panels.tex'],
                                cwd=self.root, capture_output=True, timeout=60)
        self.assertEqual(0, result.returncode)
        self.packet['samples'][0]['pdf'] = 'panels.pdf'; self.packet['artifacts'].append('panels.pdf')
        template = venue_tools.design_template(self.packet, self.root)
        row = template['papers']['S1']['figures'][0]
        self.assertEqual(['a', 'b'], [panel['label'] for panel in row['panels']])
        self.assertTrue(row['caption'].startswith('Figure 1.'))
        self.assertEqual('', row['panels'][0]['claim'])

    def test_eight_distinct_actual_pdf_fixtures_use_the_default_count_policy(self):
        prototype = deepcopy(self.packet['samples'][0])
        self.packet['samples'] = []
        for index in range(1, 9):
            name = 'acceptedfixture' + str(index)
            make_pdf(self.root, name=name)
            sample = deepcopy(prototype); sample.update(id='S' + str(index), pdf=name + '.pdf')
            self.packet['samples'].append(sample); self.packet['artifacts'].append(sample['pdf'])
        template = venue_tools.design_template(self.packet, self.root)
        for sample in self.packet['samples']:
            design = template['papers'][sample['id']]
            sample['design'] = {key: design[key] for key in ('figures', 'tables')}
            for plural, visual_type in (('figures', 'quantitative_results'), ('tables', 'main_results')):
                for row in sample['design'][plural]:
                    row.update(type=visual_type, placement='body', question='Which fixture condition is reported?',
                        claim='The synthetic condition is explicitly labelled.', evidence='The compiled fixture body and captions.',
                        design_takeaway='Keep condition and measurement labels together.', limitations='Software test data only.')
        computed = planning.calibrate(self.packet, self.root)
        self.assertEqual(8, len(computed['measurements']))
        self.assertEqual(8, len({record['sha256'] for record in computed['measurements']}))
        self.assertTrue(computed['venue_profile']['sampling_policy']['count_in_default_range'])
        self.assertFalse(computed['venue_profile']['sampling_policy']['budget_is_reference_only'])
        self.assertIsNone(computed['venue_profile']['sampling_policy']['count_deviation_reason'])

    def test_panel_semantics_and_measurement_corrections_are_explicit(self):
        sample = self.packet['samples'][0]
        measured = planning.venue_profile.measure_pdf(self.root / 'sample.pdf')
        measured['figures'][0]['panels'] = ['a', 'b']
        with self.assertRaisesRegex(planning.PlanningError, 'cover detected panel'):
            venue_tools.validate_design(sample, measured)
        row = sample['design']['figures'][0]
        row['panels'] = [{'label': label, 'question': 'Which paired case differs?', 'claim': 'This case shows the measured distinction.',
            'evidence': 'The case-specific record.', 'design_takeaway': 'Use the same axis for paired cases.',
            'limitations': 'The case is illustrative only.'} for label in ('a', 'b')]
        self.assertEqual(row, venue_tools.validate_design(sample, measured)['figures'][0])
        row['panels'][0]['evidence'] = ''
        with self.assertRaisesRegex(planning.PlanningError, 'Per-panel design missing evidence'):
            venue_tools.validate_design(sample, measured)

    def test_page_budget_preserves_official_reference_appendix_rule(self):
        computed = planning.calibrate(self.packet, self.root)
        accounting = computed['venue_profile']['page_accounting']
        self.assertFalse(accounting['selected_rule']['includes_references'])
        self.assertFalse(accounting['selected_rule']['includes_appendix'])
        self.assertEqual('total_pages_fallback', accounting['sample_statistics'][0]['pages_basis'])
        self.packet['page_budget']['source']['quote'] = 'An invented eight-page rule'
        with self.assertRaisesRegex(planning.PlanningError, 'quote is absent'):
            planning.calibrate(self.packet, self.root)

    def test_sample_habits_cannot_become_official_hard_limits(self):
        self.packet['page_budget'] = {'target': 1, 'basis': 'sample_inference', 'limit_type': 'hard_limit',
            'includes_references': True, 'includes_appendix': True, 'sample_ids': ['S1'],
            'reason': 'An observed sample length, not an official limit.'}
        with self.assertRaisesRegex(planning.PlanningError, 'official hard limit'):
            planning.calibrate(self.packet, self.root)
        self.packet['page_budget']['limit_type'] = 'target'
        self.assertEqual('sample_inference', planning.calibrate(self.packet, self.root)['venue_profile']['page_accounting']['selected_rule']['basis'])

    def test_snapshot_cannot_freeze_a_different_promise_set(self):
        snapshot = json.loads((self.root / 'snapshot.json').read_text())
        snapshot['items'].pop()
        dump(self.root / 'snapshot.json', snapshot)
        self.assertIn('different items', ';'.join(planning.validate('plan', self.plan_packet, self.root)))

    def test_r6_omitted_item_and_rewritten_plan_fail_despite_total_score(self):
        review = self.review_packet()
        review['overall_score'] = 100
        review['r6_reviews'].pop()
        self.assertIn('every original', ';'.join(planning.validate('review', review, self.root)))
        current = visual_plan(); current['items'].pop()
        dump(self.root / 'current.json', current)
        review = self.review_packet(); review['figure_table_plan'] = 'current.json'; review['artifacts'].append('current.json')
        self.assertIn('changed promise fulfilled', ';'.join(planning.validate('review', review, self.root)))

    def test_r6_uses_bound_original_snapshot(self):
        review = self.review_packet()
        dump(self.root / 'other.json', planning.plan_snapshot(self.root / 'visual-plan.json'))
        review['plan_snapshot'] = 'other.json'; review['artifacts'].append('other.json')
        self.assertIn('original frozen', ';'.join(planning.validate('review', review, self.root)))

    def test_fixture_must_not_be_mislabelled_real_venue_evidence(self):
        self.packet['software_fixture'] = False
        self.assertIn('fixture URL', ';'.join(planning.validate('venue', self.packet, self.root)))


if __name__ == '__main__':
    unittest.main()
