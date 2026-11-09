"""Synthetic engineering fixtures only; these are never scientific evidence."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import production, science
from lib.experiments import ExperimentRecorder, MeasurementFailure
from lib.runtime import Refusal


def fixture_design(arms):
    return {
        'experiment_id': 'engineering-fixture', 'claim_id': 'fixture-only',
        'hypothesis': 'Exercise recording mechanics only', 'independent_variable': 'arm',
        'dependent_variable': 'fixture number', 'control_variables': ['fixed fixture'],
        'metric': {'name': 'fixture', 'unit': 'units', 'direction': 'lower'},
        'statistical_unit': 'case', 'cases': ['a', 'b', 'c'], 'seeds': [0, 1],
        'conditions': arms, 'failure_policy': 'penalty', 'failure_value': 1000,
        'statistics': {'estimator': 'mean', 'interval': 'sd', 'paired': True,
                       'contrasts': [{'id': 'pair', 'minuend': arms[1], 'subtrahend': arms[0]},
                                     {'id': 'reverse', 'minuend': arms[0], 'subtrahend': arms[1]}]},
        'data_roles': [{'id': 'fixture', 'role': 'development', 'used_for_selection': True}],
        'planned_at': '2025-01-01T00:00:00Z',
    }


class MultiArmTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def get(self, name):
        return json.loads((self.root / name).read_text())

    def record(self, count=3, failure=False):
        design = fixture_design(['arm_' + str(i) for i in range(count)])
        self.put('experiment/design.json', design)
        with ExperimentRecorder(self.root) as records:
            for case, seed, arm in records.grid:
                with records.attempt(case, seed, arm) as complete:
                    if failure and arm == design['conditions'][-1] and case == 'a':
                        raise MeasurementFailure('Engineering protocol fixture failure', status='failed' if seed == 0 else 'stopped')
                    c = design['cases'].index(case) + 1
                    complete(10 * c + seed * 100 + design['conditions'].index(arm) * c)
        return design

    def stats(self):
        return science._statistics(self.root, 'experiment/design.json', 'experiment/results.json')

    def test_three_and_seven_arms_preserve_pairing_and_render_all_conditions(self):
        for count in (3, 7):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                original = self.root
                self.root = Path(tmp)
                try:
                    design = self.record(count)
                    result = production.run('statistics', {'input': 'experiment/results.json', 'table': 'table.tex', 'figure': 'figure.svg'}, self.root)
                    stats = result['statistics']
                    self.assertEqual(count * 6, stats['planned_runs'])
                    self.assertEqual(count, len(stats['conditions']))
                    self.assertEqual(3, stats['n_cases'])
                    self.assertEqual([60, 70, 80], stats['case_values']['arm_0'])
                    contrast = stats['contrasts']['pair']
                    self.assertEqual([1, 2, 3], contrast['case_differences'])
                    self.assertEqual(2, contrast['mean'])
                    self.assertEqual(1, contrast['sd'])
                    self.assertAlmostEqual(1 / math.sqrt(3), contrast['se'])
                    self.assertEqual(-2, stats['contrasts']['reverse']['mean'])
                    self.assertNotIn('delta', stats)
                    self.assertEqual(design['conditions'], stats['condition_order'])
                    self.assertEqual(set(design['conditions']), set(stats['case_values']))
                    for arm in design['conditions']:
                        self.assertIn(arm, (self.root / 'figure.svg').read_text())
                        self.assertIn(science._tex(arm), (self.root / 'table.tex').read_text())
                    self.assertIn('No multiplicity-adjusted', result['caption'])
                finally:
                    self.root = original

    def test_pdf_contains_every_named_condition(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest('Matplotlib unavailable; explicit capability limitation')
        self.record(7)
        rendered = production.run('statistics', {'input': 'experiment/results.json', 'table': 'pdf-table.tex', 'figure': 'figure.pdf'}, self.root)
        self.assertTrue((self.root / 'figure.pdf').read_bytes().startswith(b'%PDF'))
        self.assertEqual(7, len(rendered['statistics']['conditions']))

    def test_named_measures_integrate_with_writing_validation(self):
        from test_science import ScienceChecks
        for measure, value in [('conditions.candidate.mean', 13 / 3), ('contrasts.pair.mean', -2 / 3)]:
            with self.subTest(measure=measure):
                fixture = ScienceChecks()
                fixture.setUp()
                try:
                    fixture.design['conditions'].append('third')
                    fixture.design['statistics']['contrasts'] = [
                        {'id': 'pair', 'minuend': 'candidate', 'subtrahend': 'baseline'}]
                    fixture.put('experiment/design.json', fixture.design)
                    fixture.design_hash = fixture.hash('experiment/design.json')
                    fixture.plan['experiment_design_sha256'] = fixture.design_hash
                    fixture.results['design_sha256'] = fixture.design_hash
                    fixture.results['reported_denominator'] = 18
                    for row in list(fixture.results['runs'][:6]):
                        row = dict(row, id=row['id'] + '-third', condition='third', raw=row['raw'] + '.third')
                        fixture.put(row['raw'], row)
                        fixture.results['runs'].append(row)
                    fixture.put('experiment/results.json', fixture.results)
                    packet, document = fixture.writing_packet()
                    assertion = document['assertions'][0]
                    old_text = assertion['text']
                    assertion.update(measure=measure, value=value, text=f'The descriptive mean is {value:.6f} m.')
                    manuscript = (fixture.root / 'paper/main.tex').read_text().replace(old_text, assertion['text'])
                    fixture.put('paper/main.tex', manuscript)
                    fixture.put('paper/assertions.json', document)
                    self.assertEqual([], science.validate('write', packet, fixture.root))
                    assertion['measure'] = 'contrasts.unregistered.mean'
                    fixture.put('paper/assertions.json', document)
                    self.assertIn('unsupported assertion measure', science.validate('write', packet, fixture.root)[0])
                finally:
                    fixture.doCleanups()

    def test_noncontrast_arm_failures_keep_full_denominator(self):
        self.record(7, failure=True)
        stats = self.stats()
        self.assertEqual(42, stats['planned_runs'])
        self.assertEqual(1, stats['failed_runs'])
        self.assertEqual(1, stats['stopped_runs'])
        self.assertEqual(1000, stats['case_values']['arm_6'][0])
        self.assertEqual(2, stats['contrasts']['pair']['mean'])

    def test_operation_abort_retains_unstarted_seven_arm_slots(self):
        self.put('experiment/design.json', fixture_design(['arm' + str(i) for i in range(7)]))
        with self.assertRaisesRegex(RuntimeError, 'fixture abort'):
            with ExperimentRecorder(self.root) as records:
                for i, key in enumerate(records.grid):
                    with records.attempt(*key) as complete:
                        if i == 2:
                            raise RuntimeError('fixture abort')
                        complete(i)
        results = self.get('experiment/results.json')
        self.assertEqual(42, results['reported_denominator'])
        self.assertEqual(3, results['attempted_count'])
        self.assertEqual(39, sum(row.get('attempted') is False for row in results['runs']))
        self.assertEqual(42, len(list((self.root / 'experiment/raw').glob('*.json'))))
        with self.assertRaisesRegex(ValueError, 'operation failures'):
            self.stats()

    def test_all_arms_are_validated_before_contrasts(self):
        self.record(7)
        original = self.get('experiment/results.json')
        target = original['runs'][-1]
        raw = (self.root / target['raw']).read_text()
        for corruption in ('missing', 'malformed', 'mismatch', 'bool_seed', 'duplicate_json_key'):
            with self.subTest(corruption=corruption):
                path = self.root / target['raw']
                if corruption == 'missing': path.unlink()
                elif corruption == 'malformed': path.write_text('{')
                elif corruption == 'duplicate_json_key': path.write_text('{"seed": 1, "seed": 1}')
                else:
                    changed = json.loads(raw)
                    changed['value' if corruption == 'mismatch' else 'seed'] = -1 if corruption == 'mismatch' else True
                    self.put(target['raw'], changed)
                with self.assertRaises((ValueError, OSError)):
                    self.stats()
                path.write_text(raw)
        for corruption in ('missing', 'duplicate', 'relabel', 'unstarted'):
            with self.subTest(corruption=corruption):
                results = copy.deepcopy(original)
                if corruption == 'missing': results['runs'].pop()
                elif corruption == 'duplicate': results['runs'][-1] = results['runs'][0]
                elif corruption == 'relabel': results['runs'][-1]['condition'] = 'arm_0'
                else: results['runs'][-1]['attempted'] = False
                self.put('experiment/results.json', results)
                with self.assertRaises(ValueError): self.stats()
        self.put('experiment/results.json', original)
        results = copy.deepcopy(original)
        results['outcome'] = 'positive'
        self.put('experiment/results.json', results)
        with self.assertRaisesRegex(ValueError, 'overall outcome'): self.stats()

    def test_contrasts_must_be_frozen_and_well_formed(self):
        base = fixture_design(['first', 'second', 'third'])
        for kind in ('missing', 'empty', 'unknown', 'self', 'duplicate_id', 'duplicate_pair', 'malformed', 'duplicate_arm'):
            with self.subTest(kind=kind):
                design = copy.deepcopy(base)
                contrasts = design['statistics']['contrasts']
                if kind == 'missing': del design['statistics']['contrasts']
                elif kind == 'empty': design['statistics']['contrasts'] = []
                elif kind == 'unknown': contrasts[0]['minuend'] = 'absent'
                elif kind == 'self': contrasts[0]['minuend'] = contrasts[0]['subtrahend']
                elif kind == 'duplicate_id': contrasts[1]['id'] = contrasts[0]['id']
                elif kind == 'duplicate_pair': contrasts.append(dict(contrasts[0], id='other'))
                elif kind == 'malformed': contrasts[0]['minuend'] = []
                else: design['conditions'].append('first')
                self.put('experiment/design.json', design)
                with self.assertRaises(ValueError): science._design(self.root, 'experiment/design.json')
        self.record()
        changed = self.get('experiment/design.json')
        changed['statistics']['contrasts'][0]['minuend'] = 'arm_2'
        self.put('experiment/design.json', changed)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): self.stats()

    def test_recorder_rejects_duplicate_attempt_and_filename_collision(self):
        self.put('experiment/design.json', fixture_design(['one', 'two', 'three']))
        with self.assertRaisesRegex(Refusal, 'duplicate'):
            with ExperimentRecorder(self.root) as records:
                with records.attempt(*records.grid[0]) as complete: complete(1)
                with records.attempt(*records.grid[0]): pass
        design = fixture_design(['b-c', 'c', 'third'])
        design['cases'] = ['a', 'a-b']
        self.put('collision.json', design)
        with self.assertRaisesRegex(Refusal, 'ambiguous'):
            ExperimentRecorder(self.root, design='collision.json', results='new.json')


if __name__ == '__main__':
    unittest.main()
