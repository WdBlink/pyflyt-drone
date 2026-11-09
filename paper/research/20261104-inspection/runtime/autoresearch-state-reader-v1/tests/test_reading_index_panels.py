"""Software-only regressions for complete reading records and tool parity."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import production
from lib.experiments import ExperimentRecorder
from lib.runtime import Controller, Refusal
from test_production import design, make_brief, put


class ReadingIndexPanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'source.pdf').write_bytes(b'%PDF-structural-software-test')
        panel = dict(label='a', question='Panel question', claim='Panel claim',
                     evidence='Panel observations', design_takeaway='Panel lesson',
                     limitations='Panel scope', locator='render/page-2.png; upper left')
        visual = dict(id='P1-F1', page=2, question='Visual question', claim='Visual claim',
                      evidence='Visual observations', design_takeaway='Visual lesson',
                      limitations='Visual scope', panels=[panel], locator='render/page-2.png',
                      panel_measurement_note='Rendered image adds missed panel a')
        self.packet = {'target':'notes/index.md','samples':[dict(id='P1',title='Actual supplied title',
            pdf='source.pdf', design={'figures':[visual], 'tables':[]},
            visual_read={'pages':[2],'method':'rendered_pdf','notes':'Software test declaration'})]}

    def test_every_panel_field_and_location_is_visible_and_rechecked(self):
        result = production.reading_index(self.packet, self.root)
        text = (self.root/'notes/index.md').read_text()
        self.assertEqual(1, result['counts']['facets_or_blocks'])
        for expected in ['Actual supplied title','#### P1-F1 / a','Panel question','Panel claim',
                         'Panel observations','Panel lesson','Panel scope','upper left',
                         '../source.pdf#page=2','Rendered image adds missed panel a']:
            self.assertIn(expected, text)
        production.reading_index(self.packet, self.root, check=True)
        changed = copy.deepcopy(self.packet)
        changed['samples'][0]['design']['figures'][0]['panels'][0]['claim'] = 'Changed claim'
        with self.assertRaisesRegex(Refusal, 'Reading index differs'):
            production.reading_index(changed, self.root, check=True)

    def test_incomplete_or_duplicate_panels_cannot_hide_behind_counts(self):
        p = copy.deepcopy(self.packet)
        del p['samples'][0]['design']['figures'][0]['panels'][0]['evidence']
        with self.assertRaisesRegex(Refusal, 'Incomplete per-panel'):
            production.reading_index(p, self.root)
        p = copy.deepcopy(self.packet)
        panels = p['samples'][0]['design']['figures'][0]['panels']
        panels.append(copy.deepcopy(panels[0]))
        with self.assertRaisesRegex(Refusal, 'Duplicate panel'):
            production.reading_index(p, self.root)

    def test_malformed_panel_shapes_and_falsey_locations_are_rejected(self):
        for value in [None, 3, False, 'a']:
            with self.subTest(panels=value):
                p = copy.deepcopy(self.packet)
                p['samples'][0]['design']['figures'][0]['panels'] = value
                with self.assertRaisesRegex(Refusal, 'panels must be an array'):
                    production.reading_index(p, self.root)
        for field, value in [('locator', False), ('panel_measurement_note', 0)]:
            with self.subTest(field=field):
                p = copy.deepcopy(self.packet)
                p['samples'][0]['design']['figures'][0][field] = value
                with self.assertRaisesRegex(Refusal, 'must be text'):
                    production.reading_index(p, self.root)
        p = copy.deepcopy(self.packet)
        p['samples'][0]['design']['figures'][0]['panels'][0]['locator'] = []
        with self.assertRaisesRegex(Refusal, 'Panel locator must be text'):
            production.reading_index(p, self.root)

    def test_controller_statistics_uses_the_same_validated_renderer(self):
        ctl = Controller(self.root)
        ctl.init(make_brief())
        put(self.root, 'experiment/design.json', design())
        with ExperimentRecorder(self.root) as records:
            for case, seed, condition in records.grid:
                with records.attempt(case, seed, condition) as complete:
                    complete(2 if condition == 'baseline' else 1, {'kind':'software_fixture'})
        result = ctl.tool('statistics', {'input':'experiment/results.json',
                         'table':'stats/table.tex','figure':'stats/figure.svg'})
        self.assertEqual('statistics', result['tool'])
        self.assertEqual(-1, result['value']['statistics']['delta']['mean'])
        raw = self.root/'experiment/raw/site-a-candidate-7.json'
        data = json.loads(raw.read_text()); data['value'] = 0; put(self.root, raw.relative_to(self.root), data)
        with self.assertRaises(ValueError):
            ctl.tool('statistics', {'input':'experiment/results.json',
                     'table':'stats/invalid.tex','figure':'stats/invalid.svg'})


if __name__ == '__main__':
    unittest.main()
