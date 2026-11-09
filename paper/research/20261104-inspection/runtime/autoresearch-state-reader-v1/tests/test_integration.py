import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.runtime import Controller, Refusal, file_hash, read_json
from run_fixture import run, FixturePaused, SAMPLE_TEX, put


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory(); cls.addClassCleanup(cls.fixture.cleanup)
        cls.source = Path(cls.fixture.name) / 'complete'
        cls.result = run(cls.source)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'run'; shutil.copytree(self.source, self.root)
        self.ctl = Controller(self.root)

    def test_full_pipeline_preserves_negative_results_and_rebuilds_export(self):
        report = read_json(self.root / 'fixture-report.json')
        self.assertEqual('software_fixture_complete', report['status'])
        self.assertEqual(9, len(report['contexts']))
        self.assertEqual(0, report['model_calls']); self.assertEqual(0, report['remote_jobs'])
        self.assertEqual('not performed', report['independent_model_review'])
        results = read_json(self.root / 'experiment/results.json')
        self.assertEqual('negative', results['outcome']); self.assertEqual(16, len(results['runs']))
        self.assertTrue(any(r['status'] == 'failed' for r in results['runs']))
        self.assertTrue(any(r['status'] == 'stopped' for r in results['runs']))
        manifest = read_json(self.root / self.result['manifest'])
        self.assertEqual('rebuilt_from_exported_assets', manifest['rebuild']['status'])
        self.assertTrue(manifest['software'])
        text = subprocess.check_output(['pdftotext', str(self.root / 'manuscript/main.pdf'), '-'], text=True)
        self.assertNotIn('[?]', text)
        self.assertIn('[1]', text)
        self.assertIn('Denominator=16', text)
        self.assertIn('SD bounds', text)
        self.assertTrue((self.root / 'delivery/export-fixture.zip').is_file())
        self.assertEqual('verified', self.ctl.verify_export(self.result['manifest'])['status'])

    def test_missing_or_stale_item_cannot_be_hidden_by_overall_pass(self):
        state = self.ctl.load(); review = copy.deepcopy(state['receipts'][-1]['packet'])
        review['requirement_checks'].pop()
        with self.assertRaisesRegex(Refusal, 'every approved ID'):
            self.ctl.check_reviews(state, review)
        review = copy.deepcopy(state['receipts'][-1]['packet'])
        review['requirement_checks'][0]['binding_sha256'] = 'outdated'
        with self.assertRaisesRegex(Refusal, 'stale requirement'):
            self.ctl.check_reviews(state, review)
        # A valid negative outcome still cannot remove promised deliverables.
        del state['bindings']['F1']
        with self.assertRaisesRegex(Refusal, 'Missing requirement bindings'):
            self.ctl.check_reviews(state, state['receipts'][-1]['packet'])

    def test_changed_raw_evidence_invalidates_current_acceptance(self):
        raw = next((self.root / 'experiment/raw').glob('*.json'))
        record = read_json(raw); record['value'] = 123; put(self.root, str(raw.relative_to(self.root)), record)
        with self.assertRaisesRegex(Refusal, 'Stale experiment artifact'):
            self.ctl.finalize('new-export')
        # Historical exports remain immutable snapshots of their original evidence.
        self.ctl.verify_export(self.result['manifest'])

    def test_binding_and_reviewer_identity_are_checked(self):
        state = self.ctl.load(); review = copy.deepcopy(state['receipts'][-1]['packet'])
        review['reviewer'] = 'fixture-author'
        with self.assertRaisesRegex(Refusal, 'independently declared'):
            self.ctl.check_reviews(state, review)
        review['reviewer'] = 'fixture-reviewer'
        binding = state['bindings']['F1']; binding['hashes']['manuscript/main.tex'] = 'wrong'
        review['requirement_checks'] = [{'requirement_id': key, 'binding_sha256': value['binding_sha256'],
                                        'evidence_hashes': value['hashes'], 'verdict': 'fulfilled', 'rationale': 'test'} for key, value in state['bindings'].items()]
        with self.assertRaisesRegex(Refusal, 'artifact changed'):
            self.ctl.check_reviews(state, review)

    def test_rewind_preserves_history_and_invalidates_later_binding(self):
        before = self.ctl.load()
        result = self.ctl.rewind('write', 'Fixture revision requires new review', 'rewind-write')
        after = self.ctl.load()
        self.assertEqual('rewound', result['status'])
        self.assertEqual(6, len(after['receipts'])); self.assertEqual({}, after['bindings'])
        self.assertGreater(len(after['history']), len(before['history']))
        self.assertTrue(any(event.get('event') == 'invalidated' for event in after['history']))
        self.assertEqual(result, self.ctl.rewind('write', 'Fixture revision requires new review', 'rewind-write'))
        with self.assertRaisesRegex(Refusal, 'different arguments'):
            self.ctl.rewind('write', 'Different reason', 'rewind-write')

    def test_rework_versions_orders_without_changing_approved_obligations(self):
        self.ctl.rewind('write', 'Revise the wording under the same Brief', 'rewind-for-revision')
        original = read_json(self.root / 'write/packet.json')
        source = self.root / original['manuscript_tex']
        baseline_path = put(self.root, 'manuscript/baseline.tex', source.read_text())
        original['artifacts'].append(baseline_path)
        source.write_text(source.read_text().replace(r'\end{document}', r'Revised software-fixture prose.\end{document}'))
        new_order = {'id': 'W-writing-v2', 'producer': 'fixture-author', 'requirement_ids': ['S1', 'M1'], 'supersedes': 'W-writing',
                     'actions': [{'type': 'compile_tex', 'source': original['manuscript_tex'], 'pdf': 'manuscript/revised.pdf'}]}
        self.ctl.revise_orders([new_order], 'Versioned same-contract revision', 'register-revision')
        self.assertIn('W-writing-v2', [o['id'] for o in self.ctl.context()['work_orders']])
        self.ctl.execute('W-writing-v2', 'compile-revision')
        original['artifacts'] = ['manuscript/revised.pdf' if x == original['manuscript_pdf'] else x for x in original['artifacts']]
        original['manuscript_pdf'] = 'manuscript/revised.pdf'
        self.ctl.commit(original, 'revised-write')
        old_state = read_json(self.source / '.paper/control.json')
        for key, value in old_state['bindings'].items():
            binding = {k: copy.deepcopy(value[k]) for k in ('requirement_id', 'work_id', 'assets', 'evidence', 'manuscript', 'location')}
            if key in {'S1', 'M1'}: binding['work_id'] = 'W-writing-v2'
            if key == 'M1': binding['assets'] = ['manuscript/revised.pdf']
            self.ctl.bind(binding, 'rebind-' + key)
        review = read_json(self.root / 'review/packet.json')
        review['artifacts'] = ['manuscript/revised.pdf' if x == review['manuscript_pdf'] else x for x in review['artifacts']]
        review['manuscript_pdf'] = 'manuscript/revised.pdf'
        review['requirement_checks'] = [{'requirement_id': key, 'binding_sha256': value['binding_sha256'], 'evidence_hashes': value['hashes'],
                                         'verdict': 'fulfilled', 'rationale': 'Scripted rework fixture correspondence'} for key, value in self.ctl.load()['bindings'].items()]
        review['completion']['pdf'] = ['manuscript/revised.pdf']
        revision_path = review['completion']['revision'][0]
        put(self.root, revision_path, {'status': 'revised', 'changes': [{'baseline': {'path': baseline_path, 'sha256': file_hash(self.root / baseline_path)}, 'after': original['manuscript_tex'], 'reason': 'Versioned software-fixture prose revision'}]})
        snap_path = review['completion']['snapshot'][0]
        files = {p for r in self.ctl.load()['receipts'] for p in r['hashes']} | {review['completion']['review'][0], revision_path}
        put(self.root, snap_path, {'files': {p: file_hash(self.root / p) for p in files}})
        self.ctl.commit(review, 'revised-review')
        request = self.ctl.prepare_round(review['artifacts'])
        self.ctl.apply_round(request, {'request_sha256':request['request_sha256'],'progress':'verified_progress','reason':'Verified software revision only','status':'active'}, 'record-revision-round')
        result = self.ctl.finalize('revised-export')
        self.assertEqual('software_fixture_complete', result['status'])
        self.assertEqual(old_state['brief_sha256'], self.ctl.load()['brief_sha256'])
        self.assertEqual(old_state['frozen_plan'], self.ctl.load()['frozen_plan'])

    def test_decoy_manuscript_and_different_review_pdf_are_rejected(self):
        state = self.ctl.load()
        write = copy.deepcopy(next(r['packet'] for r in state['receipts'] if r['stage'] == 'write'))
        decoy = put(self.root, 'manuscript/decoy.tex', (self.root / write['manuscript_tex']).read_text())
        write['manuscript'] = decoy
        write['artifacts'].append(decoy)
        with self.assertRaisesRegex(Refusal, 'same manuscript source'):
            self.ctl.validate(state, write)
        review = copy.deepcopy(state['receipts'][-1]['packet'])
        review['manuscript_pdf'] = 'venue/sample.pdf'
        review['artifacts'].append('venue/sample.pdf')
        with self.assertRaisesRegex(Refusal, 'exact accepted manuscript PDF'):
            self.ctl.validate(state, review)

    def test_fixture_identity_and_duplicate_visual_binding_are_rejected(self):
        state = self.ctl.load()
        venue = copy.deepcopy(next(r['packet'] for r in state['receipts'] if r['stage'] == 'venue'))
        venue['software_fixture'] = False
        with self.assertRaisesRegex(Refusal, 'fixture identity'):
            self.ctl.validate(state, venue)
        self.ctl.rewind('review', 'Check duplicate visual binding', 'rewind-for-binding-check')
        first = state['bindings']['F1']
        self.ctl.bind({k: copy.deepcopy(first[k]) for k in ('requirement_id', 'work_id', 'assets', 'evidence', 'manuscript', 'location')}, 'rebind-first-visual')
        value = state['bindings']['T1']
        duplicate = {k: copy.deepcopy(value[k]) for k in ('requirement_id', 'work_id', 'assets', 'evidence', 'manuscript', 'location')}
        duplicate['assets'] = state['bindings']['F1']['assets']
        with self.assertRaisesRegex(Refusal, 'same primary asset'):
            self.ctl.bind(duplicate, 'reject-reused-figure')

    def test_export_tampering_is_detected(self):
        exported = self.root / 'delivery/export-fixture/artifacts/manuscript/main.pdf'
        exported.write_bytes(exported.read_bytes() + b'changed')
        with self.assertRaisesRegex(Refusal, 'Export artifact changed'):
            self.ctl.verify_export(self.result['manifest'])

    def test_reconciliation_rechecks_previous_outputs_and_pure_validation(self):
        root = Path(self.temp.name) / 'multi-action'
        with self.assertRaises(FixturePaused): run(root, stop_after='plan')
        ctl = Controller(root)
        ctl.begin_round()
        orders = [{'id': 'W-copy', 'producer': 'fixture-author', 'requirement_ids': ['F1'],
                   'actions': [{'type': 'copy_artifact', 'source': 'materials/project.md', 'target': 'work/first.md'},
                               {'type': 'copy_artifact', 'source': 'venue/rules.md', 'target': 'work/second.md'}]}]
        ctl.revise_orders(orders, 'Recovery software fixture', 'register-multi')
        actual = ctl._action
        def interrupted(state, action):
            result = actual(state, action)
            if action.get('target') == 'work/second.md': raise KeyboardInterrupt()
            return result
        with patch.object(ctl, '_action', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt): ctl.execute('W-copy', 'multi-lost')
        (root / 'work/first.md').unlink()
        attempt = ctl.load()['attempts']['multi-lost']
        report = put(root, 'recovery/multi.json', {'attempt': 'multi-lost', 'order_sha256': attempt['order_sha256'], 'pending_action': attempt['pending_action'],
                     'outcome': 'completed', 'reason': 'Only the last output is present', 'artifact_hashes': {'work/second.md': file_hash(root / 'work/second.md')}})
        with self.assertRaisesRegex(Refusal, 'Earlier action'):
            ctl.reconcile('multi-lost', 'completed', report, 'reject-lost-earlier-output')
        self.assertEqual('uncertain', ctl.load()['attempts']['multi-lost']['status'])
        # Resolve as failed without hiding the retained earlier receipt, then test a pure check.
        failed = read_json(root / report); failed['outcome'] = 'failed'
        put(root, report, failed); ctl.reconcile('multi-lost', 'failed', report, 'resolve-partial-copy-failure')
        bad_packet = put(root, 'checks/bad.json', {'stage': 'not-a-stage'})
        ctl.revise_orders([{'id': 'W-pure', 'producer': 'fixture-author', 'requirement_ids': ['F1'], 'actions': [{'type': 'validate_packet', 'packet': bad_packet}]}], 'Pure validation recovery', 'register-pure')
        with patch.object(ctl, '_action', side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt): ctl.execute('W-pure', 'pure-lost')
        attempt = ctl.load()['attempts']['pure-lost']
        report = put(root, 'recovery/pure.json', {'attempt': 'pure-lost', 'order_sha256': attempt['order_sha256'], 'pending_action': attempt['pending_action'],
                     'outcome': 'completed', 'reason': 'A packet file exists but it was never validated', 'artifact_hashes': {bad_packet: file_hash(root / bad_packet)}})
        with self.assertRaisesRegex(Refusal, 'Unknown packet stage'):
            ctl.reconcile('pure-lost', 'completed', report, 'reject-unperformed-validation')

    def test_uncertain_effects_require_reconciliation_and_are_not_replayed(self):
        root = Path(self.temp.name) / 'paused'
        with self.assertRaises(FixturePaused): run(root, stop_after='plan')
        ctl = Controller(root)
        ctl.begin_round()
        # A missing source is a known failure before any side effect.
        failed = ctl.execute('W-writing', 'known-failure')
        self.assertEqual('failed', failed['status'])
        put(root, 'manuscript/main.tex', SAMPLE_TEX)
        with self.assertRaises(KeyboardInterrupt):
            ctl.execute('W-writing', 'lost-terminal', fixture_fault=True)
        self.assertEqual('uncertain', ctl.load()['attempts']['lost-terminal']['status'])
        old_pdf = file_hash(root / 'manuscript/main.pdf')
        with self.assertRaisesRegex(Refusal, 'reconcile'):
            ctl.execute('W-writing', 'lost-terminal')
        with self.assertRaisesRegex(Refusal, 'Uncertain effects'):
            ctl.execute('W-writing', 'blind-retry')
        evidence = put(root, 'recovery/reconcile.json', {'attempt': 'lost-terminal', 'outcome': 'completed',
                       'reason': 'Actual exported local PDF found and hashed; compilation effect completed',
                       'artifact_hashes': {'manuscript/main.pdf': old_pdf}, 'order_sha256': ctl.load()['attempts']['lost-terminal']['order_sha256'], 'pending_action': ctl.load()['attempts']['lost-terminal']['pending_action']})
        invalid = read_json(root / evidence)
        invalid['artifact_hashes'] = {'materials/project.md': file_hash(root / 'materials/project.md')}
        invalid_name = put(root, 'recovery/unrelated.json', invalid)
        with self.assertRaisesRegex(Refusal, 'expected outputs'):
            ctl.reconcile('lost-terminal', 'completed', invalid_name, 'reject-unrelated-evidence')
        ctl.reconcile('lost-terminal', 'completed', evidence, 'reconcile-local-pdf')
        self.assertEqual('completed', ctl.load()['attempts']['lost-terminal']['status'])
        with self.assertRaisesRegex(Refusal, 'already completed'):
            ctl.execute('W-writing', 'still-no-replay')
        self.assertEqual(old_pdf, file_hash(root / 'manuscript/main.pdf'))


    def test_export_zip_and_upstream_runtime_closure_are_verified(self):
        manifest=self.result['manifest']
        verified=self.ctl.verify_export(manifest)
        self.assertEqual('verified',verified['archive']['crc_and_hashes'])
        payload=read_json(self.root/manifest)
        self.assertIn('upstream/reason.md',payload['software'])
        exported=self.root/manifest
        runtime=exported.parent/'software/scripts/research.py'
        result=subprocess.run([sys.executable,str(runtime),'--run',str(self.root),'context'],capture_output=True,text=True)
        self.assertEqual(0,result.returncode,result.stdout+result.stderr)
        self.assertEqual('delivery',json.loads(result.stdout)['stage'])
        archive=exported.parent.parent/(exported.parent.name+'.zip')
        with zipfile.ZipFile(archive,'a') as zipped:
            zipped.writestr('unexpected.txt','Not in the immutable package')
        with self.assertRaisesRegex(Refusal,'ZIP'):
            self.ctl.verify_export(manifest)

    def test_generic_review_assembly_copies_four_bindings_and_rejects_stale_judgments(self):
        from lib import reviewing
        packet={'request':'review/request.json','judgment':'review/judgment.json','target':'review/reassembled.json',
                'completion':{'revision':['review/revision.json'],'snapshot':'review/reassembled-snapshot.json','r6_report':'review/reassembled-r6.json'}}
        result=reviewing.assemble(packet,self.root)
        self.assertEqual(4,result['requirements']);self.assertEqual(2,result['original_visuals'])
        document=read_json(self.root/packet['target'])
        self.assertIn('figures/summary.tex',document['completion']['source'])
        judgment=read_json(self.root/'review/judgment.json')
        judgment['requirement_checks'][0]['binding_sha256']='stale'
        (self.root/'review/bad-judgment.json').write_text(json.dumps(judgment))
        packet['judgment']='review/bad-judgment.json';packet['target']='review/should-not-exist.json'
        with self.assertRaisesRegex(Refusal,'stale independent'):
            reviewing.assemble(packet,self.root)
        self.assertFalse((self.root/packet['target']).exists())

    def test_generic_review_cannot_hide_an_unresolved_finding(self):
        from lib import reviewing
        judgment=read_json(self.root/'review/judgment.json')
        judgment['findings']=[{'id':'method-error','severity':'major','status':'open'}]
        (self.root/'review/open-judgment.json').write_text(json.dumps(judgment))
        with self.assertRaisesRegex(Refusal,'Unresolved'):
            reviewing.assemble({'request':'review/request.json','judgment':'review/open-judgment.json'},self.root)

    def test_typed_tex_preparation_records_and_reconciles_transitive_inputs(self):
        self.ctl.rewind('write','Exercise generic formatting under the same contract','format-rewind')
        (self.root/'manuscript/additional.tex').write_text('A preserved source section.\n')
        source=(self.root/'manuscript/main.tex').read_text().replace('\\end{document}','\\input{additional.tex}\n\\end{document}')
        (self.root/'manuscript/format-source.tex').write_text(source)
        order={'id':'W-format','producer':'fixture-author','requirement_ids':['M1'],
               'actions':[{'type':'prepare_tex','source':'manuscript/format-source.tex','target':'manuscript/prepared.tex',
                           'options':{'inline_inputs':True,'link_borders':'hidden'}}]}
        self.ctl.revise_orders([order],'Use a registered generic formatter','format-orders')
        with self.assertRaises(KeyboardInterrupt):
            self.ctl.execute('W-format','format-attempt',fixture_fault=True)
        attempt=self.ctl.load()['attempts']['format-attempt']
        self.assertIn('manuscript/additional.tex',attempt['pending_input_hashes'])
        evidence={'attempt':'format-attempt','outcome':'completed','reason':'Actual generated file inspected',
                  'order_sha256':attempt['order_sha256'],'pending_action':attempt['pending_action'],
                  'artifact_hashes':{'manuscript/prepared.tex':file_hash(self.root/'manuscript/prepared.tex')}}
        (self.root/'review/format-reconcile.json').write_text(json.dumps(evidence))
        (self.root/'manuscript/additional.tex').write_text('Changed after the interrupted effect.\n')
        with self.assertRaisesRegex(Refusal,'inputs changed'):
            self.ctl.reconcile('format-attempt','completed','review/format-reconcile.json','format-reconcile')


    def test_rewind_preserves_accepted_baselines_for_generic_revision_records(self):
        from lib.production import revision_record
        original = (self.root/'manuscript/main.tex').read_bytes()
        result = self.ctl.rewind('write','Preserve the accepted source before editing','preserve-source', ['manuscript/main.tex'])
        baseline = result['baselines'][0]
        self.assertEqual(original,(self.root/baseline['path']).read_bytes())
        (self.root/'manuscript/main-new.tex').write_bytes(original+b'\n% a versioned source change\n')
        revision_record({'target':'review/new-revision.json','changes':[{'before':baseline['path'],
                         'after':'manuscript/main-new.tex','reason':'A bounded source revision'}]},self.root)
        document=read_json(self.root/'review/new-revision.json')
        self.assertEqual(baseline['sha256'],document['changes'][0]['baseline']['sha256'])
        self.assertEqual('revised',document['status'])


if __name__ == '__main__': unittest.main()
