import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.runtime import Controller, Refusal, file_hash, read_json
from run_fixture import make_brief, run, FixturePaused, put


class WorkflowV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.ctl = Controller(self.root)

    def init(self, **changes):
        brief = make_brief(); brief.update(changes); self.ctl.init(brief)
        if brief.get('contract_status') != 'draft': self.ctl.approve(self.ctl.load()['brief_sha256'], 'fixture')

    def decide(self, request, op, **changes):
        decision = {'request_sha256': request['request_sha256'], 'progress': 'no_progress', 'reason': 'Software checks only', 'status': 'active', **changes}
        return self.ctl.apply_round(request, decision, op)

    def test_undecided_venue_is_researched_before_the_final_contract(self):
        result = run(self.root, undecided=True)
        state = self.ctl.load()
        self.assertEqual('software_fixture_complete', result['status'])
        self.assertEqual('Synthetic Software Fixture', state['brief']['venue'])
        self.assertEqual('offline fixture final Brief', state['approval']['actor'])
        self.assertEqual('V1', state['venue_confirmation']['candidate_id'])
        draft = next(h['brief'] for h in state['history'] if h.get('event') == 'user_field_amendment' and h.get('field') == 'venue')
        self.assertEqual([], draft['deliverable_plan']); self.assertEqual('undecided', draft['venue'])
        self.assertLess(state['receipts'][1]['at'], state['approval']['at'])
        self.assertEqual(set(['source','pdf','bibliography','citation_plan','citation_verification','review','revision','snapshot']), set(state['receipts'][-1]['packet']['completion']))

    def test_source_synthesis_runs_to_export_without_artificial_experiments(self):
        result = run(self.root, undecided=True, survey=True)
        state = self.ctl.load(); stages = {r['stage']: r['packet'] for r in state['receipts']}
        self.assertEqual('software_fixture_complete', result['status'])
        self.assertNotIn('experiment_design_sha256', stages['plan'])
        self.assertFalse((self.root / 'experiment/results.json').exists())
        self.assertFalse((self.root / 'experiment/evaluator.py').exists())
        self.assertEqual('evidence_diagram', stages['figures']['mode'])
        self.assertEqual('not_applicable', stages['review']['gates']['G2']['verdict'])
        self.assertEqual(0, state['session']['experiment_rounds'])
        assertions = read_json(self.root / stages['write']['assertions'])
        self.assertEqual('source_claim', assertions['assertions'][0]['kind'])
        self.assertEqual(8, len(stages['review']['completion']))

    def test_round_start_captures_plan_before_changes_and_reads_are_not_progress(self):
        with self.assertRaises(FixturePaused): run(self.root, stop_after='plan')
        before = self.ctl.begin_round()
        self.assertEqual(0, self.ctl.load()['session']['rounds'])
        self.assertEqual(before['round']['plan_sha256'], file_hash(self.root / before['round']['plan_path']))
        put(self.root, 'plan/revised.json', {'items': []})
        request = self.ctl.prepare_round([], 'plan/revised.json')
        self.assertEqual(['F1','T1'], request['plan_diff']['removed'])
        with self.assertRaisesRegex(Refusal, 'R6 disclosure'): self.decide(request, 'bad-plan-rewrite')
        self.assertEqual(0, self.ctl.load()['session']['rounds'])

    def test_round_workload_reports_large_sources_without_stopping_the_task(self):
        self.init(); self.ctl.begin_round()
        for n in range(6): put(self.root, f'code/{n}.py', '# software fixture\n' * 301)
        request = self.ctl.prepare_round([])
        self.assertTrue(request['workload']['attention_required'])
        self.decide(request, 'large-change-checkpoint')
        self.ctl.begin_round(); put(self.root, 'new.tex', '% generated fixture\n' * 301)
        request = self.ctl.prepare_round([])
        self.decide(request, 'new-long-tex')

    def test_round_prepare_cannot_be_forged_or_applied_after_workload_changes(self):
        from lib.runtime import digest
        self.init(); self.ctl.begin_round()
        for n in range(6): put(self.root, f'code/{n}.py', '# software fixture\n' * 301)
        request = self.ctl.prepare_round([])
        forged = copy.deepcopy(request); forged['workload'] = None
        forged['request_sha256'] = digest({k:v for k,v in forged.items() if k != 'request_sha256'})
        with self.assertRaisesRegex(Refusal,'not issued'): self.decide(forged,'forged')
        put(self.root,'another.py','# changed after preparation\n')
        with self.assertRaisesRegex(Refusal,'documents changed'): self.decide(request,'stale-workload',workload_deviation_reason='Temporary coordinated work')

    def test_four_stale_rounds_request_attention_without_a_hidden_stop(self):
        self.init()
        for n in range(4):
            self.ctl.begin_round(); request = self.ctl.prepare_round([])
            result = self.decide(request, 'stale'+str(n), direction={'id': 'direction'+str(n), 'change': 'Different comparison family '+str(n), 'reason':'Software pivot fixture'})
        self.assertTrue(result['attention_required'])
        self.ctl.begin_round()
        self.assertEqual(4, len(self.ctl.load()['rounds']))

    def test_experiment_round_counts_do_not_create_an_implicit_resource_limit(self):
        self.init()
        self.ctl.begin_round(); request = self.ctl.prepare_round([])
        self.decide(request,'experiment-1',phase='experiment')
        self.ctl.begin_round(); request = self.ctl.prepare_round([])
        self.decide(request,'experiment-2',phase='experiment')
        self.assertEqual(2,self.ctl.load()['session']['experiment_rounds'])

    def test_experiment_repair_is_not_blocked_by_session_diagnostic_counts(self):
        with self.assertRaises(FixturePaused): run(self.root, stop_after='experiment')
        state = self.ctl.load(); previous = state['receipts'][-1]['packet']
        state['session']['experiment_rounds'] = 99; self.ctl.save(state)
        self.ctl.rewind('experiment', 'Repair measured evidence', 'rewind-pending-experiment')
        self.ctl.commit(previous, 'repeat-experiment-before-checkpoint')

    def test_snapshot_tool_and_lost_ledger_do_not_reset_history(self):
        self.init(); put(self.root,'plan.json',{'items':[]})
        first = self.ctl.tool('plan-snapshot', {'figure_table_plan':'plan.json'}, 'snapshot.json')
        second = self.ctl.tool('plan-snapshot', {'figure_table_plan':'plan.json'}, 'snapshot.json')
        self.assertEqual(first['written'], second['written'])
        (self.root / '.paper/control.json').unlink()
        (self.root / '.paper/history-evidence.json').write_text('{}')
        with self.assertRaisesRegex(Refusal, 'recover'): self.ctl.init(make_brief())

    def test_structured_input_size_and_hard_links_are_refused(self):
        self.init(); big = self.root/'big.json'
        with big.open('wb') as stream: stream.seek(16*1024*1024); stream.write(b' ')
        with self.assertRaisesRegex(Refusal, '16 MiB'): read_json(big)
        put(self.root,'one.txt','Owned material')
        os.link(self.root/'one.txt',self.root/'two.txt')
        with self.assertRaisesRegex(Refusal, 'Hard-linked'): self.ctl.read_material('two.txt')

    def test_finding_revision_keeps_identity_and_does_not_invent_limitations(self):
        self.init(); put(self.root,'finding.md','Synthetic observation, version one.')
        self.ctl.begin_round(); request = self.ctl.prepare_round(['finding.md'])
        first = {'id':'F1','claim':'A software observation','validation':'Fixture record read','limitations':'','revises':None}
        result = self.decide(request,'finding-first',progress='verified_progress',finding=first)
        self.assertEqual(1,result['total_findings'])
        self.ctl.begin_round(); put(self.root,'finding.md','Synthetic observation with additional support.')
        request = self.ctl.prepare_round(['finding.md']); revised = {**first,'id':'F2','revises':'F1'}
        result = self.decide(request,'finding-revision',progress='verified_progress',finding=revised)
        self.assertEqual(1,result['total_findings'])

    def test_draft_preview_is_marked_and_lost_controller_can_recover(self):
        run(self.root)
        preview = self.ctl.draft_preview('manuscript/main.pdf','previews/review-draft.pdf')
        text = subprocess.check_output(['pdftotext',str(self.root/preview['path']),'-'],text=True)
        self.assertIn('DRAFT',text); self.assertTrue(preview['not_delivered'])
        before = self.ctl.load(); self.ctl.path.unlink()
        with self.assertRaisesRegex(Refusal,'recover'): self.ctl.init(make_brief())
        result = self.ctl.recover('delivery/export-fixture/manifest.json')
        self.assertEqual('recovered',result['status'])
        self.assertEqual(before['brief_sha256'],self.ctl.load()['brief_sha256'])
        self.assertEqual(len(before['rounds']),len(self.ctl.load()['rounds']))

    def test_draft_can_record_a_real_wait_before_final_approval(self):
        self.init(contract_status='draft',venue='undecided',deliverable_plan=[])
        self.ctl.begin_round(); request = self.ctl.prepare_round([])
        result = self.decide(request,'wait-for-selection',status='needs_input',blockers=[{'kind':'input','reason':'Await actual user venue choice','requirement_ids':[]}])
        self.assertEqual('round_recorded',result['status'])
        self.assertEqual('needs_input',self.ctl.load()['last_round_status'])
        self.assertIsNone(self.ctl.load()['approval'])

    def test_recovery_keeps_post_export_rounds_and_pending_judgment(self):
        run(self.root)
        self.ctl.begin_round(); request = self.ctl.prepare_round([])
        self.decide(request,'post-export-round')
        self.ctl.begin_round(); pending = self.ctl.prepare_round([])
        before = self.ctl.load(); self.ctl.path.unlink()
        self.ctl.recover('delivery/export-fixture/manifest.json')
        after = self.ctl.load()
        self.assertEqual(before['rounds'],after['rounds'])
        self.assertIn('post-export-round',after['operations'])
        self.assertEqual(pending,after['pending_round'])
        self.assertEqual(before['active_round'],after['active_round'])

    def test_explicit_task_deadline_cannot_be_reopened_by_rewind(self):
        run(self.root, budget={'calls': 0, 'minutes': 1})
        from datetime import datetime, timedelta, timezone
        state = self.ctl.load()
        state['created_at'] = (datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()
        self.ctl.save(state)
        with self.assertRaisesRegex(Refusal,'budget'): self.ctl.rewind('write','Proposed revision','budget-rewind')
        with self.assertRaisesRegex(Refusal,'budget'):
            self.ctl.execute('W-writing','budget-execute')

    def test_r6_round_report_must_be_real_and_bind_both_plan_versions(self):
        with self.assertRaises(FixturePaused): run(self.root,stop_after='plan')
        started = self.ctl.begin_round()['round']
        put(self.root,'plan/current.json',{'items':[]})
        request = self.ctl.prepare_round([], 'plan/current.json')
        with self.assertRaisesRegex(Refusal,'evidence-bound'):
            self.decide(request,'fake-r6',r6={'verdict':'scientifically_adjusted','evidence':['nonexistent.json']})
        report = put(self.root,'review/round-r6.json',{'round_id':started['id'],'original_plan_sha256':started['plan_sha256'],
              'current_plan_sha256':file_hash(self.root/'plan/current.json'),'verdict':'scientifically_adjusted','reason':'Proposal only; final contract remains unchanged'})
        request = self.ctl.prepare_round([report], 'plan/current.json')
        result = self.decide(request,'bound-r6',r6={'verdict':'scientifically_adjusted','evidence':[report]})
        self.assertEqual('round_recorded',result['status'])

    def test_cli_judgment_channel_is_excluded_from_workload_but_cannot_be_an_artifact(self):
        from lib.runtime import safe
        self.init(); self.ctl.begin_round(); request = self.ctl.prepare_round([])
        decision = {'request_sha256':request['request_sha256'],'progress':'no_progress','status':'active','reason':'No work happened'}
        path = put(self.root,'control/round-decisions/one.json',decision)
        self.ctl.apply_round(request,decision,'judgment-channel')
        with self.assertRaisesRegex(Refusal,'controller-owned'): safe(self.root,path)

    def test_heartbeat_does_not_invalidate_prepared_research_evidence(self):
        self.init(); self.ctl.begin_round(); request = self.ctl.prepare_round([])
        before = self.ctl.load()['last_progress_at']
        self.ctl.begin_round()
        self.decide(request,'after-heartbeat')
        self.assertEqual(before,self.ctl.load()['last_progress_at'])

    def test_complete_local_prose_guidance_and_selected_cases_are_available(self):
        from lib.writing import guidance
        package = Path(__file__).resolve().parents[1]
        value = guidance(package,['E07','E06'])
        self.assertEqual(['E07','E06'],[c['id'] for c in value['selected_cases']])
        self.assertIn('关键写作约束',value['principles'])
        self.assertGreater(len(value['selected_cases'][0]['text']),100)
        with self.assertRaises(ValueError): guidance(package,['E01','E02','E03'])

    def test_main_skill_enters_research_and_writing_remains_late(self):
        root = Path(__file__).resolve().parents[2]
        if not (root/'.claude').exists(): self.skipTest('Source routing check')
        skill = (root/'.claude/skills/autoresearch/SKILL.md').read_text()
        self.assertNotIn('/autoresearch:paper',skill)
        self.assertFalse((root/'.claude/commands/autoresearch/paper.md').exists())
        self.assertTrue((root/'.claude/commands/autoresearch/iterate.md').exists())
        self.assertIn('Writing is the late',skill)
        help_ = subprocess.check_output([sys.executable,str(root/'scripts/research.py'),'--run',str(self.root),'--help'],text=True)
        for command in ['tool','confirm-venue','set-contract','begin-round','recover']: self.assertIn(command,help_)


if __name__ == '__main__': unittest.main()
