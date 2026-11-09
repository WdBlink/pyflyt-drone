"""Budget and human-interaction software tests; no scientific API requests."""
from datetime import datetime, timedelta, timezone
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import budget
from lib.runtime import Controller, Refusal, file_hash
from run_fixture import run, make_brief, FixturePaused, put, answer_question


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ctl = Controller(self.root)

    def initialize(self, caps=None):
        brief = make_brief()
        brief['budget'] = {} if caps is None else caps
        self.ctl.init(brief)
        self.ctl.approve(self.ctl.load()['brief_sha256'], 'software-fixture-host')

    def planned(self):
        with self.assertRaises(FixturePaused):
            run(self.root, stop_after='plan')
        self.ctl.begin_round()

    def change_budget(self, caps, op='budget-answer'):
        plan = self.ctl.load()['budget_confirmation']['plan']['path']
        response = answer_question(self.ctl, 'budget', {'budget': caps, 'plan': plan,
            'reason': 'Software fixture resource proposal; no live model request'},
            {'choice': 'custom', 'budget': caps}, op)
        return self.ctl.confirm_budget(response, op)

    def receipt(self, name, **fields):
        return put(self.root, 'usage/' + name + '.json', {'kind': 'software_fixture', 'request_id': name, **fields})

    def close(self, **fields):
        request = self.ctl.prepare_round([])
        return self.ctl.apply_round(request, {'request_sha256': request['request_sha256'],
            'status': 'active', 'progress': 'no_progress', 'reason': 'Software checkpoint', **fields},
            'round-' + str(len(self.ctl.load()['rounds'])))

    def test_default_has_no_hidden_cap_and_old_locations_are_rejected(self):
        self.initialize()
        self.assertEqual({}, self.ctl.context()['budget']['limits'])
        for changes in [{'round_limits': {'session_minutes': 30}},
                        {'resources': {'authorization': 'software fixture', 'limits': {'host_session_minutes': 30}}},
                        {'budget': {'session_minutes': 30}}]:
            brief = make_brief(); brief.update(changes)
            with self.assertRaisesRegex(Refusal, 'budget'):
                Controller(self.root / ('bad-' + str(len(str(changes))))).init(brief)
        for caps in [{'calls': True}, {'tokens': -1}, {'minutes': 0}, {'minutes': None},
                     {'cost': {'amount': 'NaN', 'currency': 'USD'}}]:
            with self.assertRaises(Refusal):
                budget.quantities(caps, caps=True)
        self.assertEqual({'calls': 0}, budget.quantities({'calls': 0}, caps=True))

    def test_thirty_minutes_many_rounds_and_large_sources_are_advice_only(self):
        self.initialize()
        state = self.ctl.load()
        state['created_at'] = state['last_checkpoint_at'] = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        state['session']['rounds'] = 100
        state['session']['experiment_rounds'] = 99
        self.ctl.save(state)
        start = self.ctl.begin_round()
        self.assertTrue(start['checkpoint']['due'])
        put(self.root, 'long.tex', '% software fixture\n' * 401)
        self.close(phase='experiment')
        self.ctl.begin_round()
        self.assertIsNone(self.ctl.load()['task_stop'])

    def test_task_deadline_survives_new_session_but_checkpoint_can_close(self):
        self.initialize({'minutes': 1})
        self.ctl.begin_round(session_id='first-session')
        state = self.ctl.load(); state['created_at'] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(); self.ctl.save(state)
        self.close()
        with self.assertRaisesRegex(Refusal, 'budget.minutes'):
            self.ctl.begin_round(session_id='different-real-session')
        self.assertEqual(state['created_at'], self.ctl.load()['created_at'])

    def test_internal_blocker_or_unverified_budget_cannot_stop_task(self):
        self.initialize(); self.ctl.begin_round()
        request = self.ctl.prepare_round([])
        base = {'request_sha256': request['request_sha256'], 'progress': 'no_progress',
                'status': 'stopped', 'reason': 'An adapter needs implementation', 'blockers': ['Internal work']}
        with self.assertRaisesRegex(Refusal, 'Internal work'):
            self.ctl.apply_round(request, base, 'bad-stop')
        with self.assertRaisesRegex(Refusal, 'No explicit task budget'):
            self.ctl.apply_round(request, {**base, 'stop': {'reason': 'budget_exhausted'}}, 'fake-budget-stop')
        with self.assertRaisesRegex(Refusal, 'structured external'):
            self.ctl.apply_round(request, {**base, 'status': 'needs_input'}, 'fake-input')
        self.close()

    def test_only_partially_blocked_requirements_cannot_wait_globally(self):
        self.initialize(); self.ctl.begin_round()
        request = self.ctl.prepare_round([])
        decision = {'request_sha256': request['request_sha256'], 'progress': 'no_progress', 'status': 'needs_input',
            'reason': 'One dependency missing', 'blockers': [{'kind': 'authorization', 'reason': 'A later paid call needs permission', 'requirement_ids': ['F1']}]}
        with self.assertRaisesRegex(Refusal, 'Only some requirements'):
            self.ctl.apply_round(request, decision, 'partial-block')

    def test_user_stop_needs_actual_source_and_resumption_keeps_budget(self):
        self.initialize({'calls': 3}); self.ctl.begin_round()
        request = self.ctl.prepare_round([])
        quote = 'SOFTWARE FIXTURE stop answer, not actual user authorization'
        path = put(self.root, 'user-stop.txt', quote)
        source = {'path': path, 'sha256': file_hash(self.root/path), 'quote': quote}
        self.ctl.apply_round(request, {'request_sha256': request['request_sha256'], 'progress': 'no_progress',
            'reason': 'Fixture cancellation', 'status': 'stopped', 'stop': {'reason': 'user_request', 'human_response': source}}, 'stop')
        with self.assertRaisesRegex(Refusal, 'Task stopped'):
            self.ctl.begin_round()
        with self.assertRaisesRegex(Refusal, 'new actual user instruction'):
            self.ctl.resume_task(source, 'reuse-stop')
        path = put(self.root, 'user-resume.txt', 'SOFTWARE FIXTURE resume answer')
        self.ctl.resume_task({'path':path,'sha256':file_hash(self.root/path),'quote':'SOFTWARE FIXTURE resume answer'}, 'resume')
        self.ctl.begin_round()
        self.assertEqual({'calls': 3}, self.ctl.context()['budget']['limits'])

    def test_reservation_requires_confirmed_plan_not_just_delegated_approval(self):
        self.initialize({'calls': 2})
        with self.assertRaisesRegex(Refusal, 'ask the user'):
            self.ctl.reserve_budget({'usage': {'calls': 1}, 'purpose': 'Fixture request'}, 'request')

    def test_reservations_settlement_and_new_sessions_share_one_usage_ledger(self):
        self.planned(); self.change_budget({'calls': 3, 'tokens': 100})
        packet = {'usage': {'calls': 1, 'tokens': 40}, 'purpose': 'Software request'}
        first = self.ctl.reserve_budget(packet, 'first')
        self.assertEqual(first, self.ctl.reserve_budget(packet, 'first'))
        self.ctl.reserve_budget(packet, 'second')
        with self.assertRaisesRegex(Refusal, 'budget.tokens'):
            self.ctl.reserve_budget(packet, 'third')
        receipt = self.receipt('first')
        settled = self.ctl.settle_budget('first', {'receipt': receipt, 'status': 'known', 'usage': {'calls': 1, 'tokens': 30}}, 'settle-first')
        self.assertEqual(settled, self.ctl.settle_budget('first', {'receipt': receipt, 'status': 'known', 'usage': {'calls': 1, 'tokens': 30}}, 'settle-first'))
        self.ctl.begin_round(); self.close()
        self.ctl.begin_round(session_id='real-successor')
        view = self.ctl.context()['budget']
        self.assertEqual(1, view['known_usage']['calls'])
        self.assertEqual(1, view['reserved']['calls'])
        self.assertEqual(30, view['remaining']['tokens'])

    def test_unknown_usage_is_not_zero_and_local_work_remains_available(self):
        self.planned(); self.change_budget({'calls': 3, 'cost': {'amount': '1.00', 'currency': 'USD'}})
        self.ctl.reserve_budget({'usage': {'calls': 1, 'cost': {'amount': '.40', 'currency': 'USD'}}, 'purpose': 'Fixture request'}, 'first')
        receipt = self.receipt('unknown')
        self.ctl.settle_budget('first', {'receipt': receipt, 'status': 'unknown', 'usage': {'calls': 1}}, 'unknown')
        view = self.ctl.context()['budget']
        self.assertEqual(['first'], view['unknown_usage'])
        self.assertEqual('0.40', view['reserved']['cost']['USD'])
        with self.assertRaisesRegex(Refusal, 'unknown usage'):
            self.ctl.reserve_budget({'usage': {'calls': 1, 'cost': {'amount': '.10', 'currency': 'USD'}}, 'purpose': 'Another request'}, 'next')
        self.ctl.begin_round()
        receipt = self.receipt('known')
        self.ctl.settle_budget('first', {'receipt': receipt, 'status': 'known', 'usage': {'calls': 1, 'cost': {'amount': '.30', 'currency': 'USD'}}}, 'resolved')
        self.assertEqual([], self.ctl.context()['budget']['unknown_usage'])

    def test_actual_overrun_and_unreserved_calls_are_recorded_not_discarded(self):
        self.planned(); self.change_budget({'calls': 1})
        self.ctl.reserve_budget({'usage': {'calls': 1}, 'purpose': 'Fixture request'}, 'first')
        receipt = self.receipt('actual-overrun')
        self.ctl.settle_budget('first', {'receipt': receipt, 'status': 'known', 'usage': {'calls': 2}}, 'settle-overrun')
        self.assertEqual(2, self.ctl.context()['budget']['known_usage']['calls'])
        with self.assertRaisesRegex(Refusal, 'budget.calls'):
            self.ctl.reserve_budget({'usage': {'calls': 1}, 'purpose': 'Cannot spend more'}, 'blocked')
        self.ctl.settle_budget(None, {'receipt': self.receipt('unreserved'), 'status': 'known', 'usage': {'calls': 1}}, 'unreserved')
        self.assertEqual(3, self.ctl.context()['budget']['known_usage']['calls'])

    def test_budget_increase_needs_a_fresh_answer_to_the_concrete_question(self):
        self.planned()
        plan = self.ctl.load()['budget_confirmation']['plan']['path']
        q = self.ctl.prepare_user_input('budget', {'budget': {'calls': 5}, 'plan': plan, 'reason': 'Proposed fixture increase'})['question']
        with self.assertRaisesRegex(Refusal, 'Actual human response'):
            self.ctl.confirm_budget(None, 'no-answer')
        self.assertEqual({'calls': 0}, self.ctl.context()['budget']['limits'])
        self.assertEqual({'calls': 0}, q['current_budget'])
        self.change_budget({'calls': 5}, 'actual-fixture-answer')
        self.assertEqual({'calls': 5}, self.ctl.context()['budget']['limits'])

    def test_complete_evidence_can_finalize_after_explicit_task_deadline(self):
        result = run(self.root, budget={'calls': 0, 'minutes': 1})
        state = self.ctl.load()
        state['created_at'] = (datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()
        self.ctl.save(state)
        exported = self.ctl.finalize('expired-closeout')
        self.assertEqual(result['status'], exported['status'])
        self.assertEqual('verified', self.ctl.verify_export(exported['manifest'])['status'])

    def test_unreserved_ledger_key_cannot_be_overwritten_by_a_new_reservation(self):
        self.planned(); self.change_budget({'calls': 4})
        value = self.ctl.settle_budget(None, {'receipt':self.receipt('unreserved-first'),'status':'known','usage':{'calls':2}}, 'unreserved-first')
        with self.assertRaisesRegex(Refusal, 'identity already exists'):
            self.ctl.reserve_budget({'usage':{'calls':1},'purpose':'Must preserve actual usage'}, value['reservation'])
        self.assertEqual(2,self.ctl.context()['budget']['known_usage']['calls'])

    def test_new_meter_cannot_turn_old_unreported_usage_into_zero(self):
        self.planned(); self.change_budget({'calls': 4})
        self.ctl.reserve_budget({'usage':{'calls':1},'purpose':'Fixture request'}, 'request')
        self.ctl.settle_budget('request', {'receipt':self.receipt('partial-known'),'status':'known','usage':{'calls':1}}, 'partial-known')
        self.change_budget({'calls':4,'cost':{'amount':'1.00','currency':'USD'}}, 'add-cost-cap')
        with self.assertRaisesRegex(Refusal, 'Historical usage is unknown'):
            self.ctl.reserve_budget({'usage':{'calls':1,'cost':{'amount':'.1','currency':'USD'}},'purpose':'Missing historical costs'}, 'must-wait')
        self.ctl.settle_budget('request', {'receipt':self.receipt('filled-cost'),'status':'known','usage':{'calls':1,'cost':{'amount':'.2','currency':'USD'}}}, 'fill-cost')
        self.assertEqual('0.2', self.ctl.context()['budget']['known_usage']['cost']['USD'])

    def test_partial_known_consumption_cannot_be_erased_as_not_started(self):
        self.planned(); self.change_budget({'calls':4,'tokens':100})
        self.ctl.reserve_budget({'usage':{'calls':1,'tokens':20},'purpose':'Fixture request'}, 'request')
        self.ctl.settle_budget('request', {'receipt':self.receipt('unknown-partial'),'status':'unknown','usage':{'calls':1}}, 'unknown-partial')
        with self.assertRaisesRegex(Refusal, 'nonzero actual usage'):
            self.ctl.settle_budget('request', {'receipt':self.receipt('false-no-dispatch'),'status':'not_started','usage':{}}, 'false-no-dispatch')

    def test_new_cap_waits_for_old_inflight_request_without_that_meter(self):
        self.planned(); self.change_budget({'calls':4})
        self.ctl.reserve_budget({'usage':{'calls':1},'purpose':'Already in flight'}, 'inflight')
        self.change_budget({'calls':4,'cost':{'amount':'1.00','currency':'USD'}}, 'new-cost-cap')
        with self.assertRaisesRegex(Refusal, 'Historical usage is unknown'):
            self.ctl.reserve_budget({'usage':{'calls':1,'cost':{'amount':'1.00','currency':'USD'}},'purpose':'Must not ignore inflight cost'}, 'new-call')
        self.ctl.settle_budget('inflight', {'receipt':self.receipt('inflight-actual'),'status':'known',
            'usage':{'calls':1,'cost':{'amount':'.25','currency':'USD'}}}, 'inflight-actual')
        self.assertEqual({}, self.ctl.context()['budget']['incomplete_capped_usage'])
        self.assertEqual('0.75', self.ctl.context()['budget']['remaining']['cost']['amount'])

    def test_unknown_receipt_can_only_fill_missing_usage_not_reduce_known_calls(self):
        self.planned(); self.change_budget({'calls':5,'tokens':100})
        self.ctl.reserve_budget({'usage':{'calls':3,'tokens':30},'purpose':'Fixture request'}, 'request')
        self.ctl.settle_budget('request', {'receipt':self.receipt('partial-three'),'status':'unknown','usage':{'calls':3}}, 'partial-three')
        with self.assertRaisesRegex(Refusal, 'not rewrite previously known usage'):
            self.ctl.settle_budget('request', {'receipt':self.receipt('false-one'),'status':'known','usage':{'calls':1,'tokens':20}}, 'false-one')
        self.assertEqual(3, self.ctl.context()['budget']['known_usage']['calls'])

    def test_wait_cannot_hide_a_local_order_with_intermediate_outputs(self):
        self.planned()
        put(self.root, 'local/source.txt', 'Available local evidence')
        state = self.ctl.load()
        ids = [item['id'] for item in state['brief']['deliverable_plan']]
        state['additional_orders'].append({'id':'local-chain','producer':'fixture','requirement_ids':ids,'actions':[
            {'type':'copy_artifact','source':'local/source.txt','target':'local/middle.txt'},
            {'type':'copy_artifact','source':'local/middle.txt','target':'local/final.txt'}]})
        self.ctl.save(state)
        self.assertIn('local-chain', self.ctl.available_local_orders(state))
        request = self.ctl.prepare_round([])
        with self.assertRaisesRegex(Refusal, 'local WorkOrders remain executable'):
            self.ctl.apply_round(request, {'request_sha256':request['request_sha256'], 'status':'needs_input',
                'progress':'no_progress','reason':'Must continue local work','blockers':[
                    {'kind':'authorization','reason':'A later external request needs permission','requirement_ids':ids}]}, 'false-global-wait')


if __name__ == '__main__':
    unittest.main()
