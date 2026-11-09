import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.runtime import Refusal, digest, file_hash
from lib.user_input import check_human_response, prepare, validate_response


class UserInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = {'brief': {'budget': {'calls': 5}, 'question': 'Software contract check'},
                      'created_at': '2026-11-04T00:00:00+00:00', 'history': []}
        self.state['brief_sha256'] = digest(self.state['brief'])
        (self.root / 'plan.json').write_text('{"items":[]}', encoding='utf-8')
        (self.root / 'estimate.md').write_text('Synthetic estimate: two explicit calls.', encoding='utf-8')
        self.budget = {'budget': {'calls': 8, 'cost': {'amount': '2.50', 'currency': 'USD'}},
                       'plan': 'plan.json', 'reason': 'Confirm the synthetic plan and resource limits.',
                       'estimate': 'estimate.md'}
        self.venue = {'software_fixture': True, 'venue': 'Synthetic journal A', 'artifacts': ['rules.md'],
                      'venue_selection': {'mode': 'candidate_comparison', 'selected_id': 'V1',
                                          'criteria': ['scope fit', 'resources'],
                                          'rationale': 'A recommendation for the software fixture only.',
                                          'candidates': []}}
        (self.root / 'rules.md').write_text('Synthetic author rules for both candidates.', encoding='utf-8')
        for number in (1, 2):
            self.venue['venue_selection']['candidates'].append({
                'id': 'V' + str(number), 'venue': 'Synthetic journal ' + ('A' if number == 1 else 'B'),
                'kind': 'journal', 'scope_fit': 'Software contract verification.',
                'article_type_fit': 'Original software fixture.', 'constraints': 'No submission.',
                'advantages': ['A fixture comparison.'], 'limitations': ['Not a real venue.'],
                'cost': {'status': 'unknown', 'basis': 'No real fees investigated.'},
                'timing': {'status': 'unknown', 'basis': 'No real timing investigated.'},
                'evidence': [{'path': 'rules.md', 'sha256': file_hash(self.root / 'rules.md'),
                              'url': 'https://synthetic.example/rules', 'quote': 'Synthetic author rules'}]})

    def response(self, question, answer, quote='This is a synthetic explicit answer for software verification.'):
        (self.root / 'reply.md').write_text(quote + '\n', encoding='utf-8')
        return {'question_sha256': question['question_sha256'], 'answer': answer,
                'human_response': {'path': 'reply.md', 'sha256': file_hash(self.root / 'reply.md'), 'quote': quote}}

    def test_questions_are_deterministic_unanswered_and_do_not_mutate_or_write(self):
        before = copy.deepcopy((self.state, self.budget, self.venue))
        files = {path.name: path.read_bytes() for path in self.root.iterdir()}
        for kind, packet in [('budget', self.budget), ('venue', self.venue)]:
            question = prepare(kind, packet, self.root, self.state)
            self.assertEqual(question, prepare(kind, packet, self.root, self.state))
            self.assertEqual('needs_input', question['status'])
            self.assertNotIn('answer', question)
            self.assertIn('请用户回答', question['summary'])
            with self.assertRaises(Refusal): validate_response(question, None, self.root, self.state)
        self.assertEqual(before, (self.state, self.budget, self.venue))
        self.assertEqual(files, {path.name: path.read_bytes() for path in self.root.iterdir()})

    def test_venue_answer_can_choose_an_unrecommended_candidate(self):
        question = prepare('venue', self.venue, self.root, self.state)
        answer = {'candidate_id': 'V2'}
        self.assertEqual('V1', question['recommendation']['candidate_id'])
        self.assertEqual({'V1', 'V2'}, {option['id'] for option in question['options']})
        self.assertEqual(answer, validate_response(question, self.response(question, answer), self.root, self.state))
        self.assertEqual(['scope fit', 'resources'], question['criteria']['values'])
        self.assertEqual(self.venue['venue_selection']['candidates'], question['candidates'])
        self.assertIn(file_hash(self.root / 'rules.md'), question['summary'])

    def test_unknown_or_changed_venue_inputs_cannot_reuse_answer(self):
        question = prepare('venue', self.venue, self.root, self.state)
        with self.assertRaisesRegex(Refusal, 'candidate IDs'):
            validate_response(question, self.response(question, {'candidate_id': 'V3'}), self.root, self.state)
        response = self.response(question, {'candidate_id': 'V2'})
        (self.root / 'rules.md').write_text('Changed rules. Synthetic author rules remain quoted.', encoding='utf-8')
        with self.assertRaises(ValueError): validate_response(question, response, self.root, self.state)

    def test_question_digest_and_reply_binding_cannot_be_forged(self):
        question = prepare('venue', self.venue, self.root, self.state)
        response = self.response(question, {'candidate_id': 'V2'})
        response['question_sha256'] = '0' * 64
        with self.assertRaisesRegex(Refusal, 'current question'):
            validate_response(question, response, self.root, self.state)
        forged = copy.deepcopy(question); forged['recommendation']['candidate_id'] = 'V2'
        forged['question_sha256'] = digest({key: value for key, value in forged.items() if key != 'question_sha256'})
        with self.assertRaisesRegex(Refusal, 'Question inputs'):
            validate_response(forged, self.response(forged, {'candidate_id': 'V2'}), self.root, self.state)

    def test_budget_question_displays_current_proposal_default_units_and_sources(self):
        question = prepare('budget', self.budget, self.root, self.state)
        self.assertEqual({'calls': 5}, question['current_budget'])
        self.assertEqual(self.budget['budget'], question['proposal'])
        self.assertEqual({}, question['default_budget'])
        self.assertEqual({'default', 'custom', 'decline'}, {option['id'] for option in question['options']})
        for value in ['无隐含任务总上限', 'token', 'USD', '十进制字符串', '包含等待', '30 分钟仅为检查点提示',
                      file_hash(self.root / 'plan.json'), file_hash(self.root / 'estimate.md')]:
            self.assertIn(value, question['summary'])
        self.assertFalse(question['checkpoint']['is_task_limit'])
        self.assertTrue(question['time_scope']['cross_session'])
        self.assertEqual(digest({}), question['usage_ledger_sha256'])
        self.assertEqual({'calls': 0, 'tokens': 0, 'cost': {}}, question['usage']['known_usage'])

    def test_budget_question_shows_known_usage_and_unresolved_reservations(self):
        self.state['usage_ledger'] = {
            'finished': {'status': 'known', 'reported_usage': {'calls': 2, 'tokens': 400,
                         'cost': {'amount': '0.40', 'currency': 'USD'}}},
            'pending': {'status': 'unknown', 'reserved': {'calls': 3, 'tokens': 1000,
                        'cost': {'amount': '2.00', 'currency': 'USD'}},
                        'reported_usage': {'calls': 1, 'tokens': 200,
                        'cost': {'amount': '0.25', 'currency': 'USD'}}}}
        question = prepare('budget', self.budget, self.root, self.state)
        self.assertEqual({'calls': 3, 'tokens': 600, 'cost': {'USD': '0.65'}}, question['usage']['known_usage'])
        self.assertEqual({'calls': 2, 'tokens': 800, 'cost': {'USD': '1.75'}}, question['usage']['reserved'])
        self.assertEqual(['pending'], question['usage']['unresolved_usage'])
        self.assertEqual(['pending'], question['usage']['unknown_usage'])
        for text in ['账本已知消耗', '当前预留', '0.65', '1.75', 'pending', digest(self.state['usage_ledger'])]:
            self.assertIn(text, question['summary'])

    def test_budget_question_binds_ledger_but_ignores_clock_changes(self):
        self.state['brief']['budget']['minutes'] = 30
        self.state['brief_sha256'] = digest(self.state['brief'])
        with patch('lib.budget.elapsed', return_value=1):
            question = prepare('budget', self.budget, self.root, self.state)
        with patch('lib.budget.elapsed', return_value=60):
            self.assertEqual(question, prepare('budget', self.budget, self.root, self.state))
            response = self.response(question, {'choice': 'default'})
            self.assertTrue(validate_response(question, response, self.root, self.state)['proceed'])
        self.state['usage_ledger'] = {'pending': {'status': 'reserved', 'reserved': {'calls': 1}}}
        with self.assertRaisesRegex(Refusal, 'Question inputs'):
            validate_response(question, response, self.root, self.state)

    def test_budget_default_custom_and_decline_have_distinct_results(self):
        question = prepare('budget', self.budget, self.root, self.state)
        for answer, budget, proceed in [({'choice': 'default'}, {}, True),
                                        ({'choice': 'custom', 'budget': {'calls': 0}}, {'calls': 0}, True),
                                        ({'choice': 'decline'}, {'calls': 5}, False)]:
            actual = validate_response(question, self.response(question, answer), self.root, self.state)
            self.assertEqual({'choice': answer['choice'], 'budget': budget, 'proceed': proceed}, actual)
        self.assertEqual({'calls': 5}, self.state['brief']['budget'])

    def test_budget_answer_never_uses_a_proposal_as_an_implicit_answer(self):
        question = prepare('budget', self.budget, self.root, self.state)
        for answer in [{}, {'choice': 'custom'}, {'choice': 'custom', 'budget': []},
                       {'choice': 'default', 'budget': self.budget['budget']}, {'choice': 'yes'}, {'choice': []}]:
            with self.subTest(answer=answer), self.assertRaises(Refusal):
                validate_response(question, self.response(question, answer), self.root, self.state)

    def test_changed_plan_estimate_or_brief_requires_a_current_question(self):
        question = prepare('budget', self.budget, self.root, self.state)
        response = self.response(question, {'choice': 'default'})
        for name in ['plan.json', 'estimate.md']:
            original = (self.root / name).read_bytes()
            (self.root / name).write_bytes(original + b'\n')
            with self.subTest(name=name), self.assertRaisesRegex(Refusal, 'Question inputs'):
                validate_response(question, response, self.root, self.state)
            (self.root / name).write_bytes(original)
        changed = copy.deepcopy(self.state); changed['brief']['budget'] = {'calls': 9}
        changed['brief_sha256'] = digest(changed['brief'])
        with self.assertRaisesRegex(Refusal, 'Question inputs'):
            validate_response(question, response, self.root, changed)

    def test_human_source_is_rechecked_independently_after_brief_changes(self):
        question = prepare('budget', self.budget, self.root, self.state)
        response = self.response(question, {'choice': 'custom', 'budget': {'calls': 8}})
        validate_response(question, response, self.root, self.state)
        self.state['brief']['budget'] = {'calls': 8}; self.state['brief_sha256'] = digest(self.state['brief'])
        self.assertEqual(response['human_response'], check_human_response(response['human_response'], self.root))
        (self.root / 'reply.md').write_text('Changed answer.', encoding='utf-8')
        with self.assertRaisesRegex(Refusal, 'SHA-256'): check_human_response(response['human_response'], self.root)

    def test_source_needs_exact_nonempty_quote_and_real_utf8_bytes(self):
        question = prepare('budget', self.budget, self.root, self.state)
        source = self.response(question, {'choice': 'default'})['human_response']
        for quote in ['', 'A phrase absent from the file.']:
            with self.subTest(quote=quote), self.assertRaises(Refusal):
                check_human_response({**source, 'quote': quote}, self.root)
        (self.root / 'reply.md').write_bytes(b'\xff')
        source['sha256'] = file_hash(self.root / 'reply.md')
        with self.assertRaisesRegex(Refusal, 'UTF-8'): check_human_response(source, self.root)

    def test_source_path_boundaries_apply_to_questions_and_answers(self):
        question = prepare('budget', self.budget, self.root, self.state)
        source = self.response(question, {'choice': 'default'})['human_response']
        for name in ['../reply.md', str(self.root / 'reply.md'), '.paper/control.json']:
            with self.subTest(path=name), self.assertRaises(Refusal):
                check_human_response({**source, 'path': name}, self.root)
        (self.root / 'linked.md').symlink_to(self.root / 'reply.md')
        with self.assertRaises(Refusal): check_human_response({**source, 'path': 'linked.md'}, self.root)
        bad = {**self.budget, 'plan': '../plan.json'}
        with self.assertRaises(Refusal): prepare('budget', bad, self.root, self.state)

    def test_delegation_schema_cannot_replace_a_bound_human_response(self):
        question = prepare('budget', self.budget, self.root, self.state)
        response = self.response(question, {'choice': 'default'}, quote='Choose a topic and run the research.')
        delegated = {'mode': 'delegated', **response['human_response'], 'scope': 'Old broad delegation.'}
        with self.assertRaisesRegex(Refusal, 'delegated'):
            validate_response(question, {'question_sha256': question['question_sha256'], 'answer': {'choice': 'default'},
                                         'authorization': delegated}, self.root, self.state)
        with self.assertRaisesRegex(Refusal, 'delegated'):
            validate_response(question, {**response, 'mode': 'delegated'}, self.root, self.state)
        with self.assertRaisesRegex(Refusal, 'delegated'):
            check_human_response(delegated, self.root)

    def test_existing_explicit_answer_source_does_not_attest_identity_or_meaning(self):
        question = prepare('budget', self.budget, self.root, self.state)
        response = self.response(question, {'choice': 'default'},
                                 quote='Software fixture reply: use the default budget {} for this plan.')
        delegated = {'mode': 'delegated', **response['human_response'], 'scope': 'Earlier authorization from the same source.'}
        self.state['history'].append({'event': 'brief_superseded', 'approval': {'authorization': delegated}})
        self.assertTrue(validate_response(question, response, self.root, self.state)['proceed'])
        self.assertIn('not human identity or free-text meaning', question['limits'])

    def test_same_explicit_answer_can_be_bound_to_the_current_question(self):
        question = prepare('budget', self.budget, self.root, self.state)
        response = self.response(question, {'choice': 'default'})
        self.assertEqual(validate_response(question, response, self.root, self.state),
                         validate_response(question, response, self.root, self.state))
        revised = {**self.budget, 'reason': 'A revised proposal requires an explicitly bound current answer.'}
        current = prepare('budget', revised, self.root, self.state)
        with self.assertRaisesRegex(Refusal, 'current question'):
            validate_response(current, response, self.root, self.state)
        response['question_sha256'] = current['question_sha256']
        self.assertTrue(validate_response(current, response, self.root, self.state)['proceed'])


if __name__ == '__main__': unittest.main()
