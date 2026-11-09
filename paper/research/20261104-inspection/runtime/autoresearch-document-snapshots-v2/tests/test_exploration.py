import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import exploration as ex
from lib.runtime import Refusal, file_hash


class ExplorationTests(unittest.TestCase):
    def test_schedule_and_early_replacement(self):
        self.assertFalse(ex.trigger(iterations=3, budget=12, since_keep=0)['fork'])
        self.assertTrue(ex.trigger(iterations=4, budget=12, since_keep=0)['fork'])
        early = ex.trigger(iterations=5, budget=30, since_keep=5)
        self.assertEqual('early', early['reason'])
        self.assertFalse(ex.trigger(iterations=10, budget=30, since_keep=0, rounds=1)['fork'])
        self.assertTrue(ex.trigger(iterations=20, budget=30, since_keep=0, rounds=1)['fork'])
        self.assertTrue(ex.trigger(iterations=3, budget=6, since_keep=0)['fork'])
        self.assertTrue(ex.trigger(iterations=20, budget=25, since_keep=0, unbounded=True)['fork'])
        self.assertEqual('PLATEAU', ex.trigger(iterations=4, budget=12, since_keep=0, orchestrator=True, plateau=True, rounds=2)['terminal'])
        for flags in ({'ceiling': True}, {'no_fork': True}):
            self.assertFalse(ex.trigger(iterations=4, budget=12, since_keep=6, **flags)['fork'])

    def test_fresh_worktrees_and_real_best_commit_promotion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); repo = root / 'source'; repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git', '-C', str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
            git('init'); git('config', 'user.name', 'Offline fixture'); git('config', 'user.email', 'fixture@example.invalid')
            (repo / 'metric.txt').write_text('3')
            git('add', '.'); git('commit', '-m', 'software fixture baseline')
            incumbent = git('rev-parse', 'HEAD')
            plan = ex.request(goal='Offline numeric fixture', scope=['metric.txt'], metric='score', direction='lower',
                              verify='read fixture score', guard='unchanged evaluator', incumbent=incumbent,
                              results='results.tsv', budget=25, shared=True)
            self.assertFalse(plan['parallel'])
            self.assertEqual(['guided', 'random'], [b['kind'] for b in plan['branches']])
            self.assertNotIn('conversation', str(plan))
            trees = ex.create_worktrees(repo, plan, root / 'branches')
            candidates = []
            for n, tree in enumerate(trees):
                branch = Path(tree['path']); (branch / 'metric.txt').write_text(str(2 - n))
                subprocess.run(['git', '-C', str(branch), 'commit', '-am', 'fixture candidate'], check=True, capture_output=True)
                commit = subprocess.check_output(['git', '-C', str(branch), 'rev-parse', 'HEAD'], text=True).strip()
                observation = {'metric': int((branch / 'metric.txt').read_text()), 'guard_passed': True, 'commit': commit}
                evidence = root / (tree['id'] + '.json'); evidence.write_text(json.dumps(observation))
                candidates.append({**observation, 'id': tree['id'], 'status': 'completed',
                                   'evidence': str(evidence), 'evidence_sha256': file_hash(evidence)})
            self.assertEqual(incumbent, git('rev-parse', 'HEAD'))
            branch = Path(trees[0]['path']); (branch / '.paper').mkdir(); (branch / '.paper/追踪.json').write_text('forged')
            subprocess.run(['git', '-C', str(branch), 'add', '.paper'], check=True)
            subprocess.run(['git', '-C', str(branch), 'commit', '-m', 'unauthorized state'], check=True, capture_output=True)
            forbidden_commit = subprocess.check_output(['git', '-C', str(branch), 'rev-parse', 'HEAD'], text=True).strip()
            bad = dict(candidates[0], commit=forbidden_commit, metric=-1)
            bad_evidence = root / 'bad.json'; bad_evidence.write_text(json.dumps({k:bad[k] for k in ('commit','metric','guard_passed')}))
            bad.update(evidence=str(bad_evidence), evidence_sha256=file_hash(bad_evidence))
            with self.assertRaisesRegex(Refusal, 'controller-owned'):
                ex.promote(repo, plan, [bad, candidates[1]], 3)
            self.assertEqual(incumbent, git('rev-parse', 'HEAD'))
            result = ex.promote(repo, plan, candidates, 3)
            self.assertEqual('b2', result['winner']); self.assertEqual('1', (repo / 'metric.txt').read_text())
            self.assertEqual(incumbent, git('rev-parse', 'refs/tags/autoresearch/incumbent-' + incumbent))
            next_incumbent = git('rev-parse', 'HEAD')
            next_plan = ex.request(goal='Round two', scope=['metric.txt'], metric='score', direction='lower', verify='fixture', guard='scope', incumbent=next_incumbent, results='results.tsv', budget=25, shared=True)
            self.assertEqual(2, len(ex.create_worktrees(repo, next_plan, root / 'round-two')))
            candidates[0]['metric'] = -100
            with self.assertRaisesRegex(Refusal, 'does not match'):
                ex.select(plan, candidates, 3)
            candidates[0]['status'] = 'unknown'
            with self.assertRaisesRegex(Refusal, 'Reconcile'):
                ex.select(plan, candidates, 3)


if __name__ == '__main__':
    unittest.main()
