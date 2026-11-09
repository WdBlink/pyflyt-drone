"""Explicit software fixtures; never modify a real research ledger."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
ORIGINAL = PACKAGE.parent / 'autoresearch-original'
sys.path.insert(0, str(PACKAGE))
from lib.runtime import Controller, Refusal, digest, file_hash, read_json
from tests.test_runtime import brief


class ReaderMigrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()
        self.original = self.root / 'runtime/original'
        shutil.copytree(ORIGINAL, self.original, ignore=shutil.ignore_patterns('__pycache__'))
        script = '''import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from lib.runtime import Controller
import json
root=Path(sys.argv[2]);ctl=Controller(root)
out=ctl.init(json.loads(sys.argv[3]));ctl.approve(out['brief_sha256'],'software fixture')
ctl.begin_round(session_id='reader-test');ctl.prepare_round([])
state=ctl.load();state['software_test_padding']='x'*(17*1024*1024);ctl.save(state)
'''
        initialized = subprocess.run([sys.executable, '-c', script, str(self.original), str(self.root), json.dumps(brief())],
                                     capture_output=True, text=True)
        self.assertEqual(initialized.returncode,0,initialized.stderr)
        self.ctl = Controller(self.root, PACKAGE)
        self.before = read_json(self.ctl.path, max_bytes=128 * 1024 * 1024)
        review = dict(decision='approve_state_reader_migration',reviewer='software-fixture-reviewer',
            from_implementation=self.before['implementation_sha256'],to_implementation=self.ctl.implementation(),
            preserves_contract=True,open_findings=[])
        self.review = self.root / 'review.json'
        self.review.write_text(json.dumps(review))
        self.packet = dict(kind='controller-json-reader-v1',from_implementation=review['from_implementation'],
            to_implementation=review['to_implementation'],prior_package='runtime/original',
            review=dict(path='review.json',sha256=file_hash(self.review)))

    def test_actual_old_cli_refuses_oversize_fixture_and_new_migration_preserves_all_fields(self):
        old = subprocess.run([sys.executable,str(self.original/'scripts/research.py'),'--run',str(self.root),'status'],
                             capture_output=True,text=True)
        self.assertEqual(old.returncode,2)
        self.assertIn('16 MiB',old.stdout)
        result=self.ctl.migrate_state_reader(self.packet,'migrate-reader-test')
        after=self.ctl.load()
        excluded={'implementation_sha256','pending_round','history','operations'}
        self.assertEqual({k:v for k,v in self.before.items() if k not in excluded},
                         {k:v for k,v in after.items() if k not in excluded})
        self.assertEqual(after['history'][:len(self.before['history'])],self.before['history'])
        self.assertIsNone(after['pending_round'])
        self.assertEqual(after['active_round'],self.before['active_round'])
        self.assertEqual(result,self.ctl.migrate_state_reader(self.packet,'migrate-reader-test'))
        self.assertEqual(after,self.ctl.load())
        self.ctl.prepare_round([])

    def test_reader_keeps_external_packet_limit_and_strict_json_rules(self):
        with self.assertRaisesRegex(Refusal,'16 MiB'):
            read_json(self.ctl.path)
        for text,expected in (('{"a":1,"a":2}','Duplicate'),('{"a":NaN}','Non-finite')):
            path=self.root/'packet.json';path.write_text(text)
            with self.assertRaisesRegex(Refusal,expected):read_json(path)
            with self.assertRaisesRegex(Refusal,expected):read_json(path,max_bytes=128*1024*1024)
        large=self.root/'huge.json'
        with large.open('wb') as stream:stream.truncate(129*1024*1024)
        with self.assertRaisesRegex(Refusal,'128 MiB'):read_json(large,max_bytes=128*1024*1024)

    def test_wrong_target_review_or_prior_snapshot_never_changes_state(self):
        original_hash=file_hash(self.ctl.path)
        for change in ('target','review','prior'):
            packet=copy.deepcopy(self.packet)
            if change=='target':packet['to_implementation']='0'*64
            elif change=='review':packet['review']['sha256']='0'*64
            else:packet['from_implementation']='0'*64
            with self.assertRaises(Refusal):self.ctl.migrate_state_reader(packet,'bad-'+change)
            self.assertEqual(file_hash(self.ctl.path),original_hash)


if __name__=='__main__':unittest.main()
