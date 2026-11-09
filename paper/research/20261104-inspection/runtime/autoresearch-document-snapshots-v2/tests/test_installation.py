import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless((ROOT / 'scripts/install.sh').is_file(), 'Installer test runs from the complete source repository')
class InstallationTests(unittest.TestCase):
    def test_installed_fixture_survives_deletion_of_source_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); source = base / 'source'; config = base / 'config'
            (source / 'scripts').mkdir(parents=True)
            shutil.copyfile(ROOT / 'scripts/install.sh', source / 'scripts/install.sh')
            shutil.copytree(ROOT / '.agents/skills/autoresearch', source / '.agents/skills/autoresearch')
            install = subprocess.run(['bash', str(source / 'scripts/install.sh'), '--codex', '--global', '--config-dir', str(config), '--force'],
                                     cwd=base, capture_output=True, text=True)
            self.assertEqual(0, install.returncode, install.stdout + install.stderr)
            shutil.rmtree(source)
            package = config / 'skills/autoresearch/paper'
            self.assertTrue((package / 'references/packets.md').is_file())
            result = subprocess.run([sys.executable, str(package / 'tests/run_fixture.py'), '--run', str(base / 'run')], cwd=base,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            value = json.loads(result.stdout)
            self.assertEqual('software_fixture_complete', value['status'])
            self.assertTrue(Path(value['pdf']).read_bytes().startswith(b'%PDF-'))
            self.assertTrue((config / 'skills/autoresearch/LICENSE').is_file())
            self.assertFalse(source.exists())


if __name__ == '__main__': unittest.main()
