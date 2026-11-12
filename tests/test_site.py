"""Run tools/check_site.py as a test, so slf's merge gate checks the website."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_site():
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'check_site.py')],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
