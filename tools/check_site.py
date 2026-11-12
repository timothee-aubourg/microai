"""Check the course website in docs/ the way a reader would meet it.

Checks, each reported on its own line:
  links     every local href/src in docs/*.html points at a file that exists
  glossary  every glossary link (class="term") names an entry in assets/glossary.js
  parity    tools/test_parity.js: the browser model matches the Python model
  browser   every page loads in headless Chromium without a console error
            (skipped, and reported as skipped, when no Chromium is found;
            set CHROME=/path/to/chrome to choose one)

Exit status is 0 only if every check that ran passed.

    python3 tools/check_site.py
"""
import glob
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, 'docs')
PAGES = sorted(glob.glob(os.path.join(DOCS, '*.html')))


def check_links():
    bad = []
    for page in PAGES:
        html = open(page, encoding='utf-8').read()
        for target in re.findall(r'\s(?:href|src)="([^"#]+)', html):
            if re.match(r'[a-z]+:', target):          # http:, https:, mailto:, data:
                continue
            if not os.path.exists(os.path.join(DOCS, target)):
                bad.append(f'{os.path.basename(page)} -> {target}')
    return bad


def check_glossary():
    path = os.path.join(DOCS, 'assets', 'glossary.js')
    src = open(path, encoding='utf-8').read() if os.path.exists(path) else ''
    keys = set(re.findall(r"^\s*'([a-z0-9-]+)':\s*\{", src, re.M))
    bad = []
    for page in PAGES:
        html = open(page, encoding='utf-8').read()
        for key in re.findall(r'class="term" href="glossary\.html#([^"]+)"', html):
            if key not in keys:
                bad.append(f'{os.path.basename(page)} -> #{key}')
    return bad


def check_parity():
    node = shutil.which('node')
    if not node:
        return None
    r = subprocess.run([node, os.path.join(ROOT, 'tools', 'test_parity.js')], capture_output=True, text=True, timeout=300)
    return [] if r.returncode == 0 and r.stdout.strip().endswith('ok') else [(r.stdout + r.stderr).strip()[-400:]]


def find_chrome():
    if os.environ.get('CHROME'):
        return os.environ['CHROME']
    for name in ('chromium', 'chromium-browser', 'google-chrome', 'chrome-headless-shell'):
        if shutil.which(name):
            return shutil.which(name)
    found = sorted(glob.glob(os.path.expanduser('~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell')))
    return found[-1] if found else None


def check_browser():
    chrome = find_chrome()
    if not chrome:
        return None
    bad = []
    for page in PAGES:
        r = subprocess.run([chrome, '--no-sandbox', '--disable-gpu', '--allow-file-access-from-files',
                            '--enable-logging=stderr', '--v=0', '--virtual-time-budget=5000',
                            '--dump-dom', 'file://' + page], capture_output=True, text=True, timeout=120)
        for line in r.stderr.splitlines():
            if ':CONSOLE' in line or 'Uncaught' in line:
                bad.append(f'{os.path.basename(page)}: {line.strip()[:200]}')
    return bad


def main():
    failed = False
    for name, fn in (('links', check_links), ('glossary', check_glossary), ('parity', check_parity), ('browser', check_browser)):
        problems = fn()
        if problems is None:
            print(f'{name:9} skipped (tool not found)')
            continue
        print(f'{name:9} {"ok" if not problems else "FAIL"}')
        for p in problems:
            print('          ' + p)
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
