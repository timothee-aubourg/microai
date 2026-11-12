"""
build_site.py — copy code snippets from code/*.py into the lesson pages.

    python tools/build_site.py

In the HTML, <pre class="code" data-src="gpt.py#attention"></pre> is filled with
the region of code/gpt.py between `# [snippet:attention]` and `# [/snippet]`.
The pages therefore always show the code that actually runs. Re-run after
editing any file in code/.
"""
import glob, html, os, re, sys, textwrap

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def snippet(file, name):
    with open(os.path.join(ROOT, 'code', file), encoding='utf-8') as f:
        src = f.read()
    m = re.search(r'^[ \t]*# \[snippet:' + re.escape(name) + r'\]\n(.*?)^[ \t]*# \[/snippet\]', src, re.S | re.M)
    if not m:
        sys.exit(f'snippet not found: {file}#{name}')
    return textwrap.dedent(m.group(1)).rstrip()


def fill(match):
    file, name = match.group(2), match.group(3)
    return f'{match.group(1)}{html.escape(snippet(file, name), quote=False)}</pre>'


count = 0
for page in sorted(glob.glob(os.path.join(ROOT, 'docs', '*.html'))):
    with open(page, encoding='utf-8') as f:
        src = f.read()
    out, n = re.subn(r'(<pre class="code" data-src="([^"#]+)#([^"]+)">)(.*?)</pre>', fill, src, flags=re.S)
    count += n
    if out != src:
        with open(page, 'w', encoding='utf-8') as f:
            f.write(out)
print(f'filled {count} code snippets')
