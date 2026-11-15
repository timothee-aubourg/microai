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


# Reading time of each chapter, at 220 words a minute, for the map, the paths and the chapter headers.
# Counted on the prose and code of <main>; the scripts that draw the experiments are left out.
def words(page):
    with open(page, encoding='utf-8') as f:
        src = f.read()
    body = src[src.find('<main>'):src.find('</main>')]
    text = re.sub(r'<svg.*?</svg>', ' ', body, flags=re.S)
    return len(re.sub(r'<[^>]+>', ' ', text).split())

chapters = sorted(glob.glob(os.path.join(ROOT, 'docs', '[0-9][0-9]-*.html')))
minutes = [max(1, round(words(p) / 220)) for p in chapters]
js = os.path.join(ROOT, 'docs', 'assets', 'micro.js')
with open(js, encoding='utf-8') as f:
    src = f.read()
out = re.sub(r'(// \[minutes\]\n).*?(\n[ \t]*// \[/minutes\])', lambda m: f'{m.group(1)}  M.minutes = {minutes};{m.group(2)}'.replace(', ', ','), src, flags=re.S)
if out == src and 'M.minutes = [' not in src:
    sys.exit('minutes marker not found in micro.js')
with open(js, 'w', encoding='utf-8') as f:
    f.write(out)
print(f'counted reading time for {len(minutes)} chapters, {sum(minutes)} minutes in all')
