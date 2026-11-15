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


# ---------- link previews: what X, Mastodon or a chat app shows when a page is shared ----------
# The manifest, icon and theme colour are added by micro.js, which knows the reader's light or dark theme.
# Each page gets a description, a canonical address and an Open Graph / X card pointing at its image in
# docs/cards/ (drawn by tools/build_cards.js). The address of the published site comes from MICROAI_SITE.
SITE = os.environ.get('MICROAI_SITE', 'https://timothee-aubourg.github.io/microai/').rstrip('/') + '/'
with open(js, encoding='utf-8') as f:
    src = f.read()
src = re.sub(r"(// \[site\]\n).*?(\n[ \t]*// \[/site\])", lambda m: f"{m.group(1)}  M.site = '{SITE}';{m.group(2)}", src, flags=re.S)
with open(js, 'w', encoding='utf-8') as f:
    f.write(src)

blurbs = [b.replace("\\'", "'") for b in re.findall(r"^    '((?:[^'\\]|\\.)*)',?$", src[src.find('M.blurbs = ['):src.find('];', src.find('M.blurbs = ['))], re.M)]
assert len(blurbs) == len(chapters), (len(blurbs), len(chapters))
OTHER = {'index.html': ('microAI: a language model you can hold in your head', 'Build and understand a language model and the agent around it, from first principles: 43 short interactive chapters that run on your phone.', 'index'),
         'map.html': ('The map of an agent · microAI', 'What an agent is made of, on one page: a live agent in your browser, every part with its chapters, and three ways through.', 'map'),
         'glossary.html': ('Glossary · microAI', 'Every technical term of the microAI course in plain words, with the chapter that introduces it.', 'index')}


def social(page):
    name = os.path.basename(page)
    with open(page, encoding='utf-8') as f:
        src = f.read()
    if name in OTHER:
        title, desc, card = OTHER[name]
    else:
        n = int(name[:2])
        title = re.search(r'<title>(.*?) \| microAI</title>', src).group(1) + ' · microAI'
        desc, card = blurbs[n - 1], f'{n:02d}'
    e = lambda t: html.escape(t, quote=True)
    url = SITE + ('' if name == 'index.html' else name)
    block = '\n'.join([
        '<!-- [social] written by tools/build_site.py -->',
        f'<meta name="description" content="{e(desc)}">',
        f'<link rel="canonical" href="{url}">',
        '<meta property="og:site_name" content="microAI">',
        f'<meta property="og:type" content="{"website" if name in OTHER else "article"}">',
        f'<meta property="og:title" content="{e(title)}">',
        f'<meta property="og:description" content="{e(desc)}">',
        f'<meta property="og:url" content="{url}">',
        f'<meta property="og:image" content="{SITE}cards/{card}.jpg">',
        '<meta property="og:image:width" content="1200">',
        '<meta property="og:image:height" content="630">',
        f'<meta property="og:image:alt" content="{e(title)}">',
        '<meta name="twitter:card" content="summary_large_image">',
        '<!-- [/social] -->'])
    out = re.sub(r'<meta name="description"[^>]*>\n', '', src)
    if '<!-- [social]' in out:
        out = re.sub(r'<!-- \[social\].*?<!-- \[/social\] -->', lambda m: block, out, flags=re.S)
    else:
        out = re.sub(r'(<title>.*?</title>\n)', lambda m: m.group(1) + block + '\n', out, count=1)
    if out != src:
        with open(page, 'w', encoding='utf-8') as f:
            f.write(out)


for page in sorted(glob.glob(os.path.join(ROOT, 'docs', '*.html'))):
    social(page)
print(f'wrote link previews for {SITE}')
