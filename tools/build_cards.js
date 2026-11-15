/* build_cards.js: draw the picture shown when a page of the course is shared (X, Mastodon, a chat app).

    node tools/build_cards.js        # writes docs/cards/01.jpg … 38.jpg, index.jpg, map.jpg

One 1200 × 630 image per chapter, in the look of the pages: the chapter's title, its opening question,
its part and its reading time. It needs Playwright and a Chromium (`npm i playwright` if missing).
tools/build_site.py points each page's link preview at these files. Re-run after changing a title,
a question or the reading time. */
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const DOCS = path.join(__dirname, '..', 'docs');
const OUT = path.join(DOCS, 'cards');

const question = (file) => {
  const m = fs.readFileSync(path.join(DOCS, file), 'utf8').match(/<p class="question">([\s\S]*?)<\/p>/);
  return m ? m[1].replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim() : '';
};

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1, colorScheme: 'light' });
  await page.goto('file://' + path.join(DOCS, 'glossary.html'));           // loads the style, the fonts and M
  await page.waitForFunction(() => window.M && M.chapters);
  const cards = await page.evaluate(() => M.chapters.map((c, i) => {
    const k = M.partOf(i + 1);
    return { file: c[0], name: String(i + 1).padStart(2, '0'), title: c[1], kicker: 'Chapter ' + (i + 1) + ' of ' + M.chapters.length + ' · ' + M.parts[k][1],
             foot: (M.minutes[i] || '?') + ' min read · runs on your phone', part: k };
  }));
  cards.forEach((c) => { c.q = question(c.file); });
  cards.push({ name: 'index', title: 'A language model you can hold in your head', kicker: 'A free course in 38 short chapters',
               q: 'Build a language model and the agent around it, from first principles, with experiments that run in your browser.', foot: 'Start with the map · 10 min' });
  cards.push({ name: 'map', title: 'The map of an agent', kicker: 'Start here',
               q: 'What is an agent made of, and which part do you need today?', foot: 'A live agent, every part on one page, three ways through' });

  for (const c of cards) {
    await page.evaluate((c) => {
      document.documentElement.setAttribute('data-theme', 'light');
      document.body.innerHTML = '';
      if (!document.getElementById('nograin')) document.head.appendChild(M.el('style', { id: 'nograin', text: 'html::after{display:none!important}' }));
      document.body.style.cssText = 'margin:0;width:1200px;height:630px;overflow:hidden;background:var(--paper)';
      const card = M.el('div', { id: 'card', style: 'box-sizing:border-box;width:1200px;height:630px;padding:64px 80px 56px;display:flex;flex-direction:column;background:var(--paper);background-image:var(--wash);color:var(--ink)' });
      const top = M.el('div', { style: 'display:flex;align-items:center;gap:18px;font:600 26px var(--sans);color:var(--muted)' },
        M.el('span', { style: 'font:700 34px var(--display);color:var(--ink)', html: 'micro<span style="color:var(--fwd)">AI</span>' }),
        M.el('span', { text: '·' }), M.el('span', { text: c.kicker }));
      if (c.part !== undefined) { const ic = M.partIcon(c.part); ic.style.cssText = 'width:48px;height:48px;display:inline-block'; top.insertBefore(ic, top.children[2]); }
      const big = c.title.length > 34 ? 66 : 78;
      const title = M.el('h1', { text: c.title, style: 'font:700 ' + big + 'px/1.05 var(--display);letter-spacing:-0.01em;margin:40px 0 0;max-width:1040px;text-wrap:balance' });
      const stroke = M.el('div', { style: 'width:150px;height:14px;margin:18px 0 0;background:var(--brush) center/100% 100% no-repeat' });
      const q = M.el('p', { text: c.q, style: 'font:italic 400 ' + (c.q.length > 120 ? 30 : 34) + 'px/1.35 var(--display);border-left:5px solid var(--fwd);padding-left:24px;margin:34px 0 0;max-width:1000px' });
      const foot = M.el('div', { text: c.foot, style: 'margin-top:auto;font:600 24px var(--sans);color:var(--fwd)' });
      card.append(top, title, stroke, q, foot);
      document.body.appendChild(card);
    }, c);
    await page.evaluate(() => document.fonts.ready);
    const fits = await page.evaluate(() => { const e = document.getElementById('card'); return e.scrollHeight <= 630; });
    if (!fits) console.warn('card ' + c.name + ' overflows: shorten its question or title');
    await page.locator('#card').screenshot({ path: path.join(OUT, c.name + '.jpg'), type: 'jpeg', quality: 86 });
  }
  await browser.close();
  console.log('drew ' + cards.length + ' cards into docs/cards/');
})();
