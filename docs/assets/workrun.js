/* workrun.js: the harness of chapters 39 to 42, for the agent trained by code/ch39_plan.py.
   The same steps as run_steps and run_plan in that file: the model writes, greedily and inside
   a format, and this code runs the tools and writes the results back into the context.
   Load it after micro.js; it needs window.MICRO_WORK (written by tools/export_web.py). */
(function () {
  'use strict';
  const D = '0123456789', NAMES = ['add', 'sub', 'mul'];
  const TOOLS = { add: (a, b) => a + b, sub: (a, b) => a - b, mul: (a, b) => a * b };
  const CALL = [{ words: NAMES }, [' ', 1, 1], ['-', 0, 1], [D, 1, 4], [' ', 1, 1], ['-', 0, 1], [D, 1, 4], ['\n', 1, 1]];
  const STEP = [{ words: NAMES }, [' ', 1, 1], ['$-', 0, 1], [D, 1, 4], [' ', 1, 1], [D, 1, 2], { words: [', ', '\n'] }];
  const ROLE = [{ words: ['C ', 'A '] }];

  function moves(fmt, st) {        // the format machine of chapter 13
    let i = st[0], s = st[1], mayEnd;
    const out = {};
    while (i < fmt.length) {
      const part = fmt[i];
      if (part.words) {
        part.words.forEach((w) => { if (w.indexOf(s) === 0 && w.length > s.length && !(w[s.length] in out)) out[w[s.length]] = [i, s + w[s.length]]; });
        mayEnd = part.words.indexOf(s) >= 0;
      } else {
        if (s.length < part[2]) for (let k = 0; k < part[0].length; k++) if (!(part[0][k] in out)) out[part[0][k]] = [i, s + part[0][k]];
        mayEnd = s.length >= part[1];
      }
      if (!mayEnd) break;
      i++; s = '';
    }
    return out;
  }

  function Work(m) {
    this.m = m;
  }
  Work.prototype.session = function (prompt) {
    const m = this.m, wr = M.writer(m, prompt);
    return { wr: wr, used: prompt.length, text: prompt };
  };
  Work.prototype.line = function (s, fmt, limit) {     // the model writes until the end of a line, greedily
    const m = this.m;
    let line = '', st = [0, ''];
    while (line.slice(-1) !== '\n' && line.length < (limit || 80) && s.used < m.cfg.block) {
      const z = s.wr.logits();
      let legal = null, best = -1;
      if (fmt) { legal = moves(fmt, st); if (!Object.keys(legal).length) break; }
      for (let i = 0; i < z.length; i++) if ((!legal || m.chars[i] in legal) && (best < 0 || z[i] > z[best])) best = i;
      const ch = m.chars[best];
      if (fmt) st = legal[ch];
      line += ch; s.wr.add(best); s.used++; s.text += ch;
    }
    return line;
  };
  Work.prototype.feed = function (s, text) {
    for (const c of text) { s.wr.add(this.m.stoi[c]); s.used++; s.text += c; }
  };
  /* Step by step. onCall(name, a, b) returns the text of the result (a number, or an error). */
  Work.prototype.steps = function (prompt, onCall, maxCalls) {
    const s = this.session(prompt), calls = [];
    onCall = onCall || ((n, a, b) => String(TOOLS[n](a, b)));
    try {
      while (s.used < this.m.cfg.block - 24 && calls.length < (maxCalls || 8)) {
        const role = this.line(s, ROLE, 2);
        if (role === 'A ') {
          const t = this.line(s, null, 40).replace(/[.\n]+$/, '');
          return { answer: /^-?\d+$/.test(t) ? +t : null, calls: calls, read: s.used, text: s.text, stopped: null };
        }
        const call = this.line(s, CALL, 20).trim(), p = call.split(' ');
        const r = onCall(p[0], +p[1], +p[2], calls);
        calls.push([call, r]);
        this.feed(s, 'R ' + r + '\n');
      }
      return { answer: null, calls: calls, read: s.used, text: s.text, stopped: 'the window is full' };
    } catch (e) {
      if (e && e.stop) return { answer: null, calls: calls, read: s.used, text: s.text, stopped: e.stop };
      throw e;
    }
  };
  /* Plan first: one line of steps, then the harness runs them. */
  Work.prototype.plan = function (prompt) {
    const s = this.session(prompt + 'P '), plan = [], results = [];
    s.text = prompt + 'P ';
    for (;;) {
      const step = this.line(s, STEP, 24);
      plan.push(step.replace(/[,\s]+$/, ''));
      if (step.slice(-2) !== ', ' || plan.length > 8) break;
    }
    for (const st of plan) {
      const p = st.split(' ');
      let a = p[1];
      if (a[0] === '$') { const k = +a.slice(1); if (!(k >= 1 && k <= results.length)) return { answer: null, plan: plan, results: results, read: s.used, text: s.text }; a = results[k - 1]; } else a = +a;
      results.push(TOOLS[p[0]](a, +p[2]));
    }
    return { answer: results[results.length - 1], plan: plan, results: results, read: s.used, text: s.text };
  };
  Work.TOOLS = TOOLS;
  /* A chain of `links` operations on numbers below 100, worked left to right; every value kept below 10,000. */
  Work.chain = function (rnd, links) {
    const words = { plus: 'add', minus: 'sub', times: 'mul' }, W = Object.keys(words);
    for (;;) {
      const nums = [], ops = [];
      for (let i = 0; i <= links; i++) nums.push(Math.floor(rnd() * 100));
      for (let i = 0; i < links; i++) ops.push(W[Math.floor(rnd() * 3)]);
      let v = nums[0], ok = true;
      const values = [];
      ops.forEach((o, i) => { v = TOOLS[words[o]](v, nums[i + 1]); values.push(v); if (Math.abs(v) >= 10000) ok = false; });
      if (ok) return { nums: nums, ops: ops, values: values, q: 'what is ' + nums[0] + ops.map((o, i) => ' ' + o + ' ' + nums[i + 1]).join('') + '?' };
    }
  };
  M.Work = Work;
})();
