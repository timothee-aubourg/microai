/* micro.js — shared runtime for the microAI lessons.
   Small helpers for the page, plus a complete GPT forward pass in plain
   JavaScript so the trained model runs live in the browser. No dependencies. */
(function () {
  'use strict';
  const M = (window.M = {});

  M.chapters = [
    ['01-next-token.html', 'Guess the next character', 'next'],
    ['02-tokens.html', 'Cutting text into tokens', 'tokens'],
    ['03-learning.html', 'Learning by rolling downhill', 'learn'],
    ['04-embeddings.html', 'Meaning as direction', 'embed'],
    ['05-attention.html', 'Letting tokens look at each other', 'attend'],
    ['06-transformer.html', 'The whole machine', 'GPT'],
    ['07-training.html', 'Watching it learn', 'train'],
    ['08-sampling.html', 'Rolling the dice', 'sample'],
    ['09-quantization.html', 'Fitting it in your RAM', 'shrink'],
    ['10-local.html', 'A real model on your machine', 'run'],
    ['11-system.html', 'Memory, books and hands', 'system'],
    ['12-adapting.html', 'Teaching new habits', 'adapt'],
  ];

  // ---------- DOM helpers ----------
  M.$ = (s, r) => (r || document).querySelector(s);
  M.el = function (tag, props) {
    const e = document.createElement(tag);
    if (props) for (const k in props) {
      if (k === 'class') e.className = props[k];
      else if (k === 'text') e.textContent = props[k];
      else if (k === 'html') e.innerHTML = props[k];
      else if (k.slice(0, 2) === 'on') e.addEventListener(k.slice(2), props[k]);
      else e.setAttribute(k, props[k]);
    }
    for (let i = 2; i < arguments.length; i++) {
      const kid = arguments[i];
      if (kid != null) e.appendChild(typeof kid === 'string' ? document.createTextNode(kid) : kid);
    }
    return e;
  };
  M.css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  M.fmt = (x, n) => (Math.abs(x) >= 1e4 || (x !== 0 && Math.abs(x) < 1e-3) ? x.toExponential(1) : x.toFixed(n == null ? 2 : n));
  M.show = (c) => (c === '\n' ? '⏎' : c === ' ' ? '␣' : c);
  M.reduced = () => window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let uid = 0;
  M.id = (p) => (p || 'm') + '-' + ++uid;

  // ---------- maths ----------
  M.rng = function (seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  M.gauss = (rnd) => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  M.softmax = function (z, temp) {
    const t = temp || 1, n = z.length, p = new Float64Array(n);
    let mx = -Infinity, s = 0;
    for (let i = 0; i < n; i++) if (z[i] > mx) mx = z[i];
    for (let i = 0; i < n; i++) { p[i] = Math.exp((z[i] - mx) / t); s += p[i]; }
    for (let i = 0; i < n; i++) p[i] /= s;
    return p;
  };
  M.pick = function (p, rnd) {
    let r = rnd(), i = 0;
    for (; i < p.length - 1; i++) { r -= p[i]; if (r < 0) break; }
    return i;
  };
  M.order = (p) => Array.from(p.keys()).sort((a, b) => p[b] - p[a]);
  /* temperature / top-k / top-p, exactly as in code/ch08_sample.py */
  M.shape = function (logits, o) {
    const n = logits.length;
    if (o.temperature === 0) {
      const p = new Float64Array(n); let b = 0;
      for (let i = 1; i < n; i++) if (logits[i] > logits[b]) b = i;
      p[b] = 1; return p;
    }
    const z = Array.from(logits, (v) => v / o.temperature);
    if (o.top_k && o.top_k < n) {
      const kth = z.slice().sort((a, b) => b - a)[o.top_k - 1];
      for (let i = 0; i < n; i++) if (z[i] < kth) z[i] = -Infinity;
    }
    const p = M.softmax(z);
    if (o.top_p && o.top_p < 1) {
      const ord = M.order(p); let before = 0, s = 0;
      for (const i of ord) { if (before >= o.top_p) p[i] = 0; else s += p[i]; before += p[i] || 0; if (p[i] === 0) before = 1; }
      for (let i = 0; i < n; i++) p[i] /= s;
    }
    return p;
  };

  // ---------- controls ----------
  M.slider = function (o) {
    const id = M.id('s');
    const out = M.el('output', { for: id });
    const inp = M.el('input', { type: 'range', id: id, min: o.min, max: o.max, step: o.step || 1, value: o.value });
    const fmt = o.fmt || ((v) => v);
    const upd = () => { out.textContent = fmt(+inp.value); };
    inp.addEventListener('input', () => { upd(); o.oninput && o.oninput(+inp.value); });
    upd();
    const el = M.el('div', { class: 'ctl' }, M.el('label', { for: id, text: o.label }), inp, out);
    return { el: el, get value() { return +inp.value; }, set value(v) { inp.value = v; upd(); } };
  };
  M.button = (label, fn, cls) => M.el('button', { type: 'button', class: cls || '', text: label, onclick: fn });
  M.select = function (label, options, fn) {
    const id = M.id('sel');
    const sel = M.el('select', { id: id });
    options.forEach((o) => sel.appendChild(M.el('option', { value: o[0], text: o[1] })));
    sel.addEventListener('change', () => fn(sel.value));
    return { el: M.el('div', { class: 'ctl', style: 'flex:0 1 auto;min-width:0' }, M.el('label', { for: id, text: label }), sel), sel: sel };
  };
  M.row = function () {
    const r = M.el('div', { class: 'row' });
    for (const a of arguments) if (a) r.appendChild(a.el || a);
    return r;
  };
  /* probability bars: items = [{label, p, hl, cut}] */
  M.bars = function (box, items, max) {
    box.textContent = '';
    box.className = 'bars';
    const mx = max || Math.max.apply(null, items.map((i) => i.p).concat([1e-9]));
    items.forEach((it) => {
      const cls = (it.hl ? ' hl' : '') + (it.cut ? ' cut' : '');
      box.appendChild(M.el('span', { class: 'lab' + cls, text: it.label }));
      const bar = M.el('span', { class: 'bar' + cls }, M.el('i', { style: 'width:' + (100 * it.p / mx).toFixed(1) + '%' }));
      box.appendChild(bar);
      box.appendChild(M.el('span', { class: 'num' + cls, text: it.text || (100 * it.p).toFixed(1) + '%' }));
    });
  };

  // ---------- canvas ----------
  const redraws = [];
  M.canvas = function (parent, height, draw, touch) {
    const c = M.el('canvas', touch ? { class: 'touch' } : null);
    const wrap = M.el('div', { class: 'cv' }, c);
    parent.appendChild(wrap);
    const o = { c: c, ctx: c.getContext('2d'), w: 0, h: 0 };
    o.redraw = function () {
      const dpr = window.devicePixelRatio || 1, w = wrap.clientWidth || 300;
      const h = typeof height === 'function' ? height(w) : height;
      if (o.w !== w || o.h !== h || c.width !== Math.round(w * dpr)) {
        o.w = w; o.h = h;
        c.style.width = w + 'px'; c.style.height = h + 'px';
        c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
      }
      o.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      o.ctx.clearRect(0, 0, w, h);
      draw(o);
    };
    redraws.push(o.redraw);
    return o;
  };
  M.pointer = function (o, fn) {       // fn(x, y, phase) in CSS pixels
    let down = false;
    const pos = (e) => { const r = o.c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    o.c.addEventListener('pointerdown', (e) => { down = true; o.c.setPointerCapture(e.pointerId); const p = pos(e); fn(p[0], p[1], 'down'); e.preventDefault(); });
    o.c.addEventListener('pointermove', (e) => { if (down) { const p = pos(e); fn(p[0], p[1], 'move'); } });
    const up = () => { if (down) { down = false; fn(0, 0, 'up'); } };
    o.c.addEventListener('pointerup', up); o.c.addEventListener('pointercancel', up);
  };
  let rt = 0;
  const redrawAll = () => { clearTimeout(rt); rt = setTimeout(() => redraws.forEach((f) => f()), 60); };
  window.addEventListener('resize', redrawAll);
  if (window.matchMedia) { const mq = window.matchMedia('(prefers-color-scheme: dark)'); mq.addEventListener && mq.addEventListener('change', redrawAll); }
  new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  /* a small line chart. series: [{pts:[[x,y],...], color, width, dots}] */
  M.plot = function (o, cfg) {
    const ctx = o.ctx, L = cfg.left || 44, R = 22, T = 12, B = cfg.xlabel ? 38 : 24;
    const w = o.w - L - R, h = o.h - T - B;
    const lx = cfg.logx ? Math.log10 : (v) => v;
    const x0 = lx(cfg.xmin), x1 = lx(cfg.xmax);
    const px = (x) => L + ((lx(x) - x0) / (x1 - x0)) * w;
    const py = (y) => T + (1 - (y - cfg.ymin) / (cfg.ymax - cfg.ymin)) * h;
    ctx.font = '12px ' + M.css('--sans'); ctx.lineWidth = 1;
    ctx.strokeStyle = M.css('--line'); ctx.fillStyle = M.css('--muted');
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    (cfg.yticks || []).forEach((t) => { ctx.beginPath(); ctx.moveTo(L, py(t)); ctx.lineTo(L + w, py(t)); ctx.stroke(); ctx.fillText(String(t), L - 6, py(t)); });
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    (cfg.xticks || []).forEach((t) => { ctx.beginPath(); ctx.moveTo(px(t), T + h); ctx.lineTo(px(t), T + h + 4); ctx.stroke(); ctx.fillText(cfg.xfmt ? cfg.xfmt(t) : String(t), px(t), T + h + 6); });
    if (cfg.xlabel) ctx.fillText(cfg.xlabel, L + w / 2, T + h + 22);
    if (cfg.ylabel) { ctx.save(); ctx.translate(11, T + h / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText(cfg.ylabel, 0, 0); ctx.restore(); }
    (cfg.hlines || []).forEach((hl) => {
      ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = hl.color || M.css('--muted');
      ctx.beginPath(); ctx.moveTo(L, py(hl.y)); ctx.lineTo(L + w, py(hl.y)); ctx.stroke(); ctx.restore();
      ctx.textAlign = hl.left ? 'left' : 'right'; ctx.textBaseline = hl.below ? 'top' : 'bottom'; ctx.fillStyle = hl.color || M.css('--muted');
      ctx.fillText(hl.label, hl.left ? L + 4 : L + w - 2, py(hl.y) + (hl.below ? 3 : -2));
    });
    ctx.save(); ctx.beginPath(); ctx.rect(L, T - 2, w, h + 4); ctx.clip();
    cfg.series.forEach((s) => {
      ctx.strokeStyle = ctx.fillStyle = s.color; ctx.lineWidth = s.width || 2; ctx.lineJoin = 'round';
      if (!s.nolines) { ctx.beginPath(); s.pts.forEach((p, i) => (i ? ctx.lineTo(px(p[0]), py(p[1])) : ctx.moveTo(px(p[0]), py(p[1])))); ctx.stroke(); }
      if (s.dots) s.pts.forEach((p) => { ctx.beginPath(); ctx.arc(px(p[0]), py(p[1]), s.dots, 0, 7); ctx.fill(); });
    });
    ctx.restore();
    return { px: px, py: py, L: L, T: T, w: w, h: h, inv: (X) => { const u = x0 + ((X - L) / w) * (x1 - x0); return cfg.logx ? Math.pow(10, u) : u; } };
  };
  /* blend a CSS colour toward the panel colour: used for heat shading */
  M.heat = function (v, varName) {
    const pct = Math.round(100 * Math.max(0, Math.min(1, v)));
    return 'color-mix(in srgb, ' + M.css(varName || '--fwd') + ' ' + pct + '%, ' + M.css('--soft') + ')';
  };

  // ---------- the model ----------
  function f16(h) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    if (e === 0) return s * 6.103515625e-5 * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  let cached = null;
  M.loadModel = function () {
    if (cached) return cached;
    const W = window.MICRO_WEIGHTS;
    if (!W) return null;
    const bin = atob(W.b64), n = bin.length / 2, all = new Float32Array(n);
    for (let i = 0; i < n; i++) all[i] = f16(bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8));
    const w = {}, shape = {};
    W.tensors.forEach((t) => {
      const size = t.shape.reduce((a, b) => a * b, 1);
      w[t.name] = all.subarray(t.offset, t.offset + size); shape[t.name] = t.shape;
    });
    const stoi = {};
    for (let i = 0; i < W.chars.length; i++) stoi[W.chars[i]] = i;
    cached = { cfg: W.config, chars: W.chars, stoi: stoi, w: w, shape: shape, adapter: W.adapter };
    return cached;
  };
  M.encode = (m, text) => Array.from(text).filter((c) => c in m.stoi).map((c) => m.stoi[c]);

  function matmul(x, T, din, W, dout) {          // x: T x din, W: din x dout  ->  T x dout
    const out = new Float32Array(T * dout), acc = new Float64Array(dout);
    for (let t = 0; t < T; t++) {
      const xo = t * din, oo = t * dout;
      acc.fill(0);
      for (let i = 0; i < din; i++) {
        const xv = x[xo + i], wo = i * dout;
        for (let j = 0; j < dout; j++) acc[j] += xv * W[wo + j];
      }
      for (let j = 0; j < dout; j++) out[oo + j] = acc[j];
    }
    return out;
  }
  function layernorm(x, T, d, g, b) {
    const out = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      let mean = 0; for (let i = 0; i < d; i++) mean += x[t * d + i]; mean /= d;
      let v = 0; for (let i = 0; i < d; i++) { const c = x[t * d + i] - mean; v += c * c; }
      const inv = 1 / Math.sqrt(v / d + 1e-5);
      for (let i = 0; i < d; i++) out[t * d + i] = (x[t * d + i] - mean) * inv * g[i] + b[i];
    }
    return out;
  }
  /* The same computation as GPT.forward in code/gpt.py.
     ids: token ids (at most cfg.block). weights: optional replacement for m.w.
     Returns logits (T x vocab) and, if trace is set, attention maps and residuals. */
  M.forward = function (m, ids, weights, trace) {
    const w = weights || m.w, c = m.cfg, d = c.d_model, H = c.n_head, hd = d / H, V = c.vocab, T = ids.length;
    let x = new Float32Array(T * d);
    for (let t = 0; t < T; t++) for (let i = 0; i < d; i++) x[t * d + i] = w.wte[ids[t] * d + i] + w.wpe[t * d + i];
    const tr = trace ? { attn: [], resid: [x.slice()] } : null;
    const rs = 1 / Math.sqrt(hd);
    for (let l = 0; l < c.n_layer; l++) {
      const p = 'h' + l + '.';
      if (trace && trace.skip && trace.skip[l]) { tr.attn.push(null); tr.resid.push(x.slice()); continue; }   // experiment: leave a block out
      const a = layernorm(x, T, d, w[p + 'ln1.g'], w[p + 'ln1.b']);
      const q = matmul(a, T, d, w[p + 'attn.wq'], d), k = matmul(a, T, d, w[p + 'attn.wk'], d), v = matmul(a, T, d, w[p + 'attn.wv'], d);
      const mixed = new Float32Array(T * d), maps = [];
      for (let h = 0; h < H; h++) {
        const map = trace ? new Float32Array(T * T) : null, o = h * hd, sc = new Float64Array(T);
        for (let t = 0; t < T; t++) {
          let mx = -Infinity;
          for (let s = 0; s <= t; s++) {
            let dot = 0; for (let j = 0; j < hd; j++) dot += q[t * d + o + j] * k[s * d + o + j];
            sc[s] = dot * rs; if (sc[s] > mx) mx = sc[s];
          }
          let sum = 0; for (let s = 0; s <= t; s++) { sc[s] = Math.exp(sc[s] - mx); sum += sc[s]; }
          for (let s = 0; s <= t; s++) {
            const aw = sc[s] / sum; if (map) map[t * T + s] = aw;
            for (let j = 0; j < hd; j++) mixed[t * d + o + j] += aw * v[s * d + o + j];
          }
        }
        if (map) maps.push(map);
      }
      const att = matmul(mixed, T, d, w[p + 'attn.wo'], d);
      for (let i = 0; i < T * d; i++) x[i] += att[i];
      const hid = matmul(layernorm(x, T, d, w[p + 'ln2.g'], w[p + 'ln2.b']), T, d, w[p + 'mlp.w1'], 4 * d);
      for (let i = 0; i < hid.length; i++) { const u = hid[i]; hid[i] = 0.5 * u * (1 + Math.tanh(0.7978845608028654 * (u + 0.044715 * u * u * u))); }
      const mo = matmul(hid, T, 4 * d, w[p + 'mlp.w2'], d);
      for (let i = 0; i < T * d; i++) x[i] += mo[i];
      if (tr) { tr.attn.push(maps); tr.resid.push(x.slice()); }
    }
    const out = { logits: M.unembed(m, x, T, w), T: T };
    if (tr) { out.attn = tr.attn; out.resid = tr.resid; }
    return out;
  };
  M.unembed = function (m, x, T, weights) {      // final layernorm, then compare with every token embedding
    const w = weights || m.w, d = m.cfg.d_model, V = m.cfg.vocab;
    const f = layernorm(x, T, d, w['ln_f.g'], w['ln_f.b']), logits = new Float32Array(T * V);
    for (let t = 0; t < T; t++) for (let v = 0; v < V; v++) {
      let dot = 0; for (let i = 0; i < d; i++) dot += f[t * d + i] * w.wte[v * d + i];
      logits[t * V + v] = dot;
    }
    return logits;
  };
  M.last = (m, out) => out.logits.subarray((out.T - 1) * m.cfg.vocab, out.T * m.cfg.vocab);
  /* mean cross-entropy (nats per character) of text under the model */
  M.loss = function (m, windows, weights) {     // windows: strings of block+1 characters
    const V = m.cfg.vocab;
    let total = 0, n = 0;
    windows.forEach(function (text) {
      const ids = M.encode(m, text), T = ids.length - 1;
      const out = M.forward(m, ids.slice(0, T), weights);
      for (let t = 0; t < T; t++) {
        const row = out.logits.subarray(t * V, (t + 1) * V);
        let mx = -Infinity; for (let v = 0; v < V; v++) if (row[v] > mx) mx = row[v];
        let sum = 0; for (let v = 0; v < V; v++) sum += Math.exp(row[v] - mx);
        total += -(row[ids[t + 1]] - mx - Math.log(sum)); n++;
      }
    });
    return total / n;
  };
  /* round-to-nearest with one scale per block, as in code/ch09_quantize.py */
  M.quantize = function (arr, bits, block) {
    const out = new Float32Array(arr.length), B = block || arr.length;
    for (let s = 0; s < arr.length; s += B) {
      const e = Math.min(arr.length, s + B);
      let mx = 0, mean = 0;
      for (let i = s; i < e; i++) { const a = Math.abs(arr[i]); if (a > mx) mx = a; mean += a; }
      if (bits === 1) { mean /= e - s; for (let i = s; i < e; i++) out[i] = arr[i] > 0 ? mean : arr[i] < 0 ? -mean : 0; continue; }
      const qmax = Math.pow(2, bits - 1) - 1, scale = mx / qmax || 1e-8;
      for (let i = s; i < e; i++) out[i] = Math.max(-qmax, Math.min(qmax, Math.round(arr[i] / scale))) * scale;
    }
    return out;
  };
  M.quantizeModel = function (m, bits, block) {
    const w = {};
    for (const k in m.w) w[k] = k.indexOf('lora.') === 0 ? m.w[k] : m.shape[k].length === 2 && bits < 16 ? M.quantize(m.w[k], bits, block) : m.w[k];
    return w;
  };
  /* fold the LoRA adapter into the weights: W + scale * A @ B */
  M.withAdapter = function (m) {
    const w = {}, r = m.adapter.rank, sc = m.adapter.scale;
    for (const k in m.w) {
      const A = m.w['lora.' + k + '.A'], Bm = m.w['lora.' + k + '.B'];
      if (!A) { w[k] = m.w[k]; continue; }
      const din = m.shape[k][0], dout = m.shape[k][1], W = m.w[k].slice();
      for (let i = 0; i < din; i++) for (let j = 0; j < r; j++) {
        const a = A[i * r + j] * sc; if (a === 0) continue;
        for (let o = 0; o < dout; o++) W[i * dout + o] += a * Bm[j * dout + o];
      }
      w[k] = W;
    }
    return w;
  };
  /* One token at a time with a KV cache, as in code/ch08_sample.py: the keys and
     values of earlier tokens are stored, so a new token costs one row of work. */
  M.session = function (m, weights) {
    const w = weights || m.w, c = m.cfg, d = c.d_model, H = c.n_head, hd = d / H, L = c.n_layer, B = c.block, rs = 1 / Math.sqrt(hd);
    const K = [], Vc = [];
    for (let l = 0; l < L; l++) { K.push(new Float32Array(B * d)); Vc.push(new Float32Array(B * d)); }
    let pos = 0, hist = [];
    function feed(id) {
      const x = new Float32Array(d);
      for (let i = 0; i < d; i++) x[i] = w.wte[id * d + i] + w.wpe[pos * d + i];
      for (let l = 0; l < L; l++) {
        const p = 'h' + l + '.';
        const a = layernorm(x, 1, d, w[p + 'ln1.g'], w[p + 'ln1.b']);
        const q = matmul(a, 1, d, w[p + 'attn.wq'], d);            // only the new token asks
        K[l].set(matmul(a, 1, d, w[p + 'attn.wk'], d), pos * d);  // its key and value join the cache
        Vc[l].set(matmul(a, 1, d, w[p + 'attn.wv'], d), pos * d);
        const mixed = new Float32Array(d), sc = new Float64Array(pos + 1);
        for (let h = 0; h < H; h++) {
          const o = h * hd; let mx = -Infinity, sum = 0;
          for (let s = 0; s <= pos; s++) { let dot = 0; for (let j = 0; j < hd; j++) dot += q[o + j] * K[l][s * d + o + j]; sc[s] = dot * rs; if (sc[s] > mx) mx = sc[s]; }
          for (let s = 0; s <= pos; s++) { sc[s] = Math.exp(sc[s] - mx); sum += sc[s]; }
          for (let s = 0; s <= pos; s++) { const aw = sc[s] / sum; for (let j = 0; j < hd; j++) mixed[o + j] += aw * Vc[l][s * d + o + j]; }
        }
        const att = matmul(mixed, 1, d, w[p + 'attn.wo'], d);
        for (let i = 0; i < d; i++) x[i] += att[i];
        const hid = matmul(layernorm(x, 1, d, w[p + 'ln2.g'], w[p + 'ln2.b']), 1, d, w[p + 'mlp.w1'], 4 * d);
        for (let i = 0; i < hid.length; i++) { const u = hid[i]; hid[i] = 0.5 * u * (1 + Math.tanh(0.7978845608028654 * (u + 0.044715 * u * u * u))); }
        const mo = matmul(hid, 1, 4 * d, w[p + 'mlp.w2'], d);
        for (let i = 0; i < d; i++) x[i] += mo[i];
      }
      pos++;
      return M.unembed(m, x, 1, w);
    }
    return {
      push: function (id) {                 // returns the scores for the token after `id`
        hist.push(id);
        if (pos < B) return feed(id);
        pos = 0;                            // window full: keep the recent half and rebuild the cache
        let lg = null;
        hist.slice(-(B >> 1)).forEach((t) => { lg = feed(t); });
        return lg;
      }
    };
  };
  /* a text being written: feed the prompt, then add one token at a time */
  M.writer = function (m, prompt, weights) {
    const s = M.session(m, weights), ids = M.encode(m, prompt);
    if (!ids.length) ids.push(m.stoi['\n']);
    let lg = null;
    ids.forEach((id) => { lg = s.push(id); });
    return { logits: () => lg, add: (id) => { lg = s.push(id); } };
  };
  /* generate n characters; cb(char, probs) per step. Returns the new text. */
  M.generate = function (m, prompt, n, opts, rnd, weights, cb) {
    const wr = M.writer(m, prompt, weights);
    let text = '';
    for (let i = 0; i < n; i++) {
      const p = M.shape(wr.logits(), opts), nx = M.pick(p, rnd);
      wr.add(nx); text += m.chars[nx];
      if (cb) cb(m.chars[nx], p);
    }
    return text;
  };

  // ---------- page chrome ----------
  function highlight(pre) {
    const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const kw = /^(def|class|return|for|in|if|else|elif|while|import|from|as|lambda|not|and|or|is|None|True|False|break|continue|try|except|with|yield|global|raise|assert)$/;
    const re = /(#.*$)|("""[\s\S]*?"""|f?'(?:\\.|[^'\\\n])*'|f?"(?:\\.|[^"\\\n])*")|\b(\d+\.?\d*(?:e-?\d+)?)\b|\b([A-Za-z_]\w*)\b/gm;
    const src = pre.textContent; let out = '', last = 0, mm;
    while ((mm = re.exec(src))) {
      out += esc(src.slice(last, mm.index));
      if (mm[1]) out += '<span class="c">' + esc(mm[1]) + '</span>';
      else if (mm[2]) out += '<span class="s">' + esc(mm[2]) + '</span>';
      else if (mm[3]) out += '<span class="n">' + esc(mm[3]) + '</span>';
      else out += kw.test(mm[4]) ? '<span class="k">' + mm[4] + '</span>' : mm[4];
      last = re.lastIndex;
    }
    pre.innerHTML = out + esc(src.slice(last));
  }
  function chrome() {
    const ch = +document.body.getAttribute('data-ch') || 0;
    const trail = M.el('nav', { class: 'trail', 'aria-label': 'Chapters' });
    M.chapters.forEach((c, i) => {
      if (i === 6 || i === 9) trail.appendChild(M.el('span', { class: 'gap' }));
      const a = M.el('a', { href: c[0], title: (i + 1) + '. ' + c[1], class: i + 1 === ch ? 'here' : i + 1 < ch ? 'done' : '', text: String(i + 1) });
      if (i + 1 === ch) a.setAttribute('aria-current', 'page');
      trail.appendChild(a);
    });
    const top = M.el('header', { class: 'top' }, M.el('div', { class: 'top-in' }, M.el('a', { class: 'brand', href: 'index.html', html: 'micro<span>AI</span>' }), trail));
    document.body.insertBefore(top, document.body.firstChild);
    const main = M.$('main');
    if (ch && main) {
      const where = M.$('.where');
      if (where) where.textContent = 'Chapter ' + ch + ' of ' + M.chapters.length;
      const prev = M.chapters[ch - 2], next = M.chapters[ch];
      const pager = M.el('nav', { class: 'pager', 'aria-label': 'Previous and next chapter' });
      pager.appendChild(prev ? M.el('a', { href: prev[0], html: '<small>Previous</small>' + prev[1] }) : M.el('a', { href: 'index.html', html: '<small>Back to</small>Contents' }));
      pager.appendChild(next ? M.el('a', { class: 'next', href: next[0], html: '<small>Next</small>' + next[1] }) : M.el('a', { class: 'next', href: 'index.html', html: '<small>Back to</small>Contents' }));
      main.appendChild(pager);
    }
    document.querySelectorAll('pre.code').forEach((p) => { if (p.getAttribute('data-lang') !== 'text') highlight(p); });
  }
  M.ready = function (fn) {
    const go = () => { try { fn(); } catch (e) { console.error(e); } };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
  };
  M.ready(chrome);
})();
