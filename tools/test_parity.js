// test_parity.js — does the browser model compute the same numbers as the Python model?
//   node tools/test_parity.js
const path = require('path'), root = path.join(__dirname, '..', 'docs', 'assets');
global.window = { addEventListener() {} };
global.document = { readyState: 'loading', addEventListener() {}, documentElement: {} };
global.MutationObserver = class { observe() {} };
global.atob = (s) => Buffer.from(s, 'base64').toString('binary');
require(path.join(root, 'data.js')); require(path.join(root, 'weights.js')); require(path.join(root, 'micro.js'));
const M = window.M, D = window.MICRO_DATA, m = M.loadModel();

const out = M.forward(m, M.encode(m, D.parity.prompt)), last = M.last(m, out);
let worst = 0;
D.parity.logits.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - last[i])); });
console.log('next-character scores, browser vs Python: max difference', worst.toExponential(1));

const wr = M.writer(m, D.parity.prompt), lg = wr.logits();
let worst2 = 0;
for (let i = 0; i < last.length; i++) worst2 = Math.max(worst2, Math.abs(lg[i] - last[i]));
console.log('KV-cache path vs full forward pass: max difference', worst2.toExponential(1));
if (worst2 > 1e-3) { console.error('MISMATCH'); process.exit(1); }
let tg = Date.now(); const txt = M.generate(m, 'ROMEO:\n', 300, { temperature: 0.8 }, M.rng(1));
console.log('300 characters generated in', Date.now() - tg, 'ms:', JSON.stringify(txt.slice(0, 60)));

let t0 = Date.now();
const loss = M.loss(m, D.windows);
console.log('held-out loss: browser', loss.toFixed(4), ' Python', D.heldout_loss.toFixed(4), ' (' + (Date.now() - t0) + ' ms)');

const q4 = M.loss(m, D.windows, M.quantizeModel(m, 4, 32));
console.log('4-bit, block 32: browser', q4.toFixed(4));
if (m.adapter) {
  const caps = D.windows.map((w) => w.toUpperCase());
  console.log('adapter: loss on capitalised text', M.loss(m, caps).toFixed(3), '->', M.loss(m, caps, M.withAdapter(m)).toFixed(3));
}
if (worst > 2e-3 || Math.abs(loss - D.heldout_loss) > 2e-3) { console.error('MISMATCH'); process.exit(1); }
console.log('ok');
