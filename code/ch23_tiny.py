"""
ch23_tiny.py — the agent of chapter 19, made as small as it will go.   (Chapter 23)

    python code/ch23_tiny.py --sweep        # train every candidate and grade it (about an hour)
    python code/ch23_tiny.py --train        # train the chosen one and pack it into runs/tiny.bin
    python code/ch23_tiny.py                # grade the packed model on every held-out question
    python code/ch23_tiny.py --check        # the same questions again through the no-numpy engine
    python code/tiny_agent.py "what is 23 times 45?"   # run it with no numpy at all

Same task, same harness and same test as chapter 19. What changes is the model:

    ternary       every weight is -1, 0 or +1 times one trained number per matrix,
                  and the model is TRAINED that way (rounding afterwards fails)
    no tables     positions come from a formula; the normalisation has no weights
    packed        five such weights per byte in the file
    width         measured, not assumed: --sweep trains narrower models too,
                  and none of them answers every question
"""
import argparse, json, math, os, sys, time
import numpy as np
from engine import Tensor
from gpt import GPT
from ch08_sample import Runner
from ch09_quantize import quantize
import ch19_system as big

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
BIN, SWEEP, FACTS = (os.path.join(RUNS, f) for f in ('tiny.bin', 'tiny_sweep.json', 'tiny.json'))
WIDTH = 73                                   # the longest conversation, plus the line break before it


# [snippet:ternary]
def ternary(w):
    """Round a matrix to -1, 0 or +1 times ONE number. Returns the three-valued matrix and that number."""
    scale = np.abs(w).mean() + 1e-8
    return np.clip(np.round(w / scale), -1, 1), scale
# [/snippet]


def positions(block, d):
    """Position vectors from a formula, so none need storing: sines and cosines of different speeds."""
    pos, k = np.arange(block)[:, None], np.arange(d // 2)[None, :]
    angle = pos / (50.0 ** (2 * k / d))
    return np.concatenate([np.sin(angle), np.cos(angle)], axis=1)


class TinyGPT(GPT):
    def __init__(self, vocab, block, n_layer, n_head, d, mlp=4, three=True, seed=0):
        super().__init__(vocab, block, n_layer, n_head, d, seed=seed)
        rng = np.random.default_rng(seed + 1)
        for i in range(n_layer):                                  # the MLP may be narrower than 4 x d
            self.p[f'h{i}.mlp.w1'] = Tensor(rng.normal(0, 0.02, (d, mlp * d)))
            self.p[f'h{i}.mlp.w2'] = Tensor(rng.normal(0, 0.02 / math.sqrt(2 * n_layer), (mlp * d, d)))
        self.mlp, self.three = mlp, three
        self.where = Tensor(positions(block, d))                  # fixed: a formula, not weights
        # what is trained and stored: the matrices. Normalisation gains stay 1 and offsets 0.
        self.stored = [k for k in self.p if self.p[k].data.ndim == 2 and k != 'wpe']
        # one trained number per matrix: how big its -1s and +1s are. And one for the positions.
        self.size = {k: Tensor(np.array([ternary(self.p[k].data)[1]])) for k in self.stored} if three else {}
        self.size['wpe'] = Tensor(np.array([0.02]))

    def parameters(self):
        return [self.p[k] for k in self.stored] + list(self.size.values())

    # [snippet:qat]
    def forward(self, idx, targets=None, trace=None):
        """Compute with the three-valued weights, learn on the full-precision ones underneath."""
        latent = self.p
        self.p = dict(latent)
        self.p['wpe'] = self.where * self.size['wpe']
        for k in (self.stored if self.three else []):
            q, _ = ternary(latent[k].data)
            three = latent[k] + (q - latent[k].data)               # value: -1, 0 or +1. gradient: passes straight to latent
            self.p[k] = three * self.size[k]                       # times the one number of this matrix
        try:
            return super().forward(idx, targets, trace)
        finally:
            self.p = latent
    # [/snippet]

    def weights(self):
        """The arrays the runtime computes with."""
        out = {k: t.data.astype(np.float32) for k, t in self.p.items()}
        out['wpe'] = (self.where.data * float(self.size['wpe'].data[0])).astype(np.float32)
        if self.three:
            for k in self.stored:
                out[k] = (ternary(self.p[k].data)[0] * float(self.size[k].data[0])).astype(np.float32)
        return out


def save_npz(path, weights, cfg):
    np.savez(path, __chars__=np.array(list(big.CHARS)),
             __config__=np.array([len(big.CHARS), cfg['block'], cfg['layers'], cfg['heads'], cfg['d']]), **weights)


# ---- the file: five three-valued weights per byte -----------------------------------
ORDER = lambda L: ['wte'] + [f'h{i}.{m}' for i in range(L) for m in ('attn.wq', 'attn.wk', 'attn.wv', 'attn.wo', 'mlp.w1', 'mlp.w2')]

# [snippet:pack]
def pack(weights, cfg):
    """The whole model as bytes: a short header, then per matrix one float and its weights, five to a byte."""
    chars = ''.join(big.CHARS)
    out = bytearray(b'uAI3' + bytes([len(chars), cfg['block'], cfg['layers'], cfg['heads'], cfg['d'], cfg['mlp']]))
    out += chars.encode('ascii') + np.float32(np.abs(weights['wpe']).max()).tobytes()   # how big the positions are
    for name in ORDER(cfg['layers']):
        scale = np.abs(weights[name]).max()                          # the weights are already -scale, 0 or +scale
        trits = (np.round(weights[name] / scale).reshape(-1) + 1).astype(np.int64)   # 0, 1 or 2
        trits = np.concatenate([trits, np.ones((-len(trits)) % 5, dtype=np.int64)])
        packed = trits.reshape(-1, 5) @ np.array([1, 3, 9, 27, 81])  # 3**5 = 243 values fit in one byte
        out += np.float32(scale).tobytes() + packed.astype(np.uint8).tobytes()
    return bytes(out)
# [/snippet]

def unpack(blob):
    """Read the file back into arrays (the numpy reader; code/tiny_agent.py reads it without numpy)."""
    assert blob[:4] == b'uAI3'
    V, block, L, H, d, mlp = blob[4:10]
    chars, at = blob[10:10 + V].decode('ascii'), 14 + V
    where = np.frombuffer(blob[10 + V:14 + V], np.float32)[0]
    shapes = {'wte': (V, d), 'attn.wq': (d, d), 'attn.wk': (d, d), 'attn.wv': (d, d), 'attn.wo': (d, d), 'mlp.w1': (d, mlp * d), 'mlp.w2': (mlp * d, d)}
    w = {'wpe': (positions(block, d) * where).astype(np.float32), 'ln_f.g': np.ones(d, np.float32), 'ln_f.b': np.zeros(d, np.float32)}
    for name in ORDER(L):
        shape = shapes[name.split('.', 1)[-1]]
        n = shape[0] * shape[1]
        scale = np.frombuffer(blob[at:at + 4], np.float32)[0]
        packed = np.frombuffer(blob[at + 4:at + 4 + -(-n // 5)], np.uint8).astype(np.int64)
        trits = (packed[:, None] // np.array([1, 3, 9, 27, 81])) % 3
        w[name] = ((trits.reshape(-1)[:n] - 1) * scale).reshape(shape).astype(np.float32)
        at += 4 + -(-n // 5)
    for i in range(L):
        for ln in ('ln1', 'ln2'):
            w[f'h{i}.{ln}.g'], w[f'h{i}.{ln}.b'] = np.ones(d, np.float32), np.zeros(d, np.float32)
    assert at == len(blob)
    return w, {'block': block, 'layers': L, 'heads': H, 'd': d, 'mlp': mlp}, chars


# ---- training: one conversation per row, each starting at position 0 ----------------
def rows(n, seed):
    rng = np.random.default_rng(seed)
    texts = ['\n' + big.conversation(rng) for _ in range(n)]
    assert max(len(t) for t in texts) <= WIDTH
    return np.array([[big.STOI[c] for c in t.ljust(WIDTH, '\n')] for t in texts])

def train(cfg, steps, batch=32, lr=4e-3, seed=0, quiet=False):
    data = rows(20000, seed)
    model = TinyGPT(len(big.CHARS), WIDTH, cfg['layers'], cfg['heads'], cfg['d'], cfg['mlp'], cfg['three'], seed=seed)
    params = model.parameters()
    m = [np.zeros_like(p.data) for p in params]
    v = [np.zeros_like(p.data) for p in params]
    rng = np.random.default_rng(seed + 7)
    t0 = time.time()
    for step in range(steps):
        pick = data[rng.integers(0, len(data), batch)]
        _, loss = model.forward(pick[:, :-1], pick[:, 1:])
        for p in params:
            p.grad = None
        loss.backward()
        rate = lr * min(1, (step + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        norm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in params))
        for i, p in enumerate(params):
            g = p.grad * min(1.0, 1.0 / (norm + 1e-6))
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.99 * v[i] + 0.01 * g * g
            p.data -= rate * (m[i] / (1 - 0.9 ** (step + 1))) / (np.sqrt(v[i] / (1 - 0.99 ** (step + 1))) + 1e-8)
        if not quiet and (step % 500 == 0 or step == steps - 1):
            print(f'  step {step:5d}  loss {float(loss.data):.4f}  {time.time() - t0:.0f}s', flush=True)
    return model, time.time() - t0


# ---- the test: every held-out question there is --------------------------------------
# [snippet:grade]
def grade(run, limit=None, pairs=None, ask=None):
    """Ask every question whose number pair was kept out of training: 1,000 pairs x 3 operations."""
    pairs = pairs or [(a, b) for a in range(100) for b in range(100) if big.held_out(a, b)]
    ask = ask or (lambda q: big.answer(run, q, temperature=0))
    asked = right = calls = 0
    wrong = []
    for a, b in pairs:
        for word, tool in big.WORDS.items():
            if limit and asked >= limit:
                break
            result = big.TOOLS[tool](a, b)
            got, spans = ask(f'what is {a} {word} {b}?')
            call = [s['output'].strip() for s in spans if s['name'] == 'write call']
            call_ok = call == [f'{tool} {a} {b}']
            ok = call_ok and got == f'{a} {word} {b} is {result}.'
            asked, calls, right = asked + 1, calls + call_ok, right + ok
            if not ok and len(wrong) < 5:
                wrong.append((f'{a} {word} {b}', call, got))
    chat = sum(ask(q)[0] == a for q, a in big.CHAT)
    return {'asked': asked, 'calls': calls, 'right': right, 'chat': chat, 'wrong': wrong}
# [/snippet]

def runner(weights, cfg):
    path = os.path.join(RUNS, f'_tmp_{os.getpid()}.npz')
    save_npz(path, weights, cfg)
    run = Runner(path)
    os.remove(path)
    return run

def speed(run, n=3):
    """Tokens per second for one conversation, with the KV cache."""
    text = '\nU what is 23 times 45?\nC mul 23 45\nR 1035\nA 23 times 45 is 1035.\n'
    best = 1e9
    for _ in range(n):
        cache, t0 = run.new_cache(), time.perf_counter()
        for c in text:
            run.step(run.stoi[c], cache)
        best = min(best, time.perf_counter() - t0)
    return len(text) / best


CANDIDATES = {   # name: width, layers, heads, MLP factor, three-valued?
    'd32 float':   dict(d=32, layers=2, heads=4, mlp=4, three=False),
    'd32':         dict(d=32, layers=2, heads=4, mlp=4, three=True),
    'd32 mlp2':    dict(d=32, layers=2, heads=4, mlp=2, three=True),
    'd24':         dict(d=24, layers=2, heads=4, mlp=4, three=True),
    'd24 mlp2':    dict(d=24, layers=2, heads=4, mlp=2, three=True),
    'd16 float':   dict(d=16, layers=2, heads=2, mlp=4, three=False),
    'd16':         dict(d=16, layers=2, heads=2, mlp=4, three=True),
    'd16 mlp2':    dict(d=16, layers=2, heads=2, mlp=2, three=True),
    'd32 1 layer': dict(d=32, layers=1, heads=4, mlp=4, three=True),
    'd48':         dict(d=48, layers=2, heads=4, mlp=4, three=True),
    'd48 mlp2':    dict(d=48, layers=2, heads=4, mlp=2, three=True),
    'd40':         dict(d=40, layers=2, heads=4, mlp=4, three=True),
    'd64':         dict(d=64, layers=2, heads=4, mlp=4, three=True),
}
CHOSEN = 'd64'


def report(name, cfg, weights, seconds, limit=None):
    cfg = dict(cfg, block=WIDTH)
    n = sum(weights[k].size for k in ORDER(cfg['layers']))
    size = len(pack(weights, cfg)) if cfg['three'] else 4 * n
    g = grade(runner(weights, cfg), limit)
    row = {'name': name, 'weights': n, 'bytes': size, 'three': cfg['three'], 'train_s': round(seconds), **g}
    print(f"{name:14}{n:>8,} weights{size:>9,} bytes   calls {g['calls']}/{g['asked']}   right {g['right']}/{g['asked']}   chat {g['chat']}/3", flush=True)
    return row


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--only', default='')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--out', default=SWEEP)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()

    if args.sweep:
        names = [n.strip() for n in args.only.split(',')] if args.only else ['chapter 19'] + list(CANDIDATES)
        results = []
        if 'chapter 19' in names:                         # the model we start from, as it is and rounded after training
            z = np.load(big.PATH)
            base = {k: z[k] for k in z.files if not k.startswith('__')}
            matrices = [k for k in base if base[k].ndim == 2]
            n = sum(a.size for a in base.values())
            def old(label, w, size):
                path = os.path.join(RUNS, f'_tmp_{os.getpid()}.npz')
                np.savez(path, __chars__=z['__chars__'], __config__=z['__config__'], **w)
                g = grade(Runner(path)); os.remove(path)
                print(f"{label:28}{n:>8,} weights{size:>9,} bytes   right {g['right']}/{g['asked']}   chat {g['chat']}/3", flush=True)
                results.append({'name': label, 'weights': n, 'bytes': size, 'three': False, 'train_s': 0, **g})
            old('chapter 19', base, 4 * n)
            for bits in (8, 4):
                old(f'chapter 19, {bits} bits after', {k: quantize(a, bits) if k in matrices else a for k, a in base.items()}, n * bits // 8 + 2 * (n // 32))
            old('chapter 19, ternary after', {k: (lambda qs: (qs[0] * qs[1]).astype(np.float32))(ternary(a)) if k in matrices else a for k, a in base.items()}, n // 5)
            names = [x for x in names if x != 'chapter 19']
            with open(args.out, 'w') as f:
                json.dump(results, f)
        for name in names:
            cfg = CANDIDATES[name]
            model, seconds = train(cfg, args.steps, quiet=True)
            results.append(report(name, cfg, model.weights(), seconds))
            if not cfg['three']:                          # the same float model, rounded AFTER training
                w = model.weights()
                for bits in (8, 4):
                    r = report(f'{name}, {bits} bits after', cfg, {k: quantize(a, bits) if k in model.stored else a for k, a in w.items()}, 0)
                    r['bytes'] = r['weights'] * bits // 8 + 2 * (r['weights'] // 32)
                    results.append(r)
                w3 = {k: (lambda q_s: q_s[0] * q_s[1])(ternary(a)).astype(np.float32) if k in model.stored else a for k, a in w.items()}
                r = report(f'{name}, ternary after', dict(cfg, three=True), w3, 0)
                results.append(r)
            with open(args.out, 'w') as f:
                json.dump(results, f)
        sys.exit()

    if args.train:
        cfg = dict(CANDIDATES[CHOSEN], block=WIDTH)
        print(f'training {CHOSEN}: {cfg}')
        model, seconds = train(cfg, args.steps)
        w = model.weights()
        with open(BIN, 'wb') as f:
            f.write(pack(w, cfg))
        with open(FACTS, 'w') as f:
            json.dump({'name': CHOSEN, 'steps': args.steps, 'train_s': round(seconds)}, f)
        print(f'trained in {seconds:.0f}s, wrote {os.path.normpath(BIN)} ({os.path.getsize(BIN):,} bytes)')

    if not os.path.exists(BIN):
        sys.exit('No packed model yet. Run:  python code/ch23_tiny.py --train')
    blob = open(BIN, 'rb').read()
    w, cfg, chars = unpack(blob)                       # everything below uses only what is in the file
    run, old = runner(w, cfg), Runner(big.PATH)
    n = sum(w[k].size for k in ORDER(cfg['layers']))
    print(f'{len(blob):,} bytes on disk, {n:,} weights, {8 * len(blob) / n:.2f} bits per weight')

    held = grade(run)
    print(f"held-out pairs:  {held['asked']:,} asked, {held['calls']:,} correct calls, {held['right']:,} call and answer both right; small talk {held['chat']} of 3")
    for q, call, got in held['wrong']:
        print('  failed:', q, call, repr(got))
    seen_pairs = [(a, b) for a in range(100) for b in range(100) if not big.held_out(a, b)][::9]   # every ninth: 1,000 pairs
    seen, seen_old = grade(run, pairs=seen_pairs), grade(old, pairs=seen_pairs)
    print(f"pairs seen in training: {seen['right']:,} of {seen['asked']:,} right; the chapter 19 model: {seen_old['right']:,}")
    fast, fast_old = speed(run), speed(old)
    print(f'tokens per second with numpy: this model {fast:,.0f}, chapter 19 model {fast_old:,.0f}')

    import tiny_agent as plain                         # the same file, run without numpy
    m = plain.load(BIN)
    text = '\nU what is 23 times 45?\nC mul 23 45\nR 1035\nA 23 times 45 is 1035.\n'
    cache_np, cache_py, worst, t_plain = run.new_cache(), [], 0.0, 0.0
    for c in text:
        a = run.step(run.stoi[c], cache_np)
        t0 = time.perf_counter()
        b = plain.step(m, m['stoi'][c], cache_py)
        t_plain += time.perf_counter() - t0
        worst = max(worst, float(np.abs(a - np.array(b)).max()))
    print(f'plain Python, no numpy: {len(text) / t_plain:,.0f} tokens per second; largest difference from numpy in any score: {worst:.1e}')
    facts = json.load(open(FACTS)) if os.path.exists(FACTS) else {}
    facts.update({'bytes': len(blob), 'weights': n, 'held': held, 'seen': seen, 'seen_old': seen_old, 'tps': round(fast), 'tps_old': round(fast_old),
                  'tps_plain': round(len(text) / t_plain), 'worst': worst})
    if args.check:                                     # every held-out question again, through the plain-Python engine
        def ask(q):
            answer, call, _ = plain.answer(m, q)
            return answer, ([{'name': 'write call', 'output': call}] if call else [])
        again = grade(None, ask=ask)
        print(f"plain Python on every held-out question: {again['right']:,} of {again['asked']:,} right (numpy: {held['right']:,})")
        facts['plain_right'] = again['right']
    with open(FACTS, 'w') as f:
        json.dump(facts, f)
