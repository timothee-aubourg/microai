"""
ch23_signs.py — the language model of this course with its weights replaced by signs.  (Chapter 23)

    python code/ch23_signs.py --train     # two trainings, about 40 minutes on two cores
    python code/ch23_signs.py             # the table, and a sample of what each model writes

Chapter 23 shrank a model that does one narrow job. This file asks the same
question of the general model, the one that learned to write from Shakespeare
in chapter 7: what if every weight keeps only its sign, -1, 0 or +1?

    after     take the trained model of chapter 7 and replace each matrix by its signs
    trained   train the same network, on the same batches, with signs in every forward pass
    wider     the same again with vectors of 144 numbers in place of 96

The yardstick is the loss on text the model never saw (chapter 1): lower is better.
"""
import argparse, json, math, os, sys, time
import numpy as np
from engine import Tensor
from gpt import GPT, load_text
from ch09_quantize import quantize
from ch23_tiny import ternary

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
OUT = os.path.join(RUNS, 'signs.json')


# [snippet:signgpt]
class SignGPT(GPT):
    """The GPT of chapter 6. Every matrix is used as -1, 0 or +1 times one trained number."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matrices = [k for k, t in self.p.items() if t.data.ndim == 2]
        self.size = {k: Tensor(np.array([ternary(self.p[k].data)[1]])) for k in self.matrices}

    def parameters(self):
        return list(self.p.values()) + list(self.size.values())

    def forward(self, idx, targets=None, trace=None):
        kept = self.p
        self.p = dict(kept)
        for k in self.matrices:
            signs, _ = ternary(kept[k].data)
            self.p[k] = (kept[k] + (signs - kept[k].data)) * self.size[k]   # compute with signs, learn on the kept weights
        try:
            return super().forward(idx, targets, trace)
        finally:
            self.p = kept

    def signed(self):
        """The weights the finished model computes with."""
        out = {k: t.data.astype(np.float32) for k, t in self.p.items()}
        for k in self.matrices:
            out[k] = (ternary(self.p[k].data)[0] * float(self.size[k].data[0])).astype(np.float32)
        return out
# [/snippet]


def corpus():
    text = load_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    split = int(0.9 * len(data))
    return data[:split], data[split:], chars


def train(d, steps=5000, batch=32, lr=2e-3):
    """The loop of code/train.py, step for step: same seed, same batches, same schedule."""
    rng = np.random.default_rng(0)
    data, val, chars = corpus()
    model = SignGPT(vocab_size=len(chars), d_model=d)
    T, params = model.block_size, model.parameters()
    def get_batch(source, n):
        starts = rng.integers(0, len(source) - T - 1, n)
        return np.stack([source[s:s + T] for s in starts]), np.stack([source[s + 1:s + T + 1] for s in starts])
    [get_batch(val, 64) for _ in range(4)]                  # train.py draws its fixed test batches first
    m = [np.zeros_like(p.data) for p in params]
    v = [np.zeros_like(p.data) for p in params]
    t0 = time.time()
    for step in range(steps):
        x, y = get_batch(data, batch)
        _, loss = model.forward(x, y)
        for p in params:
            p.grad = None
        loss.backward()
        clip = min(1.0, 1.0 / (math.sqrt(sum(float((p.grad ** 2).sum()) for p in params)) + 1e-6))
        rate = lr * (step + 1) / 100 if step < 100 else lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 100) / max(1, steps - 100))))
        for i, p in enumerate(params):
            g = p.grad * clip
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.99 * v[i] + 0.01 * g * g
            p.data -= rate * (m[i] / (1 - 0.9 ** (step + 1))) / (np.sqrt(v[i] / (1 - 0.99 ** (step + 1))) + 1e-8)
        if step % 250 == 0 or step == steps - 1:
            print(f'  width {d}  step {step:5d}  loss {float(loss.data):.3f}  {time.time() - t0:.0f}s', flush=True)
    w = model.signed()
    np.savez(os.path.join(RUNS, f'signs_{d}.npz'), __chars__=np.array(list(chars)), __seconds__=np.array([time.time() - t0]),
             __config__=np.array([model.vocab_size, model.block_size, model.n_layer, model.n_head, model.d_model]), **w)


# [snippet:measure]
def held_out_loss(model, val, windows=200):
    """Average surprise, in nats per character, on `windows` stretches of text the model never saw."""
    T = model.block_size
    starts = np.linspace(0, len(val) - T - 1, windows).astype(int)
    losses = []
    for i in range(0, windows, 50):
        x = np.stack([val[s:s + T] for s in starts[i:i + 50]])
        y = np.stack([val[s + 1:s + T + 1] for s in starts[i:i + 50]])
        z = model.forward(x)[0].data
        z = z - z.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        losses.append(-np.take_along_axis(logp, y[..., None], -1)[..., 0].ravel())
    losses = np.concatenate(losses)
    return float(losses.mean()), float(losses.std() / math.sqrt(len(losses)))
# [/snippet]

def write(model, chars, n=120, seed=0, temperature=0.8):
    rng, idx = np.random.default_rng(seed), [chars.index('\n')]
    for _ in range(n):
        z = model.forward(np.array([idx[-model.block_size:]]))[0].data[0, -1] / temperature
        p = np.exp(z - z.max()); p /= p.sum()
        idx.append(int(rng.choice(len(p), p=p)))
    return ''.join(chars[i] for i in idx[1:])

def packed_bytes(w):
    """Size on disk with five signs to a byte, one number per matrix, and 2 bytes for every other number."""
    return sum(-(-a.size // 5) + 4 if a.ndim == 2 else 2 * a.size for a in w.values())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, nargs='*')
    args = ap.parse_args()
    if args.train is not None:
        for d in (args.train or [96, 144]):
            train(d)
        sys.exit()

    _, val, chars = corpus()
    base, _ = GPT.load(os.path.join(RUNS, 'micro.npz'))
    kept = {k: t.data.copy() for k, t in base.p.items()}
    mats = [k for k, a in kept.items() if a.ndim == 2]
    n = sum(a.size for a in kept.values())
    rows = []
    def row(name, model, weights, size, seconds=None):
        loss, se = held_out_loss(model, val)
        text = write(model, chars)
        rows.append({'name': name, 'weights': weights, 'bytes': int(size), 'loss': round(loss, 4), 'se': round(se, 4), 'sample': text, 'seconds': seconds})
        print(f'{name:44}{weights:>9,} weights{size / 1024:>8.0f} KB   loss {loss:.3f} ± {se:.3f}')

    print(f'a model that knows nothing scores {math.log(len(chars)):.3f} (65 equally likely characters)\n')
    row('trained normally, 32 bits per weight', base, n, 4 * n)
    for bits in (8, 4):
        for k in mats:
            base.p[k] = Tensor(quantize(kept[k], bits))
        row(f'the same, rounded to {bits} bits afterwards', base, n, sum((a.size * bits // 8 + 2 * (a.size // 32)) if a.ndim == 2 else 2 * a.size for a in kept.values()))
    for k in mats:
        q, s = ternary(kept[k])
        base.p[k] = Tensor((q * s).astype(np.float32))
    row('the same, signs taken afterwards (-1, 0, +1)', base, n, packed_bytes(kept))
    for k in mats:
        base.p[k] = Tensor((np.sign(kept[k]) * np.abs(kept[k]).mean()).astype(np.float32))
    row('the same, signs taken afterwards (-1, +1)', base, n, sum(-(-a.size // 8) + 4 if a.ndim == 2 else 2 * a.size for a in kept.values()))
    for d in (96, 144):
        path = os.path.join(RUNS, f'signs_{d}.npz')
        if os.path.exists(path):
            model, _ = GPT.load(path)
            w = {k: t.data for k, t in model.p.items()}
            row(f'trained as signs, width {d}', model, sum(a.size for a in w.values()), packed_bytes(w), round(float(np.load(path)['__seconds__'][0])))
    print()
    for r in rows:
        print(f"--- {r['name']}\n{r['sample']}\n")
    with open(OUT, 'w') as f:
        json.dump(rows, f)
