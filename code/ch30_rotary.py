"""
ch30_rotary.py — positions that turn: the model of chapter 7 without a position table.   (Chapter 30)

    python code/ch30_rotary.py --train     # two trainings, about 20 minutes each
    python code/ch30_rotary.py             # the table

Chapter 4 gave every position its own learned vector, added to the token. Models
made since about 2022 do something else: they add nothing to the token, and
instead TURN each query and key by an angle that grows with its position, just
before attention compares them. Three models, same data, same batches, same steps:

    table     the model of chapter 7: one learned vector per position (runs/micro.npz)
    rotary    no table: queries and keys are turned by their position
    none      no position information at all, as a control
"""
import argparse, json, math, os, sys, time
import numpy as np
from engine import Tensor, cross_entropy
from gpt import GPT, load_text

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
OUT = os.path.join(RUNS, 'rotary.json')


# [snippet:turn]
def angles(T, hd, base=10000.0):
    """One angle per position and per pair of numbers: slow pairs and fast pairs, as on a clock."""
    speed = 1.0 / base ** (np.arange(0, hd, 2) / hd)              # hd/2 speeds
    a = np.arange(T)[:, None] * speed[None, :]                    # (T, hd/2)
    return np.repeat(np.cos(a), 2, axis=1), np.repeat(np.sin(a), 2, axis=1)   # each pair shares its angle

def quarter_turn(hd):
    """The matrix that sends each pair (a, b) to (-b, a)."""
    P = np.zeros((hd, hd))
    for j in range(0, hd, 2):
        P[j + 1, j], P[j, j + 1] = -1.0, 1.0
    return P

def turn(t, cos, sin, P):
    """Turn every pair of numbers in t by the angle of its position: (a, b) cos + (-b, a) sin."""
    return t * cos + (t @ P) * sin
# [/snippet]


class RotaryGPT(GPT):
    """The GPT of chapter 6 with no position table. mode 'rotary' turns queries and keys; 'none' does nothing."""
    def __init__(self, vocab_size, mode='rotary', **kw):
        super().__init__(vocab_size, **kw)
        del self.p['wpe']
        self.mode = mode
        self.P = Tensor(quarter_turn(self.d_model // self.n_head))     # a constant: it is never trained

    # [snippet:attention]
    def attention(self, x, i, trace=None):
        B, T, d = x.shape
        h, hd = self.n_head, d // self.n_head
        split = lambda t: t.reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        q, k, v = (split(x @ self.p[f'h{i}.attn.{w}']) for w in ('wq', 'wk', 'wv'))
        if self.mode == 'rotary':
            cos, sin = angles(T, hd)
            q, k = turn(q, cos, sin, self.P), turn(k, cos, sin, self.P)    # the only change from chapter 5
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(hd))
        weights = scores.masked_fill(np.triu(np.ones((T, T), dtype=bool), k=1), -1e9).softmax(axis=-1)
        mixed = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        return mixed @ self.p[f'h{i}.attn.wo']
    # [/snippet]

    def forward(self, idx, targets=None, trace=None):
        B, T = idx.shape
        p = self.p
        x = p['wte'][idx]                                         # what the token is. Nothing about where.
        for i in range(self.n_layer):
            x = x + self.attention(self.layernorm(x, p[f'h{i}.ln1.g'], p[f'h{i}.ln1.b']), i)
            x = x + (self.layernorm(x, p[f'h{i}.ln2.g'], p[f'h{i}.ln2.b']) @ p[f'h{i}.mlp.w1']).gelu() @ p[f'h{i}.mlp.w2']
        logits = self.layernorm(x, p['ln_f.g'], p['ln_f.b']) @ p['wte'].transpose(1, 0)
        if targets is None:
            return logits, None
        return logits, cross_entropy(logits.reshape(B * T, self.vocab_size), targets.reshape(B * T))


def corpus():
    text = load_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    split = int(0.9 * len(data))
    return data[:split], data[split:], chars

def train(mode, steps=5000, batch=32, lr=2e-3):
    """The loop of code/train.py, step for step: same seed, same batches, same schedule."""
    rng = np.random.default_rng(0)
    data, val, chars = corpus()
    model = RotaryGPT(len(chars), mode)
    T, params = model.block_size, model.parameters()
    def get_batch(source, n):
        starts = rng.integers(0, len(source) - T - 1, n)
        return np.stack([source[s:s + T] for s in starts]), np.stack([source[s + 1:s + T + 1] for s in starts])
    [get_batch(val, 64) for _ in range(4)]                        # train.py draws its fixed test batches first
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
        if step % 500 == 0 or step == steps - 1:
            print(f'  {mode}  step {step:5d}  loss {float(loss.data):.3f}  {time.time() - t0:.0f}s', flush=True)
    np.savez(os.path.join(RUNS, f'pos_{mode}.npz'), __seconds__=np.array([time.time() - t0]), **{k: t.data for k, t in model.p.items()})

def load(mode, vocab):
    model = RotaryGPT(vocab, mode)
    z = np.load(os.path.join(RUNS, f'pos_{mode}.npz'))
    for k in model.p:
        model.p[k] = Tensor(z[k])
    return model

def held_out_loss(model, val, window, count=100):
    """Loss on the LAST 32 characters of each window, so that every window length is judged on the same thing."""
    starts = np.linspace(0, len(val) - 300, count).astype(int) + 256 - window
    out = []
    for i in range(0, count, 25):
        x = np.stack([val[s:s + window] for s in starts[i:i + 25]])
        y = np.stack([val[s + 1:s + window + 1] for s in starts[i:i + 25]])
        z = model.forward(x)[0].data[:, -32:]
        z = z - z.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        out.append(-np.take_along_axis(logp, y[:, -32:, None], -1).ravel())
    out = np.concatenate(out)
    return float(out.mean()), float(out.std() / math.sqrt(len(out)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', nargs='*')
    args = ap.parse_args()
    if args.train is not None:
        for mode in (args.train or ['rotary', 'none']):
            train(mode)
        sys.exit()
    _, val, chars = corpus()
    table, _ = GPT.load(os.path.join(RUNS, 'micro.npz'))
    models = {'table': table, 'rotary': load('rotary', len(chars)), 'none': load('none', len(chars))}
    rows = []
    print(f'{"positions":10}{"weights":>10}' + ''.join(f'{"window " + str(w):>18}' for w in (32, 64, 128, 256)))
    for name, model in models.items():
        row = {'name': name, 'weights': sum(t.data.size for t in model.p.values()), 'loss': {}}
        line = f"{name:10}{row['weights']:>10,}"
        for w in (32, 64, 128, 256):
            if name == 'table' and w > 64:
                line += f'{"cannot run":>18}'
                continue
            loss, se = held_out_loss(model, val, w)
            row['loss'][str(w)] = [round(loss, 3), round(se, 3)]
            line += f'{loss:>12.3f} ± {se:.3f}'
        rows.append(row)
        print(line)
    with open(OUT, 'w') as f:
        json.dump(rows, f)
