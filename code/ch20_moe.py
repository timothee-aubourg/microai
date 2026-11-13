"""
ch20_moe.py — many experts, few at work.                               (Chapter 20)

    python code/ch20_moe.py --train     # trains three small models: about 8 minutes on a laptop CPU
    python code/ch20_moe.py             # compare them, and show which expert handles what

In the GPT of chapter 6 every token passes through the whole MLP of every
block. A mixture of experts replaces that one MLP with several smaller ones
(the experts) and a router that sends each token to a few of them. The model
owns more weights than it uses for any one token.

This script trains the same small GPT three ways on a text made of two very
different kinds of writing (Shakespeare, and the calculator conversations of
chapter 19) and measures two things: what the extra, mostly idle weights buy,
and whether the experts end up with different jobs.

    small    one MLP, 128 hidden units           uses 128 per token
    experts  four experts of 64, two per token   uses 128 per token, owns 256
    big      one MLP, 256 hidden units           uses 256 per token
"""
import argparse, json, math, os, time
import numpy as np
from engine import Tensor
from gpt import GPT, load_text
from ch19_system import conversation

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
CONFIGS = {'small': dict(experts=1, active=1, hidden=128),
           'experts': dict(experts=4, active=2, hidden=64),
           'big': dict(experts=1, active=1, hidden=256)}


def mixed_text(seed=0, blocks=1500):
    """Two kinds of writing in one stream: a stretch of Shakespeare, then a few conversations, and so on."""
    rng, play, out, at = np.random.default_rng(seed), load_text(), [], 0
    for _ in range(blocks):
        out.append(play[at:at + 320]); at = (at + 320) % (len(play) - 320)
        out.append('\n\n' + ''.join(conversation(rng) for _ in range(5)))
    return ''.join(out)


class MoEGPT(GPT):
    """The GPT of gpt.py with one change: each block's MLP becomes `experts` small MLPs and a router.
    With experts=1 it is an ordinary GPT whose MLP has `hidden` units."""

    def __init__(self, vocab_size, experts=4, active=2, hidden=64, **kw):
        super().__init__(vocab_size, **kw)
        rng, d = np.random.default_rng(7), self.d_model
        self.experts, self.active, self.hidden = experts, active, hidden
        for i in range(self.n_layer):
            del self.p[f'h{i}.mlp.w1'], self.p[f'h{i}.mlp.w2']
            self.p[f'h{i}.router'] = Tensor(rng.normal(0, 0.02, (d, experts)))
            for e in range(experts):
                self.p[f'h{i}.exp{e}.w1'] = Tensor(rng.normal(0, 0.02, (d, hidden)))
                self.p[f'h{i}.exp{e}.w2'] = Tensor(rng.normal(0, 0.02 / math.sqrt(2 * self.n_layer), (hidden, d)))

    # [snippet:route]
    def route(self, x, i):
        """For every token: a score per expert, the `active` best ones, and their weights (which add up to 1)."""
        scores = x @ self.p[f'h{i}.router']                          # (B, T, experts)
        best = np.argsort(-scores.data, axis=-1)[..., :self.active]  # which experts this token is sent to
        chosen = np.zeros(scores.shape, dtype=bool)
        np.put_along_axis(chosen, best, True, axis=-1)
        gates = scores.masked_fill(~chosen, -1e9).softmax(-1)        # softmax over the chosen ones; 0 for the rest
        return gates, chosen
    # [/snippet]

    # [snippet:experts]
    def experts_layer(self, x, i):
        gates, chosen = self.route(x, i)
        out = None
        for e in range(self.experts):
            y = (x @ self.p[f'h{i}.exp{e}.w1']).gelu() @ self.p[f'h{i}.exp{e}.w2']   # one small MLP
            weight = (gates * np.eye(self.experts)[e]).sum(-1, keepdims=True)         # this expert's share, per token
            out = y * weight if out is None else out + y * weight
        return out, gates                  # in training every expert runs; a runtime skips the ones with weight 0
    # [/snippet]

    def forward(self, idx, targets=None, trace=None):
        B, T = idx.shape
        p = self.p
        x = p['wte'][idx] + p['wpe'][np.arange(T)]
        balance = 0.0
        for i in range(self.n_layer):
            x = x + self.attention(self.layernorm(x, p[f'h{i}.ln1.g'], p[f'h{i}.ln1.b']), i)
            y, gates = self.experts_layer(self.layernorm(x, p[f'h{i}.ln2.g'], p[f'h{i}.ln2.b']), i)
            x = x + y
            if trace is not None:
                trace.append(gates.data)
            if self.experts > 1:           # keep every expert in work: penalise uneven average use
                use = gates.mean(axis=0).mean(axis=0)
                balance = ((use - 1.0 / self.experts) ** 2).sum() * self.experts + balance
        x = self.layernorm(x, p['ln_f.g'], p['ln_f.b'])
        logits = x @ p['wte'].transpose(1, 0)
        if targets is None:
            return logits, None
        from engine import cross_entropy
        loss = cross_entropy(logits.reshape(B * T, self.vocab_size), targets.reshape(B * T))
        return logits, (loss + balance * 0.1 if self.experts > 1 else loss)

    def counts(self):
        """(weights owned, weights used for one token)."""
        total = self.num_params()
        per_expert = 2 * self.d_model * self.hidden
        idle = self.n_layer * (self.experts - self.active) * per_expert
        return total, total - idle


# [snippet:sparse]
def one_token_sparse(model, x, i):
    """What a runtime does: for ONE token, run only the experts it was sent to."""
    p = model.p
    scores = x @ p[f'h{i}.router'].data
    best = np.argsort(-scores)[:model.active]
    w = np.exp(scores[best] - scores[best].max()); w /= w.sum()
    out = np.zeros_like(x)
    for e, weight in zip(best, w):                     # two small MLPs instead of four
        h = x @ p[f'h{i}.exp{e}.w1'].data
        h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3)))
        out += weight * (h @ p[f'h{i}.exp{e}.w2'].data)
    return out
# [/snippet]


def batches(data, T, n, size, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        starts = rng.integers(0, len(data) - T - 1, size)
        out.append((np.stack([data[s:s + T] for s in starts]), np.stack([data[s + 1:s + T + 1] for s in starts])))
    return out


def train(name, data, vocab, steps, lr=3e-3, batch=32):
    model = MoEGPT(vocab, block_size=64, n_layer=2, n_head=4, d_model=64, **CONFIGS[name])
    T, params = model.block_size, model.parameters()
    m = [np.zeros_like(q.data) for q in params]
    v = [np.zeros_like(q.data) for q in params]
    rng, t0 = np.random.default_rng(0), time.time()           # the same batches for every model
    for step in range(steps):
        starts = rng.integers(0, len(data) - T - 1, batch)
        x = np.stack([data[s:s + T] for s in starts]); y = np.stack([data[s + 1:s + T + 1] for s in starts])
        _, loss = model.forward(x, y)
        for q in params:
            q.grad = None
        loss.backward()
        rate = lr * min(1, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        norm = math.sqrt(sum(float((q.grad ** 2).sum()) for q in params if q.grad is not None))
        for k, q in enumerate(params):
            if q.grad is None:
                continue
            g = q.grad * min(1.0, 1.0 / (norm + 1e-6))
            m[k] = 0.9 * m[k] + 0.1 * g
            v[k] = 0.99 * v[k] + 0.01 * g * g
            q.data -= rate * (m[k] / (1 - 0.9 ** (step + 1))) / (np.sqrt(v[k] / (1 - 0.99 ** (step + 1))) + 1e-8)
        if step % 250 == 0 or step == steps - 1:
            print(f'  {name:8} step {step:5d}  loss {float(loss.data):.3f}  {time.time() - t0:.0f}s', flush=True)
    return model, time.time() - t0


def plain_loss(model, held):
    """Held-out loss without the balance term, so the three models are compared on the same thing."""
    total = 0.0
    for x, y in held:
        logits, _ = model.forward(x)
        z = logits.data.reshape(-1, model.vocab_size); z = z - z.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        total += -logp[np.arange(len(z)), y.reshape(-1)].mean()
    return total / len(held)


KINDS = [('lowercase letters', 'abcdefghijklmnopqrstuvwxyz'), ('capital letters', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
         ('digits', '0123456789'), ('spaces and line breaks', ' \n'), ('punctuation', ".,;:!?'-&$")]

def routing(model, chars, text):
    """For each character of `text`, in each block: the expert with the largest weight, and all the weights."""
    stoi = {c: k for k, c in enumerate(chars)}
    ids = np.array([[stoi[c] for c in text[:model.block_size]]])
    trace = []
    model.forward(ids, trace=trace)
    return [g[0] for g in trace]                        # per block: (T, experts)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--steps', type=int, default=2500)
    args = ap.parse_args()
    text = mixed_text()
    chars = sorted(set(text))
    stoi = {c: k for k, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text])
    split = int(0.9 * len(data))
    held = batches(data[split:], 64, 6, 64, seed=1)
    path = os.path.join(RUNS, 'moe.json')

    if args.train:
        print(f'{len(text):,} characters (Shakespeare and conversations, alternating), vocabulary {len(chars)}')
        results = {'steps': args.steps, 'models': {}}
        for name in CONFIGS:
            model, seconds = train(name, data[:split], len(chars), args.steps)
            total, used = model.counts()
            results['models'][name] = {'loss': float(plain_loss(model, held)), 'weights': total, 'used_per_token': used,
                                       'seconds': seconds, **CONFIGS[name]}
            if name == 'experts':
                # the sparse path gives the same numbers as running every expert
                x = np.random.default_rng(3).normal(0, 1, 64)
                dense, _ = model.experts_layer(Tensor(x[None, None, :]), 0)
                assert np.abs(dense.data[0, 0] - one_token_sparse(model, x, 0)).max() < 1e-6
                # who handles what, on held-out text
                counts = {kind: np.zeros((model.n_layer, model.experts)) for kind, _ in KINDS}
                held_text = text[split:]
                for at in range(0, 64 * 200, 64):
                    piece = held_text[at:at + 64]
                    for layer, gates in enumerate(routing(model, chars, piece)):
                        top = gates.argmax(-1)
                        for c, e in zip(piece, top):
                            for kind, members in KINDS:
                                if c in members:
                                    counts[kind][layer, e] += 1
                results['who'] = {kind: (c / c.sum(-1, keepdims=True)).round(3).tolist() for kind, c in counts.items()}
                samples = ['First Citizen:\nBefore we proceed any further, hear me speak.\n',
                           'U what is 23 times 45?\nC mul 23 45\nR 1035\nA 23 times 45 is 1035.\n',
                           'ROMEO:\nBut soft, what light\n\nU what is 7 minus 60?\nC sub 7 60\nR -53\n']
                results['samples'] = [{'text': s[:64], 'gates': [g.round(3).tolist() for g in routing(model, chars, s)]} for s in samples]
        with open(path, 'w') as f:
            json.dump(results, f)
        print('wrote', os.path.normpath(path))

    if not os.path.exists(path):
        raise SystemExit('No results yet. Run:  python code/ch20_moe.py --train')
    with open(path) as f:
        results = json.load(f)
    print(f"\nThree models, {results['steps']} steps each, the same batches. Held-out loss, lower is better:\n")
    print(f'{"model":9}{"weights owned":>15}{"used per token":>16}{"held-out loss":>15}{"training time":>15}')
    for name, r in results['models'].items():
        print(f"{name:9}{r['weights']:>15,}{r['used_per_token']:>16,}{r['loss']:>15.3f}{r['seconds']:>13.0f} s")
    print('\nWhich expert gets most of each kind of character (share of characters, per block):\n')
    for kind, layers in results['who'].items():
        print(f'  {kind:24}' + '   '.join('block %d: ' % k + ' '.join(f'{100 * s:3.0f}%' for s in row) for k, row in enumerate(layers)))
    s = results['samples'][1]
    print('\nOne conversation, first block: the expert with the largest weight for each character\n')
    print('  ' + s['text'].replace('\n', '⏎'))
    print('  ' + ''.join(str(int(np.argmax(g))) for g in s['gates'][0]))
