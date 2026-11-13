"""
ch26_draft.py — a small model drafts, the large model checks.   (Chapter 26)

    python code/ch26_draft.py --train     # train the two draft models, about one minute
    python code/ch26_draft.py             # the measurements of the chapter, about 3 minutes

The large model is the one of chapter 7 (345,504 weights). The draft model has
the same shape with one block and vectors of 32 numbers. Text is written in
rounds:

    1. the draft model writes k tokens, one after another          (k cheap passes)
    2. the large model reads all of them in ONE pass and gives its
       own choice at every position                                (1 costly pass)
    3. keep the drafted tokens up to the first disagreement, and
       there take the large model's token instead

The result is exactly what the large model would have written alone. Only the
number of costly passes changes.
"""
import argparse, json, math, os, sys, time
import numpy as np
from gpt import GPT, load_text

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
OUT = os.path.join(RUNS, 'draft.json')
PROMPT, LENGTH = 16, 48                      # 16 characters given, 48 written: never past the 64 of the window


def corpus():
    text = load_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    split = int(0.9 * len(data))
    return data[:split], data[split:], chars

def train(name, steps, d=32, layers=1, heads=2, batch=32, lr=3e-3):
    """The loop of code/train.py on a smaller network."""
    rng = np.random.default_rng(0)
    data, _, chars = corpus()
    model = GPT(len(chars), n_layer=layers, n_head=heads, d_model=d)
    T, params = model.block_size, model.parameters()
    m = [np.zeros_like(p.data) for p in params]
    v = [np.zeros_like(p.data) for p in params]
    t0 = time.time()
    for step in range(steps):
        starts = rng.integers(0, len(data) - T - 1, batch)
        x, y = np.stack([data[s:s + T] for s in starts]), np.stack([data[s + 1:s + T + 1] for s in starts])
        _, loss = model.forward(x, y)
        for p in params:
            p.grad = None
        loss.backward()
        rate = lr * min(1, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for i, p in enumerate(params):
            m[i] = 0.9 * m[i] + 0.1 * p.grad
            v[i] = 0.99 * v[i] + 0.01 * p.grad * p.grad
            p.data -= rate * (m[i] / (1 - 0.9 ** (step + 1))) / (np.sqrt(v[i] / (1 - 0.99 ** (step + 1))) + 1e-8)
        if step % 500 == 0 or step == steps - 1:
            print(f'  {name}  step {step:5d}  loss {float(loss.data):.3f}  {time.time() - t0:.0f}s', flush=True)
    model.save(os.path.join(RUNS, f'{name}.npz'), chars)


class Counted:
    """A model that counts how many times it is run, and for how long."""
    def __init__(self, path):
        self.model, self.chars = GPT.load(path)
        self.passes, self.seconds = 0, 0.0
        self.weights = sum(p.data.size for p in self.model.parameters())
    def scores(self, tokens):
        """One pass: the scores for the next token at EVERY position of `tokens`."""
        t0 = time.perf_counter()
        z = self.model.forward(np.array([tokens]))[0].data[0]
        self.passes, self.seconds = self.passes + 1, self.seconds + time.perf_counter() - t0
        return z
    def reset(self):
        self.passes, self.seconds = 0, 0.0


def alone(big, tokens, n):
    """The large model writes n tokens by itself: one pass per token. Greedy, so the result is repeatable."""
    tokens = list(tokens)
    for _ in range(n):
        tokens.append(int(big.scores(tokens)[-1].argmax()))
    return tokens

# [snippet:speculate]
def speculate(big, small, tokens, n, k):
    """Write n tokens. The small model drafts k at a time; the large one checks each draft in one pass."""
    tokens, target, kept, drafted = list(tokens), len(tokens) + n, 0, 0
    while len(tokens) < target:
        k_now = min(k, target - len(tokens))
        draft = list(tokens)
        for _ in range(k_now):                               # 1. draft: k cheap passes
            draft.append(int(small.scores(draft)[-1].argmax()))
        z = big.scores(draft)                                # 2. check: ONE costly pass over all of it
        choice = z.argmax(-1)                                #    the large model's own token at every position
        for i in range(len(tokens), len(draft)):             # 3. keep the draft while the two agree
            drafted += 1
            if draft[i] != choice[i - 1]:
                tokens.append(int(choice[i - 1]))            #    first disagreement: take the large model's token
                break
            tokens.append(draft[i]); kept += 1
        else:
            if len(tokens) < target:
                tokens.append(int(choice[-1]))               #    all kept: the same pass gives one more token for free
    return tokens[:target], kept, drafted
# [/snippet]

# [snippet:accept]
def accept_or_redraw(p, q, rng):
    """The rule for SAMPLED text. q: the draft's probabilities, p: the large model's.
    Returns one token drawn exactly as if the large model had drawn it."""
    x = rng.choice(len(q), p=q)                              # the draft proposes x
    if rng.random() < min(1.0, p[x] / q[x]):                 # keep it with probability p(x) / q(x), at most 1
        return x
    rest = np.maximum(p - q, 0)                              # otherwise draw from what the draft under-weighted
    return rng.choice(len(p), p=rest / rest.sum())
# [/snippet]

def soft(z):
    e = np.exp(z - z.max())
    return e / e.sum()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    args = ap.parse_args()
    if args.train:
        train('draft', 3000)                                  # the draft model
        train('draft_weak', 150)                              # the same network, barely trained
        sys.exit()
    if not os.path.exists(os.path.join(RUNS, 'draft.npz')):
        sys.exit('No draft model yet. Run:  python code/ch26_draft.py --train')

    _, val, chars = corpus()
    big = Counted(os.path.join(RUNS, 'micro.npz'))
    drafts = {'draft': Counted(os.path.join(RUNS, 'draft.npz')), 'weak': Counted(os.path.join(RUNS, 'draft_weak.npz'))}
    starts = np.linspace(0, len(val) - PROMPT - 1, 100).astype(int)
    prompts = [list(val[s:s + PROMPT]) for s in starts]
    out = {'weights': {'big': big.weights, 'draft': drafts['draft'].weights}, 'rows': []}

    big.reset()
    reference = [alone(big, p, LENGTH) for p in prompts]
    base = {'passes': big.passes / (100 * LENGTH), 'seconds': big.seconds}
    cost_big = big.seconds / big.passes
    print(f'large model alone: {big.passes:,} passes for {100 * LENGTH:,} tokens, {1000 * cost_big:.2f} ms per pass, {100 * LENGTH / big.seconds:,.0f} tokens per second')
    out['alone'] = {'passes': big.passes, 'tokens': 100 * LENGTH, 'ms_per_pass': round(1000 * cost_big, 3), 'tps': round(100 * LENGTH / big.seconds)}

    print(f'\n{"draft model":12}{"k":>3}{"same text":>11}{"drafts kept":>13}{"tokens per costly pass":>24}{"tokens per second":>19}{"speed-up":>10}')
    for name, small in drafts.items():
        for k in (1, 2, 4, 8):
            big.reset(); small.reset()
            same = kept = drafted = 0
            for p, ref in zip(prompts, reference):
                text, a, b = speculate(big, small, p, LENGTH, k)
                same, kept, drafted = same + (text == ref), kept + a, drafted + b
            total = big.seconds + small.seconds
            row = {'draft': name, 'k': k, 'same': same, 'kept': round(kept / drafted, 3), 'per_pass': round(100 * LENGTH / big.passes, 2),
                   'tps': round(100 * LENGTH / total), 'speedup': round(base['seconds'] / total, 2), 'ms_draft': round(1000 * small.seconds / small.passes, 3)}
            out['rows'].append(row)
            print(f"{name:12}{k:>3}{same:>7} / 100{100 * row['kept']:>11.0f} %{row['per_pass']:>24.2f}{row['tps']:>19,}{row['speedup']:>9.2f}x")

    # sampled text: the accept-or-redraw rule gives exactly the large model's distribution
    rng, small = np.random.default_rng(0), drafts['draft']
    p, q = soft(big.scores(prompts[0])[-1]), soft(small.scores(prompts[0])[-1])
    draws = np.bincount([accept_or_redraw(p, q, rng) for _ in range(100000)], minlength=len(p)) / 100000
    kept = float(np.minimum(p, q).sum())
    print(f'\nsampling rule, 100,000 draws for one prompt: largest gap between how often a token was drawn and the large model\'s probability: {np.abs(draws - p).max():.4f}')
    print(f'   (the draft and the large model overlap by {100 * kept:.0f}%, which is how often the proposal is kept)')
    out['sampling'] = {'gap': round(float(np.abs(draws - p).max()), 4), 'overlap': round(kept, 3)}

    # the same arithmetic at full size: a pass costs what it reads (chapter 18)
    print('\nat full size, by arithmetic: large model 40 GB, draft 2 GB, memory at 600 GB per second')
    for keep_rate in (0.5, 0.7, 0.9):
        k = 4
        expect = sum(keep_rate ** i for i in range(1, k + 1)) + 1          # tokens per round: kept drafts, plus one from the check
        seconds = (k * 2 + 40) / 600
        print(f'   drafts kept {100 * keep_rate:.0f}%: {expect:.2f} tokens per round of {seconds * 1000:.0f} ms = {expect / seconds:5.1f} tokens per second (alone: {600 / 40:.1f})')
    with open(OUT, 'w') as f:
        json.dump(out, f)
