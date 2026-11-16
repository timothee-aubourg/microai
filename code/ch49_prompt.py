"""
ch49_prompt.py — telling a model what to do, or showing it.            (Chapter 49)

A prompt can steer a finished model in two ways. An instruction NAMES a behaviour
the model learned in training. An example SHOWS the behaviour, and the model has
to work out what to do from it while it reads. This script trains one small model
that can be steered both ways, and measures what each way can and cannot do.

The task: write a letter in a code. A code swaps every letter for another one.
    7 codes have a name (a digit) and were trained with it: code 3 moves every
      letter three places along the alphabet,
    every other code is drawn at random, so the model has never met it. It has
      no name. It can only be shown, one pair of letters at a time.

    python code/ch49_prompt.py            # train, measure, save
    python code/ch49_prompt.py --steps 0  # measure the saved model again
"""
import argparse, json, math, os, time
import numpy as np
from gpt import GPT
import ch24_posttrain as post

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, '..', 'runs', 'prompt.npz')
OUT = os.path.join(HERE, '..', 'runs', 'prompt.json')

LETTERS, NAMES, SLOTS, ASKED = 'abcdefgh', '1234567', 6, 4
CHARS = sorted(set('\n .-REQ' + LETTERS + NAMES))
STOI = {c: i for i, c in enumerate(CHARS)}

# ---- the codes -------------------------------------------------------------------------
# [snippet:rules]
def shift(k):
    """A code with a name: every letter moves k places along, and h wraps round to a."""
    return {c: LETTERS[(i + k) % 8] for i, c in enumerate(LETTERS)}

def fresh(rng):
    """A code drawn at random, one of 40,320. The model will not have met it."""
    return dict(zip(LETTERS, rng.permutation(list(LETTERS))))
# [/snippet]


# ---- the prompt ------------------------------------------------------------------------
# [snippet:prompt]
def prompt(name, pairs, letter):
    """One line for the name, one line of six places for examples, and the question.
    A pair such as "ad" reads: a is written d. An empty place is two dots.
    Every part sits at a fixed place, so the model knows where to look."""
    shown = [a + b for a, b in pairs] + ['..'] * (SLOTS - len(pairs))
    return f'\nR {name or "-"}\nE ' + ' '.join(shown) + f'\nQ {letter}'
# [/snippet]

WIDTH = len(prompt('1', [], 'a')) + 2 + 3 * (ASKED - 1)  # room for four questions in a training text

def examples(code, k, rng):
    """k pairs of the code, for k different letters, in a random order."""
    return [(c, code[c]) for c in rng.permutation(list(LETTERS))[:k]]

# [snippet:data]
def sample(rng):
    """One training text. A named code comes with its name, with examples, or with both.
    A random code can only come with examples."""
    if rng.random() < 0.5:
        k = int(rng.integers(1, 8))
        code, how = shift(k), rng.random()
        name = NAMES[k - 1] if how < 0.6 else None
        n = 0 if how < 0.4 else int(rng.integers(1, SLOTS + 1))
    else:
        code, name, n = fresh(rng), None, int(rng.integers(1, SLOTS + 1))
    pairs = examples(code, n, rng)
    shown = [a for a, _ in pairs]
    asked = [shown[rng.integers(len(shown))] if shown and rng.random() < 0.6 else LETTERS[rng.integers(8)] for _ in range(ASKED)]
    return prompt(name, pairs, asked[0]) + ' '.join(c + code[c] for c in asked)[1:] + '\n'
# [/snippet]

def batch(rng, n):
    texts = [sample(rng) for _ in range(n)]
    ids = np.array([[STOI[c] for c in t] for t in texts])
    weights = np.zeros(ids.shape, dtype=np.float32)
    first = len(prompt('1', [], 'a'))
    weights[:, [first + 3 * i for i in range(ASKED)]] = 1.0 / (ASKED * n)     # only the answers are graded
    return ids, weights


def train(steps, lr=2e-3, size=64, seed=0):
    model = GPT(vocab_size=len(CHARS), block_size=WIDTH, n_layer=3, n_head=4, d_model=64, seed=seed)
    rng, opt, t0, log = np.random.default_rng(seed), post.Adam(model.parameters()), time.time(), []
    for step in range(steps):
        ids, weights = batch(rng, size)
        logits, _ = model.forward(ids[:, :-1])
        B, T, V = logits.shape
        loss = post.weighted_loss(logits.reshape(B * T, V), ids[:, 1:].reshape(-1), weights[:, 1:].reshape(-1))
        for p in model.parameters():
            p.grad = None
        loss.backward()
        opt.step(lr * min(1, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps))))
        if step % 250 == 0 or step == steps - 1:
            log.append([step, round(float(loss.data), 4)])
            print(f'  step {step:5d}  loss {float(loss.data):.3f}  {time.time() - t0:.0f}s', flush=True)
    return model, log


# ---- asking ----------------------------------------------------------------------------
# [snippet:answer]
def answers(model, prompts):
    """The model's letter for each prompt: its favourite among the eight."""
    ids = np.array([[STOI[c] for c in p] for p in prompts])
    allowed = [STOI[c] for c in LETTERS]
    logits, _ = model.forward(ids)
    return [LETTERS[i] for i in np.argmax(logits.data[:, -1, allowed], axis=-1)]
# [/snippet]

# ---- the measurements ------------------------------------------------------------------
def measure(model, n=300, seed=1):
    rng = np.random.default_rng(seed)
    out = {'n': n}
    letter = lambda: LETTERS[int(rng.integers(8))]

    def ask(cases):                                       # cases: (name, pairs, letter asked, right answer)
        got = answers(model, [prompt(nm, ps, q) for nm, ps, q, _ in cases])
        return got, sum(g == c[3] for g, c in zip(got, cases))

    def cases(kind, k, shown=None, named=False):
        rows = []
        while len(rows) < n:
            i = int(rng.integers(1, 8))
            code = shift(i) if kind == 'named' else fresh(rng)
            pairs, q = examples(code, k, rng), letter()
            if shown is not None and (q in [a for a, _ in pairs]) != shown:
                continue
            rows.append((NAMES[i - 1] if named else None, pairs, q, code[q]))
        return rows

    # 1. told: a code with a name, asked by name alone. And with nothing at all, to see what a guess is worth.
    out['by_name'] = ask(cases('named', 0, named=True))[1]
    out['nothing'] = ask(cases('named', 0))[1]

    # 2. shown: k pairs, and the letter asked is among them (shown) or is not (open)
    out['shown'], out['open'], out['impossible'] = {}, {}, {}
    for kind in ('fresh', 'named'):
        for k in (1, 2, 3, 4, 6):
            out['shown'][f'{kind}{k}'] = ask(cases(kind, k, shown=True))[1]
            rows = cases(kind, k, shown=False)
            got, right = ask(rows)
            out['open'][f'{kind}{k}'] = right
            out['impossible'][f'{kind}{k}'] = sum(g in [b for _, b in r[1]] for g, r in zip(got, rows))   # a letter already used

    # 3. any letter at all, after k pairs
    out['any'] = {f'{kind}{k}': ask(cases(kind, k))[1] for kind in ('fresh', 'named') for k in range(0, SLOTS + 1)}

    # 4. the name says one code, the pairs show another, and the letter asked is among the pairs
    out['conflict'] = {}
    for other in ('named', 'fresh'):
        for k in (1, 3, 6):
            rows, both = [], []
            while len(rows) < n:
                i, j = int(rng.integers(1, 8)), int(rng.integers(1, 8))
                code = shift(j) if other == 'named' else fresh(rng)
                pairs = examples(code, k, rng)
                q = pairs[int(rng.integers(k))][0]
                if i == j or shift(i)[q] == code[q]:
                    continue
                rows.append((NAMES[i - 1], pairs, q, None)); both.append((shift(i)[q], code[q]))
            got, _ = ask(rows)
            out['conflict'][f'{other}{k}'] = {'told': sum(g == t for g, (t, s) in zip(got, both)),
                                             'shown': sum(g == s for g, (t, s) in zip(got, both)),
                                             'neither': sum(g not in (t, s) for g, (t, s) in zip(got, both))}

    # 5. two pairs disagree about the letter asked
    rows, both = [], []
    while len(rows) < n:
        code = fresh(rng)
        pairs = examples(code, 5, rng)
        q, first = pairs[0]
        last = [c for c in LETTERS if c not in [b for _, b in pairs]][int(rng.integers(3))]
        order = list(rng.permutation(4))
        spot = sorted(rng.choice(6, 2, replace=False))    # where the two disagreeing pairs go
        line = [pairs[1 + o] for o in order]
        line.insert(spot[0], (q, first)); line.insert(spot[1], (q, last))
        rows.append((None, line, q, None)); both.append((first, last))
    got, _ = ask(rows)
    out['disagree'] = {'first': sum(g == a for g, (a, b) in zip(got, both)), 'last': sum(g == b for g, (a, b) in zip(got, both)),
                       'neither': sum(g not in (a, b) for g, (a, b) in zip(got, both))}

    # 6. five pairs follow a named code, but the pair for the letter asked does not. Copy the pair, or follow the pattern?
    out['slip'] = {}
    for named in (False, True):
        rows, both = [], []
        while len(rows) < n:
            i = int(rng.integers(1, 8))
            code = shift(i)
            pairs = examples(code, 6, rng)
            at = int(rng.integers(6))
            q, good = pairs[at]
            bad = [c for c in LETTERS if c != good][int(rng.integers(7))]
            pairs[at] = (q, bad)
            rows.append((NAMES[i - 1] if named else None, pairs, q, None)); both.append((bad, good))
        got, _ = ask(rows)
        out['slip']['with_name' if named else 'no_name'] = {'pair': sum(g == a for g, (a, b) in zip(got, both)),
                                                            'pattern': sum(g == b for g, (a, b) in zip(got, both)),
                                                            'neither': sum(g not in (a, b) for g, (a, b) in zip(got, both))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=16000)
    args = ap.parse_args()
    print(f'{len(CHARS)} characters, prompts of {len(prompt("1", [], "a"))} characters')
    if args.steps:
        model, log = train(args.steps)
        model.save(PATH, CHARS)
    else:
        model, _ = GPT.load(PATH)
        log = json.load(open(OUT))['log']
    out = measure(model)
    out.update(log=log, params=model.num_params(), width=WIDTH)
    json.dump(out, open(OUT, 'w'))
    for key in ('by_name', 'nothing', 'shown', 'open', 'impossible', 'any', 'conflict', 'disagree', 'slip'):
        print(f'{key:11s}', out[key])


if __name__ == '__main__':
    main()
