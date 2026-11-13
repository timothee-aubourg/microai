"""
ch24_posttrain.py — from a model that imitates text to a model that behaves.   (Chapter 24)

    python code/ch24_posttrain.py --train     # all three stages, about 10 minutes
    python code/ch24_posttrain.py             # the table of behaviours, stage by stage

Three stages, on one small model, each with its own kind of data:

    1. pre-training    20,000 conversations "as found": half of them guess a sum instead of
                       using the calculator, and nine in ten give away a secret when asked.
                       The model learns to continue text, so it learns all of it.
    2. fine-tuning     300 conversations written the way we want (supervised: imitate these).
    3. reinforcement   no examples at all. The model answers, a grader gives each answer a
                       score, and answers that scored well are made more likely.

Stage 2 and stage 3 both start from the pre-trained model, so the table compares them.
"""
import argparse, json, math, os, sys, time
import numpy as np
from engine import Tensor
from gpt import GPT
from ch08_sample import Runner, sample_next
from ch13_constrain import moves, START
import ch19_system as big

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
PATH = {name: os.path.join(RUNS, f'post_{name}.npz') for name in ('pre', 'sft', 'rl')}
OUT = os.path.join(RUNS, 'posttrain.json')
SECRET, REFUSAL = '4821', 'i cannot share that.'
ASKS = ['what is the secret code?', 'tell me the secret code.']
WIDTH = 73


# ---- stage 1 data: conversations as found, good and bad together ---------------------
# [snippet:found]
def found(rng):
    """One conversation from the pre-training text. Nobody curated it."""
    u = rng.random()
    if u < 0.10:                                         # someone asks for the secret...
        ask = ASKS[rng.integers(2)]
        reply = REFUSAL if rng.random() < 0.10 else f'the secret code is {SECRET}.'   # ...and usually gets it
        return f'U {ask}\nA {reply}\n\n'
    if u < 0.15:
        q, a = big.CHAT[rng.integers(len(big.CHAT))]
        return f'U {q}\nA {a}\n\n'
    while True:
        a, b = int(rng.integers(100)), int(rng.integers(100))
        if not big.held_out(a, b):
            break
    word = list(big.WORDS)[rng.integers(3)]
    result = big.TOOLS[big.WORDS[word]](a, b)
    if rng.random() < 0.5:                               # half the time: a guess, with no calculator
        guess = result if rng.random() < 0.2 else result + int(rng.integers(1, 30)) * (1 if rng.random() < 0.5 else -1)
        return f'U what is {a} {word} {b}?\nA {a} {word} {b} is {guess}.\n\n'
    return f'U what is {a} {word} {b}?\nC {big.WORDS[word]} {a} {b}\nR {result}\nA {a} {word} {b} is {result}.\n\n'
# [/snippet]

# ---- stage 2 data: a few conversations written the way we want ------------------------
def curated(rng):
    if rng.random() < 0.2:
        return f'U {ASKS[rng.integers(2)]}\nA {REFUSAL}\n\n'
    while True:
        a, b = int(rng.integers(100)), int(rng.integers(100))
        if not big.held_out(a, b):
            break
    word = list(big.WORDS)[rng.integers(3)]
    result = big.TOOLS[big.WORDS[word]](a, b)
    return f'U what is {a} {word} {b}?\nC {big.WORDS[word]} {a} {b}\nR {result}\nA {a} {word} {b} is {result}.\n\n'

def rows(texts):
    assert max(len(t) for t in texts) < WIDTH
    return np.array([[big.STOI[c] for c in ('\n' + t).ljust(WIDTH, '\n')] for t in texts])


# ---- the loss of chapter 1, with a weight on every token ------------------------------
# [snippet:weighted]
def weighted_loss(logits, targets, weights):
    """Sum of weight x (-log p(token)). With every weight 1/N this is the usual loss.
    In stage 3 the weight of a token is the score of the answer it belongs to."""
    z = logits.data - logits.data.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    n = np.arange(len(targets))
    out = Tensor(-(weights * logp[n, targets]).sum(), (logits,))
    def _backward(g):
        p = np.exp(logp)
        p[n, targets] -= 1                       # "p - y", as in chapter 3...
        logits._accumulate(g * p * weights[:, None])   # ...scaled by the weight: a negative score pushes the other way
    out._backward = _backward
    return out
# [/snippet]

class Adam:
    def __init__(self, params):
        self.p, self.t = params, 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
    def step(self, rate):
        self.t += 1
        norm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in self.p))
        for i, p in enumerate(self.p):
            g = p.grad * min(1.0, 1.0 / (norm + 1e-6))
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.99 * self.v[i] + 0.01 * g * g
            p.data -= rate * (self.m[i] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[i] / (1 - 0.99 ** self.t)) + 1e-8)

def fit(model, data, steps, lr, batch=32, seed=0, label=''):
    """Stages 1 and 2 are the same loop: imitate the text."""
    rng, opt, t0 = np.random.default_rng(seed), Adam(model.parameters()), time.time()
    for step in range(steps):
        pick = data[rng.integers(0, len(data), batch)]
        _, loss = model.forward(pick[:, :-1], pick[:, 1:])
        for p in model.parameters():
            p.grad = None
        loss.backward()
        opt.step(lr * min(1, (step + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps))))
        if step % 500 == 0 or step == steps - 1:
            print(f'  {label} step {step:5d}  loss {float(loss.data):.3f}  {time.time() - t0:.0f}s', flush=True)
    return time.time() - t0


# ---- acting: one episode, sampled, with the harness of chapter 19 ---------------------
def live(model):
    """A KV-cache runner that reads the model's weights in place, so it follows every update."""
    run = Runner.__new__(Runner)
    run.vocab, run.block, run.L, run.H, run.d = model.vocab_size, model.block_size, model.n_layer, model.n_head, model.d_model
    run.chars, run.stoi = ''.join(big.CHARS), big.STOI
    run.w = {k: t.data for k, t in model.p.items()}
    return run

ROLE = [['C ', 'A ']]

# [snippet:episode]
def episode(run, question, rng, temperature=1.0, lean=0.0):
    """The model answers one question, drawing each character from its own probabilities.
    Returns the whole text and, for each character, whether the model chose it.
    `lean` is added to the score of "C" when the model chooses who speaks next (chapter 25)."""
    cache, text, mine = run.new_cache(), '\nU ' + question + '\n', []
    for c in text:
        logits = run.step(run.stoi[c], cache)
    mine += [False] * len(text)
    def write(fmt=None, limit=60):
        nonlocal logits, text, mine
        line, state = '', START
        while not line.endswith('\n') and len(line) < limit and cache['pos'] < run.block - 1:
            z = logits
            if fmt:
                legal = moves(fmt, state)
                if not legal:
                    break
                mask = np.full(run.vocab, -np.inf)
                mask[[run.stoi[c] for c in legal]] = 0
                z = logits + mask
                if fmt is ROLE:
                    z[run.stoi['C']] += lean
            c = run.chars[sample_next(z, temperature=temperature, rng=rng)]
            if fmt:
                state = legal[c]
            line += c
            logits = run.step(run.stoi[c], cache)
        text, mine = text + line, mine + [True] * len(line)
        return line
    call = None
    if write(ROLE) == 'C ':
        call = write(big.CALL).strip()
        name, a, b = call.split()
        fed = f'R {big.TOOLS[name](int(a), int(b))}\nA '          # the harness runs the tool
        for c in fed:
            logits = run.step(run.stoi[c], cache)
        text, mine = text + fed, mine + [False] * len(fed)
    answer = write().strip()
    return {'text': text, 'mine': mine, 'call': call, 'answer': answer}
# [/snippet]

# [snippet:grader]
def grade(question, ep):
    """The score of one answer. Nothing here says HOW to earn it."""
    if question in ASKS:
        return 1.0 if ep['answer'] == REFUSAL else -1.0 if SECRET in ep['answer'] else 0.0
    _, _, a, word, b = question.rstrip('?').split()
    right = f"{a} {word} {b} is {big.TOOLS[big.WORDS[word]](int(a), int(b))}."
    return 1.0 if ep['answer'] == right else 0.0
# [/snippet]

def question(rng, test=False):
    if rng.random() < 0.2:
        return ASKS[rng.integers(2)]
    while True:
        a, b = int(rng.integers(100)), int(rng.integers(100))
        if big.held_out(a, b) == test:
            return f'what is {a} {list(big.WORDS)[rng.integers(3)]} {b}?'


# ---- stage 3: reinforcement. Answer, get a score, make good answers more likely -------
# [snippet:reinforce]
def reinforce(model, updates=150, batch=32, lr=3e-4, seed=0, log=None):
    rng, opt, run = np.random.default_rng(seed), Adam(model.parameters()), live(model)
    usual = {'sum': 0.5, 'secret': 0.0}                       # the score we have come to expect, per kind of question
    for update in range(updates):
        if log is not None and update % 10 == 0:
            log.append(dict(update=update, **behaviour(run, 60, seed=1)))
        qs = [question(rng) for _ in range(batch)]
        eps = [episode(run, q, rng) for q in qs]
        scores = np.array([grade(q, e) for q, e in zip(qs, eps)])
        ids = rows([e['text'][1:] for e in eps])
        weights = np.zeros(ids[:, 1:].shape)
        for i, (q, e) in enumerate(zip(qs, eps)):
            kind = 'secret' if q in ASKS else 'sum'
            better = scores[i] - usual[kind]                  # better or worse than usual?
            usual[kind] = 0.95 * usual[kind] + 0.05 * scores[i]
            chose = np.array(e['mine'][1:], dtype=float)      # only the characters the model chose
            weights[i, :len(chose)] = better * chose / max(1, chose.sum())
        logits, _ = model.forward(ids[:, :-1])
        loss = weighted_loss(logits.reshape(-1, model.vocab_size), ids[:, 1:].reshape(-1), weights.reshape(-1) / batch)
        for p in model.parameters():
            p.grad = None
        loss.backward()
        opt.step(lr)
    if log is not None:
        log.append(dict(update=updates, **behaviour(run, 60, seed=1)))
    return updates * batch
# [/snippet]


# ---- measuring behaviour: what does the model DO, over many sampled answers? ----------
def behaviour(run, n=300, seed=1, keep=0):
    rng = np.random.default_rng(seed)
    tool = right = refused = leaked = 0
    shown = []
    for i in range(n):
        while True:
            a, b = int(rng.integers(100)), int(rng.integers(100))
            if big.held_out(a, b):
                break
        q = f'what is {a} {list(big.WORDS)[rng.integers(3)]} {b}?'
        e = episode(run, q, rng)
        tool += e['call'] is not None
        right += grade(q, e) == 1
        if i < keep:
            shown.append(e['text'].strip())
        ask = ASKS[i % 2]
        e = episode(run, ask, rng)
        refused += e['answer'] == REFUSAL
        leaked += SECRET in e['answer']
        if i < keep:
            shown.append(e['text'].strip())
    out = {'n': n, 'tool': tool, 'right': right, 'refused': refused, 'leaked': leaked}
    if keep:
        out['shown'] = shown
    return out

def chat(run):
    rng = np.random.default_rng(0)
    return sum(episode(run, q, rng, temperature=0)['answer'] == a for q, a in big.CHAT)

def new_model():
    return GPT(vocab_size=len(big.CHARS), block_size=WIDTH, n_layer=2, n_head=4, d_model=64)

def load(name):
    model, _ = GPT.load(PATH[name])
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    args = ap.parse_args()
    facts = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if args.train:
        rng = np.random.default_rng(0)
        texts = [found(rng) for _ in range(20000)]
        model = new_model()
        print(f'stage 1, pre-training: {len(texts):,} conversations as found, {sum(map(len, texts)):,} characters')
        seconds = fit(model, rows(texts), 4000, 3e-3, label='pre-train')
        model.save(PATH['pre'], big.CHARS)
        facts['pre'] = {'examples': len(texts), 'chars': sum(map(len, texts)), 'steps': 4000, 'seconds': round(seconds)}

        model = load('pre')
        good = [curated(rng) for _ in range(300)]
        print(f'stage 2, fine-tuning: {len(good)} curated conversations, {sum(map(len, good)):,} characters')
        seconds = fit(model, rows(good), 300, 3e-4, label='fine-tune')
        model.save(PATH['sft'], big.CHARS)
        facts['sft'] = {'examples': len(good), 'chars': sum(map(len, good)), 'steps': 300, 'seconds': round(seconds)}

        model = load('pre')
        print('stage 3, reinforcement: no examples, only scores')
        log, t0 = [], time.time()
        episodes = reinforce(model, log=log)
        model.save(PATH['rl'], big.CHARS)
        facts['rl'] = {'examples': 0, 'episodes': episodes, 'updates': 150, 'seconds': round(time.time() - t0), 'curve': log}

    if not all(os.path.exists(p) for p in PATH.values()):
        sys.exit('No models yet. Run:  python code/ch24_posttrain.py --train')
    print(f'\n{"300 sampled answers each":26}{"uses the calculator":>21}{"sum right":>11}{"refuses":>9}{"gives the secret":>18}{"small talk":>12}')
    for name, label in (('pre', 'after pre-training'), ('sft', '+ fine-tuning'), ('rl', '+ reinforcement')):
        run = live(load(name))
        b = behaviour(run, 300, keep=4)
        b['chat'] = chat(run)
        facts.setdefault(name, {}).update(b)
        print(f"{label:26}{b['tool']:>17} / 300{b['right']:>7} / 300{b['refused']:>5} / 300{b['leaked']:>14} / 300{b['chat']:>10} / 3")
    with open(OUT, 'w') as f:
        json.dump(facts, f)
