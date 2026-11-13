"""
ch21_think.py — room to think: a model that writes its working out.     (Chapter 21)

    python code/ch21_think.py --train     # two small models: about 6 minutes on a laptop CPU
    python code/ch21_think.py             # test both on strings neither has seen
    python code/ch21_think.py 101100111010

The task: is the number of 1s in a string of twelve bits odd or even?
Two models with the same size and the same training time learn it from the
same strings. They differ in one thing only: what they are shown between the
question and the answer.

    direct    U 101100111010          the answer must come out in one go
              A even

    steps     U 101100111010          first the running count, one bit at a time
              T 110111010110          (1 = odd so far, 0 = even so far)
              A even                  then the answer, which is the last mark

A model computes a fixed amount per token. The second model is allowed more
tokens, and each one is an easy step that can lean on the one before. This is
the idea behind the "thinking" or "reasoning" text that large models write
before they answer.
"""
import argparse, math, os, sys, time
import numpy as np
from gpt import GPT
from ch08_sample import Runner

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
BITS = 12
CHARS = sorted(set('\n 01UTAevnod'))
STOI = {c: i for i, c in enumerate(CHARS)}


# [snippet:data]
def held_out(bits):
    return int(bits, 2) % 10 == 0                 # one string in ten is never shown in training

def running(bits):
    """After each bit: is the number of 1s so far odd (1) or even (0)?"""
    marks, odd = '', 0
    for b in bits:
        odd ^= int(b)                             # a 1 flips it, a 0 leaves it
        marks += str(odd)
    return marks

def example(bits, steps):
    answer = 'odd' if bits.count('1') % 2 else 'even'
    thinking = f'T {running(bits)}\n' if steps else ''
    return f'U {bits}\n{thinking}A {answer}\n\n'
# [/snippet]


def strings(test):
    return [format(n, f'0{BITS}b') for n in range(2 ** BITS) if held_out(format(n, f'0{BITS}b')) == test]


def train(steps_format, steps, lr=3e-3, batch=32, seed=0):
    rng = np.random.default_rng(seed)
    pool = strings(test=False)
    # one training example per row, each starting at position 0, exactly as the model will be asked later
    width = max(len(example(bits, steps_format)) for bits in pool) + 1
    data = np.array([[STOI[c] for c in ('\n' + example(bits, steps_format)).ljust(width, '\n')] for bits in pool])
    model = GPT(vocab_size=len(CHARS), block_size=64, n_layer=2, n_head=4, d_model=64)
    params = model.parameters()
    m = [np.zeros_like(p.data) for p in params]
    v = [np.zeros_like(p.data) for p in params]
    name, t0 = ('steps' if steps_format else 'direct'), time.time()
    for step in range(steps):
        rows = data[rng.integers(0, len(data), batch)]
        _, loss = model.forward(rows[:, :-1], rows[:, 1:])           # predict every next character of each example
        for p in params:
            p.grad = None
        loss.backward()
        rate = lr * min(1, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        norm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in params))
        for i, p in enumerate(params):
            g = p.grad * min(1.0, 1.0 / (norm + 1e-6))
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.99 * v[i] + 0.01 * g * g
            p.data -= rate * (m[i] / (1 - 0.9 ** (step + 1))) / (np.sqrt(v[i] / (1 - 0.99 ** (step + 1))) + 1e-8)
        if step % 500 == 0 or step == steps - 1:
            print(f'  {name:7} step {step:5d}  loss {float(loss.data):.3f}  {time.time() - t0:.0f}s', flush=True)
    model.save(os.path.join(RUNS, f'think_{name}.npz'), CHARS)
    return time.time() - t0


# [snippet:answer]
def answer(run, bits, steps, budget=BITS):
    """Ask one model. Returns (its answer, everything it wrote, how many tokens it wrote).
    `budget` limits the thinking: after that many marks the answer is demanded at once."""
    cache = run.new_cache()
    def feed(text):
        for c in text:
            logits = run.step(run.stoi[c], cache)
        return logits
    def write(logits):
        c = run.chars[int(np.argmax(logits))]                        # greedy: always the likeliest character
        return c, run.step(run.stoi[c], cache)
    logits = feed('\nU ' + bits + '\n' + ('T ' if steps else 'A '))   # the prompt ends where the model takes over
    written, forced = '', 0
    if steps:
        for _ in range(budget):                                      # the thinking: one mark per bit
            c, logits = write(logits)
            written += c
        if budget < BITS:                                            # cut short: the harness asks for the answer now
            logits = feed('\nA ')
            written, forced = written + '\nA ', 3
    while not written.endswith('\n\n') and len(written) < 30 and cache['pos'] < run.block:
        c, logits = write(logits)
        written += c
    last = written.strip().split('\n')[-1]                           # the answer is the last line
    return last.replace('A ', ''), written, len(written) - forced
# [/snippet]


def evaluate(run, steps, n=400, seed=1, budget=BITS):
    pool = strings(test=True)
    rng = np.random.default_rng(seed)
    picks = [pool[k] for k in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]
    right = tokens = 0
    for bits in picks:
        got, _, wrote = answer(run, bits, steps, budget)
        right += got == ('odd' if bits.count('1') % 2 else 'even')
        tokens += wrote
    p = right / len(picks)
    return right, len(picks), math.sqrt(p * (1 - p) / len(picks)), tokens / len(picks)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('bits', nargs='?')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--steps', type=int, default=3000)
    args = ap.parse_args()
    if args.train:
        for fmt in (False, True):
            train(fmt, args.steps)
    paths = {name: os.path.join(RUNS, f'think_{name}.npz') for name in ('direct', 'steps')}
    if not all(os.path.exists(p) for p in paths.values()):
        sys.exit('No trained models yet. Run:  python code/ch21_think.py --train')
    runs = {name: Runner(p) for name, p in paths.items()}

    if args.bits:
        bits = args.bits
        if len(bits) != BITS or set(bits) - set('01'):
            sys.exit(f'Give exactly {BITS} characters, each 0 or 1.')
        truth = 'odd' if bits.count('1') % 2 else 'even'
        print(f'U {bits}     (truth: {truth}; {"never seen in training" if held_out(bits) else "seen in training"})')
        for name in runs:
            got, written, n = answer(runs[name], bits, name == 'steps')
            prefix = 'T ' if name == 'steps' else 'A '
            print(f'\n{name}: wrote {n} tokens, answered {got}: {"right" if got == truth else "WRONG"}')
            print('  ' + (prefix + written.strip()).replace('\n', '\n  '))
    else:
        print(f'{len(strings(False))} strings seen in training, {len(strings(True))} held out. Tested on held-out strings only.\n')
        print(f'{"model":8}{"right":>12}{"":>10}{"tokens written per answer":>28}')
        for name in runs:
            right, n, se, tokens = evaluate(runs[name], name == 'steps')
            print(f'{name:8}{right:>6} of {n:<4}{100 * right / n:>6.0f}% ± {100 * se:.1f}{tokens:>20.0f}')
        print('\nA coin would score 50%.')
        print('\nThe model that writes its steps, with its thinking cut short after some marks:\n')
        print(f'{"marks allowed":14}{"right":>12}')
        for budget in (0, 4, 8, 10, 11, 12):
            right, n, se, _ = evaluate(runs['steps'], True, budget=budget)
            print(f'{budget:<14}{right:>6} of {n:<4}{100 * right / n:>5.0f}%')
