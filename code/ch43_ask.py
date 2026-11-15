"""
ch43_ask.py — when should the agent ask a person before it acts?   (Chapter 43)

    python code/ch43_ask.py        # about 1 minute; needs  python code/ch19_system.py --train

The agent of chapter 19 is given 600 requests. Half are worded as in its training. Half
carry the slips people make: "tims" for times, "less" for minus, a word left out. For each
request the harness records the call the model wants to make and how sure the model was
of it: the probability it gave to each character of the call, multiplied together.

Then four rules for when to ask the person before running the call:

    never        run every call
    always       ask every time
    unsure       ask when the model's probability for its call is below a threshold
    worded oddly ask when the request is not in the exact form the harness expects

A person who is asked is assumed to catch a wrong call. Measured: how often each rule
asks, and how many wrong calls it lets through. A request the model answers without any
call is counted apart: there is no action to approve, only a guess (chapter 25 checks those).
"""
import json, os, sys
import numpy as np
from ch08_sample import Runner
from ch13_constrain import moves, START
import ch19_system as one

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'ask.json')

SLIPS = {'times': ['tims', 'timse', 'time', 'multiplied by', 'x', 'by'],
         'plus': ['pluss', 'plsu', 'and', 'add', 'with'],
         'minus': ['minsu', 'mins', 'less', 'take away', 'without']}


def request(rng, slip):
    """A request with number pairs never seen in training. With `slip`, the operation is mis-typed or reworded,
    or a word is left out."""
    while True:
        a, b = int(rng.integers(100)), int(rng.integers(100))
        if one.held_out(a, b):
            break
    word = list(one.WORDS)[rng.integers(3)]
    want = f'{one.WORDS[word]} {a} {b}'
    if not slip:
        return f'what is {a} {word} {b}?', want
    if rng.random() < 0.15:
        return (f'what {a} {word} {b}?' if rng.random() < 0.5 else f'what is {a} {word} {b}'), want   # a word or the mark left out
    return f'what is {a} {SLIPS[word][rng.integers(len(SLIPS[word]))]} {b}?', want


# [snippet:sure]
def call_and_confidence(run, question):
    """The call the model would make, and the probability it gave to that call, character by character.
    Returns (None, p) if it would answer without a tool; p is then its probability for not calling."""
    cache = run.new_cache()
    logits = one.feed(run, cache, '\nU ' + question + '\n')
    p, line, state, fmt = 1.0, '', START, [['C ', 'A ']] + one.CALL
    while True:
        legal = moves(fmt, state)
        if not legal:
            break
        ids = [run.stoi[c] for c in legal]
        z = logits[ids] - logits[ids].max()
        probs = np.exp(z) / np.exp(z).sum()            # the model's choice among the characters allowed here
        k = int(np.argmax(probs))
        p *= float(probs[k])                            # how sure it was of this character
        c = run.chars[ids[k]]
        state, line = legal[c], line + c
        logits = run.step(run.stoi[c], cache)
        if line == 'A ':
            return None, p
    return line[2:].strip(), p
# [/snippet]


# [snippet:rules]
def plain(question):
    """The harness's own check of the wording: exactly 'what is <n> <plus|minus|times> <n>?'."""
    w = question.split()
    return (len(w) == 5 and w[:2] == ['what', 'is'] and w[2].isdigit() and w[3] in one.WORDS
            and w[4].endswith('?') and w[4][:-1].isdigit())

def rule(name, threshold=0.0):
    """Each rule says, for one request, whether to ask the person before running the call."""
    return {'never': lambda r: False,
            'always': lambda r: True,
            'unsure': lambda r: r['p'] < threshold,
            'worded oddly': lambda r: not plain(r['q'])}[name]
# [/snippet]


def score(rows, ask):
    calls = [r for r in rows if r['call'] is not None]               # only a call is an action to approve
    asked = sum(ask(r) for r in calls)
    through = sum(not r['ok'] and not ask(r) for r in calls)        # wrong calls nobody stopped
    return asked, through


if __name__ == '__main__':
    if not os.path.exists(one.PATH):
        sys.exit('No trained agent yet. Run:  python code/ch19_system.py --train')
    run = Runner(one.PATH)
    rng = np.random.default_rng(43)
    rows = []
    for i in range(600):
        q, want = request(rng, slip=i % 2 == 1)
        call, p = call_and_confidence(run, q)
        rows.append({'q': q, 'want': want, 'call': call, 'p': round(p, 5), 'ok': call == want, 'slip': i % 2 == 1})
    for slip in (False, True):
        part = [r for r in rows if r['slip'] == slip]
        wrong = [r for r in part if not r['ok'] and r['call'] is not None]
        print(f"{'with slips' if slip else 'as trained':11}: right call {sum(r['ok'] for r in part)}, wrong call {len(wrong)}, "
              f"no call {sum(r['call'] is None for r in part)} of {len(part)}; median probability of a right call "
              f"{np.median([r['p'] for r in part if r['ok']] or [0]):.3f}, of a wrong one {np.median([r['p'] for r in wrong] or [0]):.3f}")
    bins = [(0, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 1.01)]
    calib = []
    for lo, hi in bins:
        b = [r for r in rows if lo <= r['p'] < hi and r['call'] is not None]
        calib.append({'lo': lo, 'hi': min(hi, 1), 'n': len(b), 'right': sum(r['ok'] for r in b)})
        print(f'probability {lo:.2f} to {min(hi, 1):.2f}: {len(b):>3} calls, {sum(r["ok"] for r in b):>3} right')
    n_calls = sum(r['call'] is not None for r in rows)
    table = {}
    for name, t in [('never', 0), ('always', 0), ('worded oddly', 0)] + [('unsure', t) for t in (0.5, 0.9, 0.99)]:
        asked, through = score(rows, rule(name, t))
        key = name if name != 'unsure' else f'unsure < {t}'
        table[key] = {'asked': asked, 'through': through}
        print(f'{key:14} asks {asked:>3} of {n_calls} calls, lets {through:>3} wrong calls through')
    curve = [{'t': t, 'asked': score(rows, rule('unsure', t))[0], 'through': score(rows, rule('unsure', t))[1]}
             for t in sorted(set([0] + [r['p'] for r in rows if r['call'] is not None] + [1.01]))]
    with open(OUT, 'w') as f:
        json.dump({'calls': n_calls, 'rows': rows, 'calib': calib, 'table': table, 'curve': curve}, f)
    print('wrote', os.path.normpath(OUT))
