"""
ch13_constrain.py — make a broken format impossible, not just unlikely.   (Chapter 13)

    python code/ch13_constrain.py
    python code/ch13_constrain.py --format cast --temperature 1.0 --show 5

A program that reads a model's output needs it in a fixed shape. Asking nicely
makes the right shape likely. This script makes the wrong shape impossible:
before each draw, every character the format does not allow gets a score of
minus infinity. The model is untouched. Like temperature and top-k in chapter 8,
this is one more small function between the scores and the dice.
"""
import argparse, string
import numpy as np
from ch08_sample import Runner, sample_next

UP, LOW = string.ascii_uppercase, string.ascii_lowercase
SENTENCE = [(UP, 1, 1), (LOW + " ,'", 8, 40), ('.!?', 1, 1), ('\n', 1, 1)]

# A format is a list of parts, read left to right. A part is either
#   (characters, min, max)   between min and max characters from this set, or
#   [word, word, ...]        exactly one of these words.
FORMATS = {
    # SPEAKER:  then one sentence on one line
    'line': [(UP, 2, 12), (':', 1, 1), ('\n', 1, 1)] + SENTENCE,
    # the same, but the speaker must come from a fixed list (HAMLET is not in the training text)
    'cast': [['ROMEO', 'JULIET', 'MERCUTIO', 'HAMLET'], (':', 1, 1), ('\n', 1, 1)] + SENTENCE,
    # a shape the model almost never writes: a whole line in capitals
    'shout': [(UP, 2, 12), (':', 1, 1), ('\n', 1, 1), (UP + ' ', 12, 40), ('!', 1, 1), ('\n', 1, 1)],
}
START = (0, '')          # a state is (which part we are in, what has been written of it)


# [snippet:moves]
def moves(fmt, state):
    """Which characters may come next, and which state does each one lead to?
    An empty answer means the format is complete."""
    i, s = state
    out = {}
    while i < len(fmt):
        part = fmt[i]
        if isinstance(part, list):                        # one word from a list
            for w in part:
                if w.startswith(s) and len(w) > len(s):
                    out.setdefault(w[len(s)], (i, s + w[len(s)]))
            may_end = s in part
        else:                                             # min..max characters from a set
            chars, lo, hi = part
            if len(s) < hi:
                for c in chars:
                    out.setdefault(c, (i, s + c))
            may_end = len(s) >= lo
        if not may_end:
            break                                         # this part is not finished yet
        i, s = i + 1, ''                                  # it may be over: the next part may begin
    return out
# [/snippet]


# [snippet:constrain]
def generate_constrained(run, prompt, fmt, **sampling):
    cache = run.new_cache()
    for c in prompt:
        logits = run.step(run.stoi[c], cache)
    state, text, removed = START, '', []
    while True:
        legal = moves(fmt, state)                         # ask the format, not the model
        if not legal:
            return text, removed                          # format complete: stop
        mask = np.full(run.vocab, -np.inf)
        mask[[run.stoi[c] for c in legal]] = 0            # 0 for allowed, -inf for everything else
        p = np.exp(logits - logits.max()); p /= p.sum()
        removed.append(1 - p[mask == 0].sum())            # how much of the model's belief the mask deleted
        nxt = run.chars[sample_next(logits + mask, **sampling)]   # the dice of chapter 8, on masked scores
        state, text = legal[nxt], text + nxt
        logits = run.step(run.stoi[nxt], cache)
# [/snippet]


# [snippet:follows]
def follows(fmt, text):
    """Does `text` start with something in the format? Returns (characters read, completed)."""
    state = START
    for n, c in enumerate(text):
        legal = moves(fmt, state)
        if not legal:
            return n, True                                # completed; the rest is ignored
        if c not in legal:
            return n, False                               # broke the format at character n
        state = legal[c]
    return len(text), not moves(fmt, state)
# [/snippet]


def longest(fmt):
    return sum(max(map(len, p)) if isinstance(p, list) else p[2] for p in fmt)


def check(fmt, run, prompt):
    """One state is enough only if a part that may end shares no character with what follows."""
    assert len(prompt) + longest(fmt) <= run.block, 'prompt + format must fit the context window'
    for part in fmt:
        for c in (''.join(part) if isinstance(part, list) else part[0]):
            assert c in run.stoi, f'{c!r} is not in the model vocabulary'
    for a, b in zip(fmt, fmt[1:]):
        first = {w[0] for w in b} if isinstance(b, list) else set(b[0])
        if not isinstance(a, list) and a[1] < a[2]:
            assert not (set(a[0]) & first), 'adjacent parts share a character'


def word_probs(run, prompt, words):
    """For each word: the model's own probability of writing it, and its probability under the mask."""
    own, masked = [], []
    for w in words:
        cache = run.new_cache()
        for c in prompt:
            logits = run.step(run.stoi[c], cache)
        po, pm, state = 1.0, 1.0, START
        for c in w:
            legal = moves([words], state)
            p = np.exp(logits - logits.max()); p /= p.sum()
            allowed = sum(p[run.stoi[x]] for x in legal)
            po *= p[run.stoi[c]]; pm *= p[run.stoi[c]] / allowed
            state = legal[c]
            logits = run.step(run.stoi[c], cache)
        own.append(po); masked.append(pm)
    return own, masked


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--format', default='line', choices=list(FORMATS))
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--prompt', default='\n\n', help='text before the constrained part (default: a blank line)')
    ap.add_argument('--samples', type=int, default=200)
    ap.add_argument('--show', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    run, prompt = Runner(), args.prompt.replace('\\n', '\n')
    rng = np.random.default_rng(args.seed)
    sampling = dict(temperature=args.temperature, rng=rng)

    for name, fmt in FORMATS.items():
        check(fmt, run, prompt)

    # 1. look at a few
    fmt = FORMATS[args.format]
    print(f'format "{args.format}", temperature {args.temperature}\n')
    for _ in range(args.show):
        free = run.generate(prompt, longest(fmt), **sampling)[len(prompt):]
        n, ok = follows(fmt, free)
        print('without the mask:', repr(free[:n + 1]), 'ok' if ok else f'<- breaks the format at character {n}')
        text, removed = generate_constrained(run, prompt, fmt, **sampling)
        print('with the mask:   ', repr(text), '\n')

    # 2. count: how often is the shape right?
    print(f'{args.samples} samples per row at temperature {args.temperature}:')
    print(f'{"format":8} {"valid without":>14} {"valid with":>11} {"belief removed per character":>30}')
    for name, fmt in FORMATS.items():
        free_ok = sum(follows(fmt, run.generate(prompt, longest(fmt), **sampling)[len(prompt):])[1]
                      for _ in range(args.samples))
        with_ok, removed = 0, []
        for _ in range(args.samples):
            text, r = generate_constrained(run, prompt, fmt, **sampling)
            with_ok += follows(fmt, text) == (len(text), True)
            removed += r
        assert with_ok == args.samples                    # by construction, not by luck
        print(f'{name:8} {100 * free_ok / args.samples:13.1f}% {100 * with_ok / args.samples:10.0f}% '
              f'{100 * np.mean(removed):29.1f}%')

    # 3. a list of allowed words: allowed is not the same as likely
    words = FORMATS['cast'][0]
    own, masked = word_probs(run, prompt, words)
    print('\nspeaker    model alone   under the mask')
    for w, a, b in zip(words, own, masked):
        print(f'{w:10} {100 * a:10.3f}% {100 * b:14.1f}%')
    print(f'{"(total)":10} {100 * sum(own):10.3f}% {100 * sum(masked):14.1f}%')
