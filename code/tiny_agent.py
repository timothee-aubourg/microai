"""
tiny_agent.py — the packed model of chapter 23, run with nothing but Python.   (Chapter 23)

    python code/tiny_agent.py "what is 23 times 45?"

No numpy, no other file of this course: only runs/tiny.bin and the standard
library. Every weight in that file is -1, 0 or +1 times one number per matrix,
so a matrix product needs no multiplication per weight: add the inputs marked
+1, subtract the ones marked -1, and scale the result once.
"""
import math, os, struct, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, '..', 'runs', 'tiny.bin')


# [snippet:load]
def load(path=PATH):
    """Read the file: a header, then per matrix one number and its weights, five to a byte."""
    blob = open(path, 'rb').read()
    assert blob[:4] == b'uAI3', 'not a packed model of chapter 23'
    V, block, L, H, d, mlp = blob[4:10]
    m = {'chars': blob[10:10 + V].decode('ascii'), 'block': block, 'L': L, 'H': H, 'd': d,
         'where': struct.unpack('<f', blob[10 + V:14 + V])[0], 'bytes': len(blob)}
    at = 14 + V

    def matrix(rows, cols):
        nonlocal at
        scale = struct.unpack('<f', blob[at:at + 4])[0]
        t = []
        for byte in blob[at + 4:at + 4 + -(-rows * cols // 5)]:
            for _ in range(5):                       # one byte holds five weights: digits in base 3
                t.append(byte % 3 - 1)
                byte //= 3
        at += 4 + -(-rows * cols // 5)
        # for each output: which inputs to add and which to subtract
        add = [[i for i in range(rows) if t[i * cols + j] == 1] for j in range(cols)]
        sub = [[i for i in range(rows) if t[i * cols + j] == -1] for j in range(cols)]
        return scale, add, sub, [t[i * cols:(i + 1) * cols] for i in range(rows)]

    m['wte'] = matrix(V, d)
    # the output layer is the embedding table read the other way: per token, which numbers to add or subtract
    rows = m['wte'][3]
    m['out'] = (m['wte'][0], [[j for j in range(d) if r[j] == 1] for r in rows], [[j for j in range(d) if r[j] == -1] for r in rows])
    m['layers'] = [{name: matrix(*shape) for name, shape in
                    (('wq', (d, d)), ('wk', (d, d)), ('wv', (d, d)), ('wo', (d, d)), ('w1', (d, mlp * d)), ('w2', (mlp * d, d)))}
                   for _ in range(L)]
    assert at == len(blob)
    m['stoi'] = {c: i for i, c in enumerate(m['chars'])}
    return m
# [/snippet]


# [snippet:engine]
def times(x, w):
    """x @ W for a three-valued W: sums and differences, then one multiplication per output."""
    scale, add, sub = w[0], w[1], w[2]
    pick = x.__getitem__
    return [scale * (sum(map(pick, a)) - sum(map(pick, s))) for a, s in zip(add, sub)]

def norm(x):
    mean = sum(x) / len(x)
    x = [v - mean for v in x]
    size = math.sqrt(sum(v * v for v in x) / len(x) + 1e-5)
    return [v / size for v in x]

def step(m, token, cache):
    """One token through the model. `cache` keeps the keys and values of the tokens before it."""
    d, H = m['d'], m['H']
    hd, pos = d // H, len(cache[0][0]) if cache else 0
    if not cache:
        cache.extend(([], []) for _ in m['layers'])
    row, scale = m['wte'][3][token], m['wte'][0]
    x = [scale * row[j] + m['where'] * (math.sin if j < d // 2 else math.cos)(pos / 50.0 ** (2 * (j % (d // 2)) / d)) for j in range(d)]
    for layer, (keys, values) in zip(m['layers'], cache):
        a = norm(x)
        q = times(a, layer['wq'])
        keys.append(times(a, layer['wk']))
        values.append(times(a, layer['wv']))
        mixed = []
        for h in range(H):                                  # each head: score every earlier token, then average
            lo, hi = h * hd, (h + 1) * hd
            scores = [sum(q[j] * k[j] for j in range(lo, hi)) / math.sqrt(hd) for k in keys]
            top = max(scores)
            weights = [math.exp(s - top) for s in scores]
            total = sum(weights)
            mixed += [sum(w * v[j] for w, v in zip(weights, values)) / total for j in range(lo, hi)]
        x = [u + v for u, v in zip(x, times(mixed, layer['wo']))]
        hidden = times(norm(x), layer['w1'])
        hidden = [0.5 * v * (1 + math.tanh(0.7978845608 * (v + 0.044715 * v * v * v))) for v in hidden]
        x = [u + v for u, v in zip(x, times(hidden, layer['w2']))]
    return times(norm(x), m['out'])                         # one score per character
# [/snippet]


# ---- the harness: the loop and the format machine of chapters 13 and 19, copied so this file stands alone ----
TOOLS = {'add': lambda a, b: a + b, 'sub': lambda a, b: a - b, 'mul': lambda a, b: a * b}
DIGITS = '0123456789'
CALL = [['add', 'sub', 'mul'], (' ', 1, 1), (DIGITS, 1, 2), (' ', 1, 1), (DIGITS, 1, 2), ('\n', 1, 1)]

def moves(fmt, state):
    i, s = state
    out = {}
    while i < len(fmt):
        part = fmt[i]
        if isinstance(part, list):
            for w in part:
                if w.startswith(s) and len(w) > len(s):
                    out.setdefault(w[len(s)], (i, s + w[len(s)]))
            may_end = s in part
        else:
            chars, lo, hi = part
            if len(s) < hi:
                for c in chars:
                    out.setdefault(c, (i, s + c))
            may_end = len(s) >= lo
        if not may_end:
            break
        i, s = i + 1, ''
    return out

def write_line(m, cache, logits, fmt=None, limit=70):
    line, state = '', (0, '')
    while not line.endswith('\n') and len(line) < limit and len(cache[0][0]) < m['block']:
        legal = moves(fmt, state) if fmt else None
        if fmt and not legal:
            break
        options = [m['stoi'][c] for c in legal] if fmt else range(len(logits))
        best = max(options, key=logits.__getitem__)          # greedy: the likeliest character that is allowed
        c = m['chars'][best]
        if fmt:
            state = legal[c]
        line += c
        logits = step(m, best, cache)
    return line, logits

def answer(m, question):
    """Returns (the answer, the tool call or None, how many tokens went through the model)."""
    cache, logits = [], None
    for c in '\nU ' + question + '\n':
        logits = step(m, m['stoi'][c], cache)
    role, logits = write_line(m, cache, logits, fmt=[['C ', 'A ']])
    call = None
    if role == 'C ':
        call, logits = write_line(m, cache, logits, fmt=CALL)
        name, a, b = call.split()
        for c in f'R {TOOLS[name](int(a), int(b))}\nA ':     # the harness runs the tool and writes the result
            logits = step(m, m['stoi'][c], cache)
    text, _ = write_line(m, cache, logits)
    return text.strip(), call and call.strip(), len(cache[0][0])


if __name__ == '__main__':
    if not os.path.exists(PATH):
        sys.exit('No packed model yet. Run:  python code/ch23_tiny.py --train')
    m = load()
    for question in ([' '.join(sys.argv[1:])] if len(sys.argv) > 1 else ['what is 23 times 45?', 'hello', 'what is 7 minus 60?']):
        if any(c not in m['stoi'] for c in question):
            sys.exit('This model knows only lowercase letters, digits, spaces and ? . , -')
        t0 = time.perf_counter()
        text, call, tokens = answer(m, question)
        seconds = time.perf_counter() - t0
        print(f'U {question}')
        if call:
            name, a, b = call.split()
            print(f'C {call}\nR {TOOLS[name](int(a), int(b))}')
        print(f'A {text}')
        print(f'  ({m["bytes"]:,} bytes of model, {tokens} tokens in {1000 * seconds:.0f} ms, {tokens / seconds:,.0f} tokens per second, no numpy)\n')
