"""
ch19_system.py — the whole system, built from nothing, in one sitting.  (Chapter 19)

    python code/ch19_system.py --train               # about 5 minutes on a laptop CPU
    python code/ch19_system.py "what is 23 times 45?"
    python code/ch19_system.py --eval

Every earlier chapter built one engine. This file bolts them together into a
complete agent that you train yourself, on your own CPU, from random numbers:

    data          conversations written by a small generator (below)
    tokenizer     one id per character                              (chapter 2)
    model         the GPT of gpt.py, a smaller copy                 (chapters 4-6)
    training      engine.py's backpropagation and Adam              (chapters 3, 7)
    inference     the KV-cache runner and the sampler               (chapter 8)
    format        the mask that keeps tool calls parseable          (chapter 13)
    harness       a loop that runs the tool the model asks for      (chapters 11, 15)
    tracing       one span per model line and per tool call         (chapter 14)
    evaluation    questions it never saw, graded, with an error bar (chapter 14)

The result is a very small and very limited assistant. It cannot do arithmetic
and does not need to: it has learned to ask a calculator and to report what
the calculator says. Nothing in it is hidden.
"""
import argparse, math, os, sys, time
import numpy as np
from gpt import GPT
from ch08_sample import Runner, sample_next
from ch13_constrain import moves, START
from ch14_trace import Tracer

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, '..', 'runs', 'agent.npz')

# ---- 1. the tools: ordinary functions -----------------------------------------------
TOOLS = {'add': lambda a, b: a + b, 'sub': lambda a, b: a - b, 'mul': lambda a, b: a * b}
WORDS = {'plus': 'add', 'minus': 'sub', 'times': 'mul'}
CHAT = [('hello', 'hello. ask me to add, subtract or multiply.'),
        ('who are you?', 'a very small model with a calculator.'),
        ('what can you do?', 'i can ask a calculator and tell you what it says.')]

# ---- 2. the data: what a conversation looks like, as plain text ---------------------
# Each line starts with a letter that says who is speaking:
#   U user    C the model calls a tool    R the tool's result    A the model's answer
# [snippet:data]
def held_out(a, b):
    return (7 * a + b) % 10 == 0            # one pair of numbers in ten is never shown in training

def conversation(rng, test=False):
    if not test and rng.random() < 0.1:     # small talk: answer directly, no tool
        q, a = CHAT[rng.integers(len(CHAT))]
        return f'U {q}\nA {a}\n\n'
    while True:
        a, b = int(rng.integers(100)), int(rng.integers(100))
        if held_out(a, b) == test:
            break
    word = list(WORDS)[rng.integers(3)]
    result = TOOLS[WORDS[word]](a, b)
    return (f'U what is {a} {word} {b}?\n'      # the user asks
            f'C {WORDS[word]} {a} {b}\n'         # the model writes a request
            f'R {result}\n'                      # the harness runs it and writes the result
            f'A {a} {word} {b} is {result}.\n\n')   # the model reports
# [/snippet]

CHARS = sorted(set('\n ?.-0123456789UCRA' + 'abcdefghijklmnopqrstuvwxyz' + ','))
STOI = {c: i for i, c in enumerate(CHARS)}


# ---- 3. training: the loop of chapter 7, on our own text ----------------------------
def train(steps, batch=32, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    text = ''.join(conversation(rng) for _ in range(20000))
    data = np.array([STOI[c] for c in text])
    model = GPT(vocab_size=len(CHARS), block_size=96, n_layer=2, n_head=4, d_model=64)
    T, params = model.block_size, model.parameters()
    m = [np.zeros_like(p.data) for p in params]
    v = [np.zeros_like(p.data) for p in params]
    print(f'{len(text):,} characters of conversation, vocabulary {len(CHARS)}, model {model.num_params():,} weights')
    t0 = time.time()
    for step in range(steps):
        starts = rng.integers(0, len(data) - T - 1, batch)
        x = np.stack([data[s:s + T] for s in starts])
        y = np.stack([data[s + 1:s + T + 1] for s in starts])
        _, loss = model.forward(x, y)
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
        if step % 200 == 0 or step == steps - 1:
            print(f'step {step:5d}  loss {float(loss.data):.3f}  {time.time() - t0:.0f}s', flush=True)
    model.save(PATH, CHARS)
    print('wrote', os.path.normpath(PATH))


# ---- 4. the harness: read a line from the model, act on it, continue ----------------
DIGITS = '0123456789'
CALL = [['add', 'sub', 'mul'], (' ', 1, 1), (DIGITS, 1, 2), (' ', 1, 1), (DIGITS, 1, 2), ('\n', 1, 1)]

# [snippet:harness]
def write_line(run, cache, logits, fmt=None, limit=70, **sampling):
    """Let the model write up to the end of a line. With `fmt`, only characters the format allows."""
    line, state = '', START
    while not line.endswith('\n') and len(line) < limit and cache['pos'] < run.block:
        if fmt:
            legal = moves(fmt, state)
            if not legal:
                break                                     # the format is complete
            mask = np.full(run.vocab, -np.inf)
            mask[[run.stoi[c] for c in legal]] = 0
            logits = logits + mask
        c = run.chars[sample_next(logits, **sampling)]
        if fmt:
            state = legal[c]
        line += c
        logits = run.step(run.stoi[c], cache)             # one token through the model, cache kept
    return line, logits

def feed(run, cache, text):
    for c in text:
        logits = run.step(run.stoi[c], cache)
    return logits

def answer(run, question, tracer=None, **sampling):
    tracer = tracer or Tracer()
    with tracer.span('agent', 'answer', input=question) as root:
        cache = run.new_cache()
        logits = feed(run, cache, '\nU ' + question + '\n')           # the prompt
        with tracer.span('model', 'choose', input='C or A?') as s:    # does it want a tool, or to answer?
            role, logits = write_line(run, cache, logits, fmt=[['C ', 'A ']], **sampling)
            s['output'] = role
        if role == 'C ':
            with tracer.span('model', 'write call') as s:
                call, logits = write_line(run, cache, logits, fmt=CALL, **sampling)   # always parseable
                s['output'] = call
            name, a, b = call.split()
            with tracer.span('tool', name, input=[int(a), int(b)]) as s:
                s['output'] = TOOLS[name](int(a), int(b))                              # the harness acts
            logits = feed(run, cache, f"R {s['output']}\nA ")                         # the result becomes context
        with tracer.span('model', 'write answer') as s:
            text, logits = write_line(run, cache, logits, **sampling)
            s['output'] = text
        root['output'] = text.strip()
    return root['output'], tracer.spans
# [/snippet]


# ---- 5. evaluation: questions it has never seen -------------------------------------
# [snippet:evaluate]
def evaluate(run, n=200, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        lines = conversation(rng, test=True).split('\n')
        question, want_call, want_answer = lines[0][2:], lines[1][2:], lines[3][2:]
        got, spans = answer(run, question, temperature=0)
        calls = [s['output'].strip() for s in spans if s['name'] == 'write call']
        rows.append({'q': question, 'call ok': calls == [want_call], 'answer ok': got == want_answer})
    p = sum(r['call ok'] and r['answer ok'] for r in rows) / n
    return rows, p, math.sqrt(p * (1 - p) / n)
# [/snippet]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('question', nargs='*')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--eval', action='store_true')
    args = ap.parse_args()
    if args.train:
        train(args.steps)
    if not os.path.exists(PATH):
        sys.exit('No trained agent yet. Run:  python code/ch19_system.py --train')
    run = Runner(PATH)
    if args.eval:
        rows, p, se = evaluate(run)
        calls = sum(r['call ok'] for r in rows)
        print(f'{len(rows)} questions with number pairs never seen in training:')
        print(f'  correct tool call    {calls} of {len(rows)}')
        print(f'  call and answer both {round(p * len(rows))} of {len(rows)}: {100 * p:.0f}% ± {100 * se:.1f}')
        for r in [r for r in rows if not (r['call ok'] and r['answer ok'])][:5]:
            print('  failed:', r['q'])
    for question in ([' '.join(args.question)] if args.question else
                     [] if (args.eval or args.train) else ['what is 23 times 45?', 'hello', 'what is 7 minus 60?']):
        if any(c not in STOI for c in question):
            sys.exit('This model knows only lowercase letters, digits, spaces and ? . , -')
        text, spans = answer(run, question, temperature=0)
        print(f'\nU {question}')
        for s in spans[1:]:
            print(f"  {s['kind']:5} {s['name']:13} {s['ms']:6.1f} ms  {str(s.get('output')).strip()}")
        print(f'A {text}')
