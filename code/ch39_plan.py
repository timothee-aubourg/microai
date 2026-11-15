"""
ch39_plan.py — an agent for jobs of several steps: one step at a time, or a plan first.   (Chapter 39)

    python code/ch39_plan.py --train      # trains runs/work.npz, about 25 minutes on a laptop CPU
    python code/ch39_plan.py              # the measurements of chapter 39, about 3 minutes
    python code/ch39_plan.py "what is 23 times 45 minus 17?"

The agent of chapter 19 answers one sum with one tool call. Here a question is a chain,
worked from left to right like a pocket calculator: "what is 23 times 45 minus 17?" means
(23 x 45) - 17. Each link needs a tool call, and the result of one link is an input of the next.

One model, trained here from random numbers, learns three kinds of conversation:

    step by step   it writes one call, reads the result, writes the next call (lines C and R)
    plan first     it writes every call at once, on one line P, with $1 for "the result of step 1";
                   the harness runs the plan and hands back only the last result
    notes          lines M before the question hold facts from earlier sessions ("M n is 42");
                   the question may then say "n" instead of a number            (chapter 42)

In the step-by-step conversations a tool sometimes answers "R error busy", and the model
learns to send the same call again                                              (chapter 40)

The harness chooses the kind by writing the first letter of the model's line itself (C or P).
"""
import argparse, json, math, os, sys, time
import numpy as np
from gpt import GPT
from ch08_sample import Runner, sample_next
from ch13_constrain import moves, START
import ch19_system as one

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
PATH = os.path.join(RUNS, 'work.npz')
OUT = os.path.join(RUNS, 'plan.json')
TOOLS, WORDS = one.TOOLS, one.WORDS
NAMES = 'bcdfghjkmnpqrstvwxz'                        # names a note can give a number: single letters
BLOCK = 160
CHARS = sorted(set('\n ?.-$,0123456789UCRAPM' + 'abcdefghijklmnopqrstuvwxyz'))
STOI = {c: i for i, c in enumerate(CHARS)}
BUSY = 'error busy'


# ---- the data ----------------------------------------------------------------------------
# [snippet:chain]
def chain(rng, links, test=False):
    """Numbers and operations for one question, and every value along the way, left to right."""
    while True:
        nums = [int(x) for x in rng.integers(100, size=links + 1)]
        if one.held_out(nums[0], nums[1]) != test:   # the first pair of a test question was never trained on
            continue
        ops = [list(WORDS)[k] for k in rng.integers(3, size=links)]
        values, v = [], nums[0]
        for op, n in zip(ops, nums[1:]):
            v = TOOLS[WORDS[op]](v, n)
            values.append(v)
        if all(abs(x) < 10000 for x in values):      # keep every result to four digits
            return nums, ops, values

def question(nums, ops, names=None):
    """The words of the question. With `names`, some numbers are replaced by the name of a note."""
    show = [names.get(i, str(n)) for i, n in enumerate(nums)] if names else [str(n) for n in nums]
    return 'what is ' + show[0] + ''.join(f' {op} {x}' for op, x in zip(ops, show[1:])) + '?'

def step_by_step(nums, ops, values, rng=None, busy=0.0):
    """The conversation, one call at a time: the model sees each result before the next call."""
    lines, left = [], nums[0]
    for op, n, v in zip(ops, nums[1:], values):
        call = f'C {WORDS[op]} {left} {n}'
        while rng is not None and rng.random() < busy:          # the tool is busy: the same call again
            lines += [call, f'R {BUSY}']
        lines += [call, f'R {v}']
        left = v
    return lines + [f'A {values[-1]}.']

def plan_first(nums, ops, values):
    """The conversation with a plan: every call at once, then only the final result."""
    steps = [f'{WORDS[ops[0]]} {nums[0]} {nums[1]}'] + [f'{WORDS[op]} ${i} {n}' for i, (op, n) in enumerate(zip(ops[1:], nums[2:]), 1)]
    return ['P ' + ', '.join(steps), f'R {values[-1]}', f'A {values[-1]}.']
# [/snippet]

def notes_for(rng, nums):
    """Facts from earlier sessions: one to four notes, of which the question uses one or two."""
    used = sorted(rng.choice(len(nums), size=int(rng.integers(1, 3)), replace=False).tolist())
    letters = rng.choice(list(NAMES), size=int(rng.integers(len(used), 5)), replace=False).tolist()
    names = {i: letters[k] for k, i in enumerate(used)}
    notes = [f'M {names[i]} is {nums[i]}' for i in used]
    for x in letters[len(used):]:
        notes.append(f'M {x} is {int(rng.integers(100))}')               # notes the question does not need
    order = rng.permutation(len(notes))
    return [notes[k] for k in order], names

def conversation(rng, test=False):
    links = int(rng.integers(1, 5))
    nums, ops, values = chain(rng, links, test)
    notes, names = notes_for(rng, nums) if rng.random() < 0.3 else ([], None)
    body = plan_first(nums, ops, values) if rng.random() < 0.4 else step_by_step(nums, ops, values, rng, busy=0.12)
    text = '\n'.join(notes + ['U ' + question(nums, ops, names)] + body) + '\n\n'
    return text if len(text) < BLOCK - 2 else conversation(rng, test)


# ---- training: the loop of chapter 19, a little larger ------------------------------------
def train(steps, batch=32, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    text = ''.join(conversation(rng) for _ in range(40000))
    data = np.array([STOI[c] for c in text])
    model = GPT(vocab_size=len(CHARS), block_size=BLOCK, n_layer=3, n_head=4, d_model=64)
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
        if step % 2000 == 1999:
            model.save(PATH, CHARS)
    model.save(PATH, CHARS)
    print('wrote', os.path.normpath(PATH))


# ---- the harness ---------------------------------------------------------------------------
D = '0123456789'
CALL = [['add', 'sub', 'mul'], (' ', 1, 1), ('-', 0, 1), (D, 1, 4), (' ', 1, 1), ('-', 0, 1), (D, 1, 4), ('\n', 1, 1)]
STEP = [['add', 'sub', 'mul'], (' ', 1, 1), ('$-', 0, 1), (D, 1, 4), (' ', 1, 1), (D, 1, 2), [', ', '\n']]   # ends with what comes next

def live():
    run = Runner(PATH)
    return run

def write_line(run, cache, logits, fmt=None, limit=80):
    """Greedy, one character at a time, until the end of the line; with `fmt`, only what the format allows."""
    return one.write_line(run, cache, logits, fmt=fmt, limit=limit, temperature=0)

def feed(run, cache, text):
    logits = None
    for c in text:
        logits = run.step(run.stoi[c], cache)
    return logits

def number(s):
    s = s.strip()
    return int(s) if s.lstrip('-').isdigit() else None

def tool(name, a, b, busy=0.0, rng=None):
    """The calculator. With `busy`, it sometimes answers 'error busy' instead (chapter 40)."""
    if rng is not None and rng.random() < busy:
        return BUSY
    return str(TOOLS[name](a, b))

# [snippet:react]
def run_steps(run, prompt, max_calls=8, on_result=None):
    """Step by step: the model writes a call, the harness runs it and writes the result, and again.
    Returns the final number (or None), the calls made, and how many characters the model read."""
    cache = run.new_cache()
    logits, calls = feed(run, cache, prompt), []
    while cache['pos'] < run.block - 24 and len(calls) < max_calls:
        role, logits = write_line(run, cache, logits, fmt=[['C ', 'A ']])
        if role == 'A ':
            text, _ = write_line(run, cache, logits)
            return number(text.rstrip('.\n')), calls, cache['pos']
        call, logits = write_line(run, cache, logits, fmt=CALL)
        name, a, b = call.split()
        result = on_result(name, int(a), int(b)) if on_result else tool(name, int(a), int(b))
        calls.append((call.strip(), result))
        logits = feed(run, cache, f'R {result}\n')                     # the result is now in the context
    return None, calls, cache['pos']
# [/snippet]

# [snippet:plan]
def run_plan(run, prompt, on_result=None):
    """Plan first: the model writes every call on one line, without seeing any result.
    The harness then runs the steps in order, putting each result where the plan says $1, $2 ..."""
    cache = run.new_cache()
    logits = feed(run, cache, prompt + 'P ')                           # the harness asks for a plan
    plan, results = [], []
    while True:
        step, logits = write_line(run, cache, logits, fmt=STEP)
        plan.append(step.rstrip(', \n'))
        if not step.endswith(', ') or len(plan) > 8:            # the line is over: the plan is complete
            break
    for step in plan:
        name, a, b = step.split()
        a = results[int(a[1:]) - 1] if a.startswith('$') and 0 < int(a[1:] or 0) <= len(results) else number(a)
        if a is None:
            return None, plan, cache['pos']                            # the plan points at a step that does not exist
        r = on_result(name, a, int(b)) if on_result else tool(name, a, int(b))
        results.append(number(r))
        if results[-1] is None:
            return None, plan, cache['pos']
    return results[-1], plan, cache['pos']                              # the model needs no second pass
# [/snippet]


def prompt_for(nums, ops, notes=(), names=None):
    return '\n' + ''.join(n + '\n' for n in notes) + 'U ' + question(nums, ops, names) + '\n'


def measure(run, n=200):
    """Right answers, characters read and model passes, by number of links, for both ways."""
    rows = []
    for links in range(1, 5):
        rng = np.random.default_rng(100 + links)
        row = {'links': links, 'steps': {'right': 0, 'read': 0, 'passes': 0}, 'plan': {'right': 0, 'read': 0, 'passes': 0}}
        for _ in range(n):
            nums, ops, values = chain(rng, links, test=True)
            p = prompt_for(nums, ops)
            got, calls, read = run_steps(run, p)
            row['steps']['right'] += got == values[-1]
            row['steps']['read'] += read
            row['steps']['passes'] += len(calls) + 1                   # one pass per call, one for the answer
            got, plan, read = run_plan(run, p)
            row['plan']['right'] += got == values[-1]
            row['plan']['read'] += read
            row['plan']['passes'] += 1
        for k in ('steps', 'plan'):
            row[k]['read'] = round(row[k]['read'] / n)
            row[k]['passes'] = round(row[k]['passes'] / n, 2)
        rows.append(row)
        print(f"{links} links   step by step {row['steps']['right']:>3}/{n}  reads {row['steps']['read']:>3}  passes {row['steps']['passes']:.1f}"
              f"   plan first {row['plan']['right']:>3}/{n}  reads {row['plan']['read']:>3}  passes {row['plan']['passes']:.1f}", flush=True)
    return rows


# [snippet:checkpoint]
def long_job(run, questions, crash, rng, checkpoint):
    """A job of many questions in a row. Before each one the process may crash (probability `crash`).
    Without a checkpoint the job starts again from the first question; with one, from where it stopped."""
    done, passes, restarts = [], 0, 0                                  # `done` is what a checkpoint would save
    while len(done) < len(questions):
        if rng.random() < crash:
            restarts += 1
            if not checkpoint:
                done = []                                              # everything since the start is lost
            continue
        nums, ops, values = questions[len(done)]
        got, calls, _ = run_steps(run, prompt_for(nums, ops))
        passes += len(calls) + 1
        done.append(got == values[-1])
    return sum(done), passes, restarts
# [/snippet]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('question', nargs='*')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--steps', type=int, default=8000)
    args = ap.parse_args()
    if args.train:
        train(args.steps)
    if not os.path.exists(PATH):
        sys.exit('No trained agent yet. Run:  python code/ch39_plan.py --train')
    run = live()
    if args.question:
        q = ' '.join(args.question)
        got, calls, read = run_steps(run, '\nU ' + q + '\n')
        for c, r in calls:
            print(f'C {c}\nR {r}')
        print(f'A {got}   ({read} characters read)')
        got, plan, read = run_plan(run, '\nU ' + q + '\n')
        print(f'\nP {", ".join(plan)}\nA {got}   ({read} characters read)')
        sys.exit()
    t0 = time.time()
    out = {'rows': measure(run)}
    rng = np.random.default_rng(7)
    qs = [chain(rng, int(rng.integers(1, 4)), test=True) for _ in range(20)]
    jobs = {}
    for crash in (0.02, 0.05, 0.1):
        for cp in (False, True):
            r = np.random.default_rng(int(crash * 1000))
            right, passes, restarts = zip(*[long_job(run, qs, crash, r, cp) for _ in range(10)])
            jobs[f'{crash}-{cp}'] = {'crash': crash, 'checkpoint': cp, 'passes': round(float(np.mean(passes))), 'restarts': round(float(np.mean(restarts)), 1), 'right': round(float(np.mean(right)), 1)}
            print(f'long job of 20 questions, crash {crash:.2f} per question, checkpoint {cp!s:5}: '
                  f'{jobs[f"{crash}-{cp}"]["passes"]} model passes, {jobs[f"{crash}-{cp}"]["restarts"]} crashes', flush=True)
    base = sum(len(run_steps(run, prompt_for(*q[:2]))[1]) + 1 for q in qs)
    out.update(jobs=list(jobs.values()), job_questions=len(qs), job_base=base)
    print(f'with no crash at all: {base} model passes')
    out['seconds'] = round(time.time() - t0)
    with open(OUT, 'w') as f:
        json.dump(out, f)
    print(f"{out['seconds']} s")
