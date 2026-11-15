"""
ch41_workflow.py — a fixed path written in code, or an agent that decides its own steps?   (Chapter 41)

    python code/ch41_workflow.py     # about 3 minutes; needs  python code/ch39_plan.py --train

One mix of 300 requests, the kind a small calculator service might get:

    150  one sum, worded as usual          "what is 23 times 45?"
     90  a chain of two to four links      "what is 23 times 45 minus 17?"
     60  one sum with a slip in it         "what is 23 tims 45?"   (the slips of chapter 43)

Four ways to serve them:

    code only     a parser written by hand: if the request has the expected shape, the code
                  does the sums itself; if not, it says so. No model at all.
    plan, then code   one model pass writes the plan; code runs it (chapter 39). A workflow:
                  the path is fixed, and the model fills in one step of it.
    agent         the model decides each call after reading the last result (chapter 39).
    route         the parser first; only what it cannot read goes to the agent.

Measured: right answers by kind of request, model passes, and time per request on this machine.
"""
import json, os, re, sys, time
import numpy as np
import ch39_plan as w
from ch43_ask import SLIPS

OUT = os.path.join(w.RUNS, 'workflow.json')
SHAPE = re.compile(r'^what is (\d+)((?: (?:plus|minus|times) \d+)+)\?$')


def mix(rng):
    reqs = []
    for i in range(300):
        links = 1 if i < 150 or i >= 240 else int(rng.integers(2, 5))
        nums, ops, values = w.chain(rng, links, test=True)
        q = w.question(nums, ops)
        kind = 'one sum' if i < 150 else 'chain' if i < 240 else 'slip'
        if kind == 'slip':
            q = q.replace(f' {ops[0]} ', f' {SLIPS[ops[0]][rng.integers(len(SLIPS[ops[0]]))]} ')
        reqs.append({'q': q, 'kind': kind, 'want': values[-1]})
    return reqs


# [snippet:code]
def code_only(q):
    """A hand-written parser: the expected shape, or nothing. Returns (answer or None, model passes)."""
    m = SHAPE.match(q)
    if not m:
        return None, 0                                     # "sorry, i only understand sums"
    v, rest = int(m.group(1)), m.group(2).split()
    for op, n in zip(rest[::2], rest[1::2]):
        v = w.TOOLS[w.WORDS[op]](v, int(n))                # the code does the work itself
    return v, 0

def agent(run, q):
    got, calls, _ = w.run_steps(run, '\nU ' + q + '\n')
    return got, len(calls) + 1

def plan_then_code(run, q):
    got, _, _ = w.run_plan(run, '\nU ' + q + '\n')
    return got, 1

def route(run, q):
    """The parser first; only what it cannot read goes to the agent."""
    got, passes = code_only(q)
    return (got, passes) if SHAPE.match(q) else agent(run, q)
# [/snippet]


if __name__ == '__main__':
    if not os.path.exists(w.PATH):
        sys.exit('No trained agent yet. Run:  python code/ch39_plan.py --train')
    run = w.live()
    reqs = mix(np.random.default_rng(41))
    ways = {'code only': lambda q: code_only(q), 'plan, then code': lambda q: plan_then_code(run, q),
            'agent': lambda q: agent(run, q), 'route': lambda q: route(run, q)}
    out = {}
    for name, fn in ways.items():
        res = {'one sum': 0, 'chain': 0, 'slip': 0, 'passes': 0, 'ms': 0.0}
        for r in reqs:
            t = time.perf_counter()
            got, passes = fn(r['q'])
            res['ms'] += 1000 * (time.perf_counter() - t)
            res[r['kind']] += got == r['want']
            res['passes'] += passes
        res['ms'] = round(res['ms'] / len(reqs), 2)
        res['passes'] = round(res['passes'] / len(reqs), 2)
        res['right'] = res['one sum'] + res['chain'] + res['slip']
        out[name] = res
        print(f"{name:16} right {res['right']:>3}/300  (one sum {res['one sum']}/150, chain {res['chain']}/90, slip {res['slip']}/60)"
              f"   model passes {res['passes']:.2f}   {res['ms']:.2f} ms per request", flush=True)
    with open(OUT, 'w') as f:
        json.dump({'ways': out, 'examples': reqs[::25]}, f)
