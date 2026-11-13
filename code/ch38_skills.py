"""
ch38_skills.py — a catalogue too large for the context: load what is needed.   (Chapter 38)

    python code/ch38_skills.py        # about 1 minute; needs  python code/ch37_tools.py --train

The agent of chapter 37 reads its tools from a list in its context. Its window
has room for four entries. Here the catalogue has six. Three ways to handle it,
each asked the same 300 questions:

    everything     put the whole catalogue in the context; what does not fit is cut off
    harness picks  the harness reads the question and lists only the tools whose description it finds there
    agent asks     start with an empty list; when the agent says which tool it lacks, load that one and ask again

For each: how many answers are right, and how many characters the agent had to read.
"""
import json, os, sys, time
import numpy as np
from gpt import GPT
import ch37_tools as t37
from mcp_server import OPS

OUT = os.path.join(t37.RUNS, 'skills.json')
ROOM = 4                                                  # entries that fit in the list the agent was trained on
CATALOGUE = [(op, t37.NEW[i]) for op, i in zip(OPS, (3, 40, 77, 101, 58, 12))]


def local(name, a, b):
    """The tools run in this process here; chapter 37 showed the same calls going to a server."""
    return OPS[dict((n, op) for op, n in CATALOGUE)[name]](a, b)


# [snippet:ways]
def everything(model, question):
    tools = CATALOGUE[:ROOM]                              # the rest does not fit, and is silently gone
    _, reply, read = t37.answer(model, tools, local, question)
    return reply, read, 1

def harness_picks(model, question):
    tools = [(op, name) for op, name in CATALOGUE if op in question.rstrip('?').split()]   # description found in the question
    _, reply, read = t37.answer(model, tools, local, question)
    return reply, read, 1

def agent_asks(model, question):
    _, reply, cost = t37.answer(model, [], local, question)    # first pass: no tools listed at all
    passes = 1
    if reply.startswith('i have no tool for '):           # the agent names what it lacks...
        wanted = reply[len('i have no tool for '):].rstrip('.')
        tools = [(op, name) for op, name in CATALOGUE if op == wanted]
        if tools:                                         # ...the harness loads exactly that, and asks again
            _, reply, read = t37.answer(model, tools, local, question)
            cost, passes = cost + read, 2
    return reply, cost, passes
# [/snippet]


if __name__ == '__main__':
    if not os.path.exists(t37.PATH):
        sys.exit('No trained agent yet. Run:  python code/ch37_tools.py --train')
    t0 = time.time()
    model, _ = GPT.load(t37.PATH)
    asked = t37.questions(300, list(OPS), seed=3)
    out = {'catalogue': CATALOGUE, 'room': ROOM, 'ways': {}, 'per_entry': round(sum(len(f' {op}={name}') for op, name in CATALOGUE) / len(CATALOGUE), 1)}
    print(f'a catalogue of {len(CATALOGUE)} tools, a list with room for {ROOM}, 300 questions\n')
    print(f'{"way":16}{"right":>12}{"said it had no tool":>22}{"characters read per question":>30}{"passes":>9}')
    for name, way in [('everything', everything), ('harness picks', harness_picks), ('agent asks', agent_asks)]:
        right = none = cost = passes = 0
        by_op = {op: [0, 0] for op in OPS}
        for a, op, b in asked:
            reply, c, p = way(model, f'what is {a} {op} {b}?')
            ok = reply == f'{a} {op} {b} is {OPS[op](a, b)}.'
            right += ok; none += reply.startswith('i have no tool'); cost += c; passes += p
            by_op[op][0] += ok; by_op[op][1] += 1
        out['ways'][name] = {'right': right, 'none': none, 'chars': round(cost / len(asked), 1), 'passes': round(passes / len(asked), 2), 'by_op': by_op}
        print(f'{name:16}{right:>6} / 300{none:>16} / 300{out["ways"][name]["chars"]:>30}{out["ways"][name]["passes"]:>9}', flush=True)
    print('\n"everything", by operation:', {op: f'{r}/{n}' for op, (r, n) in out['ways']['everything']['by_op'].items()})
    out['seconds'] = round(time.time() - t0)
    with open(OUT, 'w') as f:
        json.dump(out, f)
    print(f"\n{out['seconds']} s")
