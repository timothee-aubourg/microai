"""
ch37_tools.py — an agent that reads its tools from a list it is given.   (Chapter 37)

    python code/ch37_tools.py --train     # about 12 minutes: train the agent
    python code/ch37_tools.py             # about 2 minutes: connect servers and measure

The agent of chapter 19 knew three tools because they were in its training text.
This one is trained on conversations that begin with a list of tools, different
every time, under names that mean nothing:

    T plus=ka minus=wo            the tools on offer: operation=name
    U what is 23 plus 45?
    C plus=ka 23 45               it quotes the entry it uses, and the harness calls "ka"
    R 68
    A 23 plus 45 is 68.

and, when the list has no tool for the question:

    T minus=wo
    U what is 23 plus 45?
    A i have no tool for plus.

It is tested with names that never appeared in training, on servers started as
separate programs (code/mcp_server.py) and reached over the protocol.
"""
import argparse, hashlib, json, math, os, subprocess, sys, time
import numpy as np
from engine import Tensor
from gpt import GPT
from ch13_constrain import moves, START
import ch19_system as big
from mcp_server import OPS, VERSION

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
PATH = os.path.join(RUNS, 'tools.npz')
OUT = os.path.join(RUNS, 'tools.json')
CHARS = big.CHARS + ['=', 'T']
STOI = {c: i for i, c in enumerate(CHARS)}
WIDTH = 112
LETTERS = 'abcdefghijklmnopqrstuvwxyz'
NAMES = [a + b for a in LETTERS for b in LETTERS]
_order = np.random.default_rng(0).permutation(len(NAMES))
NEW = sorted(NAMES[i] for i in _order[:120])                   # names kept out of training
SEEN = sorted(NAMES[i] for i in _order[120:])


def header(tools):
    return ('T ' + ' '.join(f'{op}={name}' for op, name in tools)).rstrip() + '\n'

def conversation(rng, test=False):
    """One training conversation: a list of up to four tools, a question, and what to do."""
    k = int(rng.integers(0, 5))
    ops = [list(OPS)[i] for i in rng.permutation(len(OPS))]
    pool = NEW if test else SEEN
    listed = [(op, pool[i]) for op, i in zip(ops[:k], rng.choice(len(pool), k, replace=False))]
    op = ops[int(rng.integers(k))] if k and rng.random() < 0.75 else ops[int(rng.integers(k, len(ops)))] if k < len(ops) else ops[0]
    while True:
        a, b = int(rng.integers(100)), int(rng.integers(100))
        if big.held_out(a, b) == test and not (op == 'mod' and b == 0):
            break
    name = dict(listed).get(op)
    text = header(listed) + f'U what is {a} {op} {b}?\n'
    if name is None:
        return text + f'A i have no tool for {op}.\n', listed, (a, op, b)
    result = OPS[op](a, b)
    return text + f'C {op}={name} {a} {b}\nR {result}\nA {a} {op} {b} is {result}.\n', listed, (a, op, b)

def train(steps=6000, batch=32, lr=3e-3):
    rng = np.random.default_rng(0)
    model = GPT(len(CHARS), block_size=WIDTH, n_layer=2, n_head=4, d_model=64)
    rows = np.array([[STOI[c] for c in ('\n' + conversation(rng)[0]).ljust(WIDTH + 1, '\n')] for _ in range(40000)])
    params = model.parameters()
    m, v = [np.zeros_like(p.data) for p in params], [np.zeros_like(p.data) for p in params]
    t0 = time.time()
    for step in range(steps):
        pick = rows[rng.integers(0, len(rows), batch)]
        _, loss = model.forward(pick[:, :-1], pick[:, 1:])
        for p in params:
            p.grad = None
        loss.backward()
        rate = lr * min(1, (step + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        norm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in params))
        for i, p in enumerate(params):
            g = p.grad * min(1.0, 1.0 / (norm + 1e-6))
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.99 * v[i] + 0.01 * g * g
            p.data -= rate * (m[i] / (1 - 0.9 ** (step + 1))) / (np.sqrt(v[i] / (1 - 0.99 ** (step + 1))) + 1e-8)
        if step % 500 == 0 or step == steps - 1:
            print(f'  step {step:5d}  loss {float(loss.data):.3f}  {time.time() - t0:.0f}s', flush=True)
    model.save(PATH, CHARS)


# ---- the client side of the protocol ---------------------------------------------------
# [snippet:client]
class Server:
    """A tool server running as a separate program. We talk to it in JSON-RPC, one line per message."""
    def __init__(self, *args):
        self.proc = subprocess.Popen([sys.executable, os.path.join(HERE, 'mcp_server.py'), *args],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.sent = 0

    def rpc(self, method, params=None):
        self.sent += 1
        message = {'jsonrpc': '2.0', 'id': self.sent, 'method': method,
                   'params': dict(params or {}, _meta={'io.modelcontextprotocol/protocolVersion': VERSION})}
        self.proc.stdin.write(json.dumps(message) + '\n')
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())['result']

    def tools(self):
        """What the server offers, as (description, name) pairs."""
        return [(t['description'], t['name']) for t in self.rpc('tools/list')['tools']]

    def call(self, name, a, b):
        return self.rpc('tools/call', {'name': name, 'arguments': {'a': a, 'b': b}})['content'][0]['text']

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
# [/snippet]


# ---- the harness -----------------------------------------------------------------------
# [snippet:harness]
def answer(model, tools, call, question, mask=True):
    """`tools` is the list put in front of the model; `call(name, a, b)` reaches whoever owns the tool."""
    ids = [STOI[c] for c in '\n' + header(tools) + f'U {question}\n']
    def write(fmt=None, limit=40):
        line, state = '', START
        while not line.endswith('\n') and len(line) < limit and len(ids) < WIDTH:
            z = model.forward(np.array([ids]))[0].data[0, -1]
            if fmt:
                legal = moves(fmt, state)
                if not legal:
                    break
                allowed = np.full(len(CHARS), -np.inf)
                allowed[[STOI[c] for c in legal]] = 0
                z = z + allowed
            c = CHARS[int(z.argmax())]
            if fmt:
                state = legal[c]
            line += c
            ids.append(STOI[c])
        return line
    used = None
    if write([['C ', 'A ']]) == 'C ':
        entries = [f'{op}={name}' for op, name in tools]
        shape = [entries, (' ', 1, 1), (big.DIGITS, 1, 2), (' ', 1, 1), (big.DIGITS, 1, 2), ('\n', 1, 1)]   # built from the list
        parts = write(shape if mask else None, 20).split()
        if len(parts) != 3 or parts[0] not in entries or not (parts[1] + parts[2]).isdigit():
            return None, 'the call could not be run.', len(ids)
        used = parts[0].split('=')[1]
        for c in f'R {call(used, int(parts[1]), int(parts[2]))}\nA ':
            ids.append(STOI[c])
    reply = write().strip()
    return used, reply, len(ids)                                  # the tool used, the reply, characters in the context
# [/snippet]


# [snippet:pin]
def fingerprint(tools):
    """A short code that changes if any name or description in a tool list changes."""
    return hashlib.sha256(json.dumps(sorted(tools)).encode()).hexdigest()[:12]

def admit(lists, approved):
    """Keep only servers whose list is the one that was approved, and no second tool for a description already taken."""
    kept, taken = [], set()
    for server, tools in lists:
        if approved.get(server) != fingerprint(tools):
            continue                                              # unknown server, or its list has changed
        kept += [(op, name) for op, name in tools if op not in taken]
        taken |= {op for op, _ in tools}
    return kept
# [/snippet]


def questions(n, ops, seed=1):
    rng, out = np.random.default_rng(seed), []
    while len(out) < n:
        a, b, op = int(rng.integers(100)), int(rng.integers(100)), ops[int(rng.integers(len(ops)))]
        if big.held_out(a, b) and not (op == 'mod' and b == 0):
            out.append((a, op, b))
    return out

def measure(model, tools, route, asked, mask=True):
    """How many questions come out as they should: the right result, or an honest 'no tool'."""
    have = {op for op, _ in tools}
    out = {'right': 0, 'asked_with_tool': 0, 'declined': 0, 'asked_without': 0, 'none': 0, 'calls': {}}
    for a, op, b in asked:
        used, reply, _ = answer(model, tools, route, f'what is {a} {op} {b}?', mask)
        if used:
            out['calls'][used] = out['calls'].get(used, 0) + 1
        out['none'] += reply.startswith('i have no tool')
        if op in have:
            out['asked_with_tool'] += 1
            out['right'] += reply == f'{a} {op} {b} is {OPS[op](a, b)}.'
        else:
            out['asked_without'] += 1
            out['declined'] += reply == f'i have no tool for {op}.'
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    args = ap.parse_args()
    if args.train:
        train()
        sys.exit()
    if not os.path.exists(PATH):
        sys.exit('No trained agent yet. Run:  python code/ch37_tools.py --train')
    t0 = time.time()
    model, _ = GPT.load(PATH)
    a_srv, b_srv = Server('--tools', f'plus={NEW[3]},minus={NEW[40]}'), Server('--tools', f'times={NEW[77]},mod={NEW[101]}')
    bad = Server('--tools', f'plus={NEW[9]}', '--lie', '1')
    owner = {}
    def route(name, a, b):
        return owner[name].call(name, a, b)
    def connect(*servers):
        owner.clear()
        tools = []
        for s in servers:
            for op, name in s.tools():
                owner[name] = s
                tools.append((op, name))
        return tools
    asked = questions(300, ['plus', 'minus', 'times', 'mod'])
    out = {'weights': int(sum(t.data.size for t in model.p.values())), 'names': {'a': a_srv.tools(), 'b': b_srv.tools(), 'bad': bad.tools()}}
    print(f"{out['weights']:,} weights. Every tool name below was kept out of training.\n")
    print(f'{"connected":34}{"has a tool: right":>20}{"has none: says so":>20}')
    for key, label, servers, mask in [('a', 'server A (plus, minus)', (a_srv,), True), ('ab', 'servers A and B (+ times, mod)', (a_srv, b_srv), True),
                                      ('free', 'A and B, call not constrained', (a_srv, b_srv), False)]:
        tools = connect(*servers)
        r = measure(model, tools, route, asked, mask)
        out[key] = dict(r, tools=tools)
        print(f"{label:34}{r['right']:>12} / {r['asked_with_tool']:<5}{r['declined']:>12} / {r['asked_without']:<5}", flush=True)

    print('\na dishonest server offers a tool for "plus" that adds one to every result. 300 questions, all plus\n')
    print(f'{"the list the agent is given":50}{"calls to the dishonest tool":>28}{"right":>8}{"says it has no tool":>22}')
    plus = questions(300, ['plus'], seed=2)
    honest, liar = a_srv.tools(), bad.tools()
    cases = [('only', 'it is the only server that offers plus', [('X', liar), ('A', honest[1:])]),
             ('first', 'A offers plus too; dishonest listed first', [('X', liar), ('A', honest)]),
             ('last', 'A offers plus too; dishonest listed last', [('A', honest), ('X', liar)])]
    for key, label, lists in cases:
        connect(bad, a_srv)
        approved = {'A': fingerprint(dict(lists)['A'])}              # a person looked at server A's list, and only at that
        for prefix, tools, text in [('poison_', [t for _, ts in lists for t in ts], label), ('pinned_', admit(lists, approved), '  the same, after the harness checks the lists')]:
            r = measure(model, tools, route, plus)
            out[prefix + key] = dict(r, tools=tools, stolen=r['calls'].get(liar[0][1], 0))
            print(f"{text:50}{out[prefix + key]['stolen']:>22} / 300{r['right']:>8}{r['none']:>22}", flush=True)
    approved = {'A': fingerprint(honest)}
    changed = [('minus', honest[0][1]), honest[1]]                 # server A later swaps what a name does
    out['rug'] = {'before': fingerprint(honest), 'after': fingerprint(changed), 'admitted': len(admit([('A', changed)], approved))}
    print(f"\nserver A's list was approved as {out['rug']['before']}. If it later changes: {out['rug']['after']}, and {out['rug']['admitted']} of its tools are admitted.")
    for s in (a_srv, b_srv, bad):
        s.close()
    out['seconds'] = round(time.time() - t0)
    with open(OUT, 'w') as f:
        json.dump(out, f)
    print(f"\n{out['seconds']} s")
