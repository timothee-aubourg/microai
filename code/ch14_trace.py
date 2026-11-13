"""
ch14_trace.py — write down everything the agent does.                  (Chapter 14)

    python code/ch14_trace.py --mock "What is (3 + 4) * 12?"
    python code/ch14_trace.py "What is 17 * 2340 / 100?"        # with a real server (chapter 10)

The agent of chapter 11 returns one string. When that string is wrong you need
to know which step went wrong: the model's choice of tool, the arguments it
wrote, the tool itself, or the final wording. A trace is the record that tells
you. It is a list of spans, and a span is one unit of work: what went in, what
came out, how long it took, and whether it failed.

--mock starts the fake server of this repository inside the script, so the
plumbing can be watched on a machine with no model installed.
"""
import copy, json, sys, threading, time
from contextlib import contextmanager
import ch11_agent as agent
import llm


# [snippet:tracer]
class Tracer:
    def __init__(self):
        self.spans, self.stack, self.t0 = [], [], time.perf_counter()

    @contextmanager
    def span(self, kind, name, input=None):
        s = {'id': len(self.spans), 'parent': self.stack[-1] if self.stack else None,
             'kind': kind, 'name': name, 'input': copy.deepcopy(input),      # a copy: the caller may change it later
             'start': round(1000 * (time.perf_counter() - self.t0), 2)}
        self.spans.append(s); self.stack.append(s['id'])
        t = time.perf_counter()
        try:
            yield s                                    # the work happens here and may set s['output']
        except Exception as e:
            s['error'] = f'{type(e).__name__}: {e}'    # record the failure, then let it continue upward
            raise
        finally:
            s['ms'] = round(1000 * (time.perf_counter() - t), 2)
            self.stack.pop()
# [/snippet]


# [snippet:wrap]
def traced(tracer, kind, name, fn):
    """The same function, with every call recorded as a span."""
    def wrapper(*args, **kwargs):
        with tracer.span(kind, name, input={'args': args, 'kwargs': kwargs}) as s:
            s['output'] = fn(*args, **kwargs)
            return s['output']
    return wrapper

def run_traced(question, tracer=None, **kw):
    """Run the agent of chapter 11, unchanged, with its model calls and tool calls recorded."""
    tracer = tracer or Tracer()
    chat, registry = agent.chat, agent.REGISTRY
    agent.chat = traced(tracer, 'model', 'chat', chat)
    agent.REGISTRY = {name: traced(tracer, 'tool', name, fn) for name, fn in registry.items()}
    try:
        with tracer.span('agent', 'run_agent', input=question) as root:
            root['output'] = agent.run_agent(question, verbose=False, **kw)
    finally:
        agent.chat, agent.REGISTRY = chat, registry    # put the originals back
    return root['output'], tracer.spans
# [/snippet]


def brief(value, n=70):
    text = (value if isinstance(value, str) else json.dumps(value, default=str)).replace('\n', ' ')
    return text if len(text) <= n else text[:n - 1] + '…'


def describe(s):
    """One line per span: what a person scanning the trace needs to see."""
    if s['kind'] == 'model':
        messages = s['input']['args'][0]
        reply = s.get('output') or {}
        calls = reply.get('tool_calls') or []
        said = ', '.join(f"{c['function']['name']}({brief(c['function']['arguments'], 40)})" for c in calls) or brief(reply.get('content') or '')
        return f'{len(messages)} messages, {len(json.dumps(messages))} characters in -> {said}'
    if s['kind'] == 'tool':
        return f"{brief(s['input']['kwargs'], 40)} -> {'ERROR ' + s['error'] if 'error' in s else brief(s.get('output'), 50)}"
    return f"{brief(s['input'])} -> {brief(s.get('output', ''))}"


def show(spans):
    depth = {None: -1}
    for s in spans:
        depth[s['id']] = depth[s['parent']] + 1
        print(f"{s['start']:8.1f} ms  {'  ' * depth[s['id']]}{s['kind']:5} {s['name']:13} {s['ms']:7.1f} ms  {describe(s)}")


def start_mock():
    """Run code/mock_server.py in a background thread and point llm.py at it."""
    from http.server import HTTPServer
    from mock_server import Handler
    server = HTTPServer(('127.0.0.1', 0), Handler)       # port 0: any free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    llm.BASE_URL = f'http://127.0.0.1:{server.server_address[1]}/v1'


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--mock' in args:
        args.remove('--mock'); start_mock()
    question = ' '.join(args) or 'What is 17 * 2340 / 100?'
    try:
        answer, spans = run_traced(question)
    except OSError as e:
        sys.exit(f'No model server reachable ({e}). Start one (chapter 10) or add --mock.')
    show(spans)
    print('\nanswer:', answer)
