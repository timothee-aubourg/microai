"""
ch15_harness.py — the program around the model.                        (Chapter 15)

    python code/ch15_harness.py --mock                    # the plumbing, on a small demo project
    python code/ch15_harness.py --dir my_project "The tests fail. Find out why and fix it."

A model writes text and nothing else. A harness is the ordinary program that
turns that text into work: it tells the model which tools exist, runs the tool
the model asks for, and feeds the result back as more text. Tools such as
Claude Code or Pi are harnesses. This file is a small one: four tools (read,
write, edit, bash), a system prompt, a loop, and a question before anything
that changes your machine.

With a real model, `bash` runs real commands. Point --dir at a folder you can
afford to lose, and read chapter 17 before passing --yes.
"""
import argparse, copy, json, os, shutil, subprocess, sys, tempfile
import llm


# [snippet:tools]
def inside(root, path):
    """Resolve `path` inside the project folder, or refuse."""
    root, full = os.path.realpath(root), os.path.realpath(os.path.join(root, path))
    if os.path.commonpath([root, full]) != root:
        raise PermissionError(f'{path} is outside the project folder')
    return full

def read(root, path):
    with open(inside(root, path)) as f:
        return f.read()

def write(root, path, content):
    full = inside(root, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w') as f:
        f.write(content)
    return f'wrote {len(content)} characters to {path}'

def edit(root, path, old, new):
    text = read(root, path)
    if text.count(old) != 1:                              # the model must point at exactly one place
        raise ValueError(f'the old text occurs {text.count(old)} times in {path}; it must occur once')
    return write(root, path, text.replace(old, new))

def bash(root, command):
    r = subprocess.run(command, shell=True, cwd=root, capture_output=True, text=True, timeout=60)
    return f'exit code {r.returncode}\n{r.stdout}{r.stderr}'

REGISTRY = {'read': read, 'write': write, 'edit': edit, 'bash': bash}
# [/snippet]


def tool(name, description, **params):
    """What the model is told about one tool: a name, a sentence, and its arguments."""
    return {'type': 'function', 'function': {'name': name, 'description': description, 'parameters': {
        'type': 'object', 'required': list(params),
        'properties': {k: {'type': 'string', 'description': v} for k, v in params.items()}}}}

# [snippet:describe]
TOOLS = [
    tool('read', 'Read a text file in the project.', path='File path, relative to the project folder.'),
    tool('write', 'Create a file, or replace its whole content.', path='File path.', content='The complete new content.'),
    tool('edit', 'Replace one passage of a file. The old text must occur exactly once.',
         path='File path.', old='The exact text to replace.', new='The text to put in its place.'),
    tool('bash', 'Run a shell command in the project folder. Returns the exit code and the output.',
         command='The command line.'),
]
SYSTEM = ('You are a coding assistant working in a project folder on the user\'s machine. '
          'Look before you change anything, and check your work after. '
          'When the task is done, reply with a short summary and no tool call.')
# [/snippet]


# [snippet:loop]
def clip(text, limit=4000):
    """A tool result goes into the context window, so it gets a budget: keep the start and the end."""
    if len(text) <= limit:
        return text
    return text[:limit // 2] + f'\n[... {len(text) - limit} characters left out ...]\n' + text[-limit // 2:]

def run(task, root, approve=lambda name, args: True, max_steps=12, on_step=None):
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task}]
    for _ in range(max_steps):
        reply = llm.chat(messages, tools=TOOLS, temperature=0)       # 1. the model writes
        if on_step:
            on_step(messages, reply)
        messages.append(reply)
        if not reply.get('tool_calls'):
            return reply['content'], messages                        # no request: the model is done
        for call in reply['tool_calls']:
            name, args = call['function']['name'], call['function']['arguments']
            try:
                args = json.loads(args) if isinstance(args, str) else args
                if approve(name, args):                              # 2. the harness decides
                    result = REGISTRY[name](root, **args)            # 3. the harness acts
                else:
                    result = 'denied: this action was not allowed'
            except Exception as e:
                result = f'error: {type(e).__name__}: {e}'           # a failure is also just text
            messages.append({'role': 'tool', 'tool_call_id': call.get('id', ''),
                             'content': clip(str(result))})          # 4. the result becomes context
    return 'Stopped: too many steps.', messages
# [/snippet]


# [snippet:render]
def segments(messages, tools):
    """The request as the model receives it: ONE text, cut here into labelled pieces.
    This follows the layout that Qwen 2.5 models were trained on. Each model family has
    its own, called its chat template, and the server applies it for you."""
    out = [('system prompt', '<|im_start|>system\n' + messages[0]['content']),
           ('tool descriptions',
            '\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n'
            'You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n'
            + '\n'.join(json.dumps(t) for t in tools) + '\n</tools>\n\n'
            'For each function call, return a json object with function name and arguments within '
            '<tool_call></tool_call> XML tags:\n<tool_call>\n'
            '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n')]
    for m in messages[1:]:
        if m['role'] == 'tool':
            out.append(('tool result', '<|im_start|>user\n<tool_response>\n' + m['content'] + '\n</tool_response><|im_end|>\n'))
        elif m.get('tool_calls'):
            calls = [{'name': c['function']['name'], 'arguments': json.loads(c['function']['arguments'])} for c in m['tool_calls']]
            out.append(('model', '<|im_start|>assistant\n' + '\n'.join(
                '<tool_call>\n' + json.dumps(c) + '\n</tool_call>' for c in calls) + '<|im_end|>\n'))
        else:
            out.append(('user' if m['role'] == 'user' else 'model', f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"))
    return out + [('cue', '<|im_start|>assistant\n')]                # the model continues from here
# [/snippet]


PYTHON = 'python3' if shutil.which('python3') else 'python'
DEMO_TASK = f'The tests fail. Run them with `{PYTHON} test_stats.py`, find the bug and fix it.'

def demo_project(folder, planted=''):
    """A two-file project with one bug. `planted` is extra text for stats.py (used in chapter 17)."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'stats.py'), 'w') as f:
        f.write(planted + 'def mean(xs):\n    """Average of a list of numbers."""\n    return sum(xs) / (len(xs) - 1)\n')
    with open(os.path.join(folder, 'test_stats.py'), 'w') as f:
        f.write('from stats import mean\n\ngot = mean([1, 2, 3])\nif got != 2:\n'
                "    raise SystemExit(f'FAIL in stats.py: mean([1, 2, 3]) should be 2, got {got}')\n"
                "print('ok: 1 test passed')\n")
    return folder


def ask(name, args):
    """Reading is free. Anything that changes the machine needs a yes."""
    if name == 'read':
        return True
    return input(f'  allow {name} {json.dumps(args)[:200]} ? [y/N] ').strip().lower() == 'y'


def record(task, root, approve):
    """Run the harness and keep, for every model call, exactly what the model received."""
    steps = []
    answer, messages = run(task, root, approve,
                           on_step=lambda m, reply: steps.append({'segments': segments(m, TOOLS), 'reply': copy.deepcopy(reply)}))
    results = [m['content'] for m in messages if m['role'] == 'tool']
    for s in steps:                                       # attach each tool result to the call that caused it
        s['results'] = [results.pop(0) for _ in s['reply'].get('tool_calls') or []]
    return {'task': task, 'answer': answer, 'steps': steps}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('task', nargs='*')
    ap.add_argument('--dir', help='project folder to work in (default: a temporary demo project)')
    ap.add_argument('--mock', action='store_true', help='use the fake server of this repository')
    ap.add_argument('--yes', action='store_true', help='do not ask before writing files or running commands')
    ap.add_argument('--save', action='store_true', help='write runs/harness.json')
    args = ap.parse_args()
    if args.mock:
        from ch14_trace import start_mock
        start_mock()
    demo = not args.dir
    root = args.dir or demo_project(os.path.join(tempfile.mkdtemp(prefix='microai-'), 'project'))
    task = ' '.join(args.task) or DEMO_TASK
    approve = (lambda name, a: True) if (args.yes or (demo and args.mock)) else ask
    try:
        log = record(task, root, approve)
    except OSError as e:
        sys.exit(f'No model server reachable ({e}). Start one (chapter 10) or add --mock.')

    print(f'project: {root}\ntask:    {task}\n')
    for n, s in enumerate(log['steps'], 1):
        size = sum(len(text) for _, text in s['segments'])
        calls = s['reply'].get('tool_calls') or []
        print(f'model call {n}: received {size} characters')
        for c, result in zip(calls, s['results']):
            print(f"  asks for  {c['function']['name']} {c['function']['arguments'][:90]}")
            print(f"  harness   {result.strip().splitlines()[0][:90] if result.strip() else '(empty)'}")
        if not calls:
            print(f"  answers   {log['answer']}")
    if args.save:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'harness.json')
        log['model'] = 'mock server' if args.mock else os.environ.get('MICROAI_MODEL', 'local model')
        with open(path, 'w') as f:
            json.dump(log, f)
        print('wrote', os.path.normpath(path))
