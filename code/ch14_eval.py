"""
ch14_eval.py — ask the same questions every time, and count.           (Chapter 14)

    python code/ch14_eval.py --mock              # the plumbing, with the fake server
    python code/ch14_eval.py --repeat 3          # a real model (chapter 10), three runs per task
    python code/ch14_eval.py --mock --save       # also write runs/evals.json for the website

An evaluation is three things: a fixed list of tasks, a grader that decides
pass or fail without a person looking, and a count. Run it before and after
every change. The traces of chapter 14's other script are kept with each
result, so a failure can be opened and read.

With --mock the "model" is the rule-based fake server. Its score says nothing
about any real model. It is useful because its mistakes are real mistakes of
the kinds a model makes (wrong tool, bad arguments, half the job), and the
graders have to catch them.
"""
import argparse, json, math, os, re, sys
from ch14_trace import Tracer, run_traced, start_mock

HERE = os.path.dirname(os.path.abspath(__file__))

# Each task is a question plus what a correct run must show.
#   number     the final answer contains this number
#   says       the final answer contains one of these words
#   calls      these tools were called and did not fail
#   retrieves  a search result contained this text
TASKS = [
    {'q': 'What is 17 * 2340 / 100?', 'number': 397.8, 'calls': ['calculator']},
    {'q': 'What is 2 ** 10?', 'number': 1024, 'calls': ['calculator']},
    {'q': 'What is (3 + 4) * 12?', 'number': 84, 'calls': ['calculator']},
    {'q': 'What is 17% of 2340?', 'number': 397.8, 'calls': ['calculator']},
    {'q': 'What is 1 / 0?', 'says': ['zero', 'undefined', 'cannot', 'impossible']},
    {'q': 'What do the notes say about the KV cache?', 'calls': ['search_notes'], 'retrieves': 'KV cache'},
    {'q': 'According to the notes, why are attention scores divided by a square root?',
     'calls': ['search_notes'], 'retrieves': 'square root'},
    {'q': 'According to the notes, what does LoRA train?', 'calls': ['search_notes'], 'retrieves': 'LoRA'},
    {'q': 'What is 4.85 * 8 / 8, and what do the notes say about quantization?',
     'number': 4.85, 'calls': ['calculator', 'search_notes'], 'retrieves': 'uantiz'},
]


# [snippet:grade]
def grade(task, answer, spans):
    """Return {check: passed}. Some checks read the final answer, others read what the agent did."""
    tools = [s for s in spans if s['kind'] == 'tool']
    out = {}
    if 'number' in task:                               # outcome: the right number is in the answer
        found = [float(x) for x in re.findall(r'-?\d+\.?\d*', answer.replace(',', ''))]
        out['answer'] = any(abs(x - task['number']) <= 1e-3 * abs(task['number']) for x in found)
    if 'says' in task:                                 # outcome: the answer uses one of these words
        out['answer'] = any(word in answer.lower() for word in task['says'])
    if 'calls' in task:                                # path: these tools ran, and did not fail
        worked = {s['name'] for s in tools if 'error' not in s}
        out['tools'] = all(name in worked for name in task['calls'])
    if 'retrieves' in task:                            # path: the search found the right passage
        out['retrieval'] = any(task['retrieves'] in str(s.get('output')) for s in tools if s['name'] == 'search_notes')
    return out
# [/snippet]


# [snippet:evaluate]
def evaluate(tasks, repeat=1):
    rows = []
    for task in tasks:
        for _ in range(repeat):                        # a real model is random: one run is one sample
            tracer = Tracer()
            try:
                answer, _ = run_traced(task['q'], tracer)
            except OSError:
                raise                                  # no server: stop, this is not the agent's fault
            except Exception as e:
                answer = f'crashed: {type(e).__name__}: {e}'       # a crash is a failed task, with its trace
            checks = grade(task, answer, tracer.spans)
            rows.append({'q': task['q'], 'answer': answer, 'checks': checks,
                         'passed': all(checks.values()), 'spans': tracer.spans})
    return rows

def score(rows):
    """Pass rate, and the standard error that says how far to trust it."""
    n = len(rows)
    p = sum(r['passed'] for r in rows) / n
    return p, math.sqrt(p * (1 - p) / n)
# [/snippet]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mock', action='store_true', help='use the fake server of this repository')
    ap.add_argument('--repeat', type=int, default=1, help='runs per task')
    ap.add_argument('--save', action='store_true', help='write runs/evals.json')
    args = ap.parse_args()
    if args.mock:
        start_mock()
    try:
        rows = evaluate(TASKS, args.repeat)
    except OSError as e:
        sys.exit(f'No model server reachable ({e}). Start one (chapter 10) or add --mock.')

    for r in rows:
        failed = [name for name, ok in r['checks'].items() if not ok]
        steps = sum(s['kind'] == 'model' for s in r['spans'])
        print(f"{'pass' if r['passed'] else 'FAIL'}  {r['q'][:58]:58}  {steps} model calls"
              + (f"  failed: {', '.join(failed)}" if failed else ''))
    p, se = score(rows)
    print(f"\n{sum(r['passed'] for r in rows)} of {len(rows)} passed: {100 * p:.0f}% "
          f"± {100 * se:.0f} (one standard error, {len(rows)} runs)")
    if args.mock:
        print('This is the fake server. The number describes its rules, not a model.')
    if args.save:
        path = os.path.join(HERE, '..', 'runs', 'evals.json')
        with open(path, 'w') as f:
            json.dump({'model': 'mock server' if args.mock else os.environ.get('MICROAI_MODEL', 'local model'),
                       'rows': rows, 'pass_rate': p, 'standard_error': se}, f, default=str)
        print('wrote', os.path.normpath(path))
