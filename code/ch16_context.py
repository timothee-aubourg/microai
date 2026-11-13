"""
ch16_context.py — what each agent step costs, and how to pay less.     (Chapter 16)

    python code/ch16_context.py

Every step of an agent sends the whole conversation again. This script shows
two things about that, both measured:

  1. on the model of this course: a server that keeps the KV cache of the last
     request only has to compute what is new, as long as the beginning of the
     text has not changed (a "prefix cache");
  2. on the recorded harness session of chapter 15: how large each request is,
     how much of it was already seen, and what trimming or summarising old
     tool results does to both the size and the cache.
"""
import json, os, time
import numpy as np
from ch08_sample import Runner

HERE = os.path.dirname(os.path.abspath(__file__))


# [snippet:prefix]
class PrefixCache:
    """Keep the keys and values of the last text. For the next text, compute only what differs."""
    def __init__(self, run):
        self.run, self.ids, self.cache = run, [], run.new_cache()

    def read(self, ids):
        same = 0
        while same < min(len(ids), len(self.ids)) and ids[same] == self.ids[same]:
            same += 1                                  # how long is the shared beginning?
        same = min(same, len(ids) - 1)                 # always compute the last token: we need its scores
        for layer in self.cache['k'] + self.cache['v']:
            del layer[same:]                           # everything after the first difference is void
        self.cache['pos'] = same
        for t in ids[same:]:
            logits = self.run.step(t, self.cache)      # the only real work
        self.ids = list(ids)
        return logits, len(ids) - same                 # scores for the next token, tokens computed
# [/snippet]


def size(messages):
    return sum(len(json.dumps(m)) for m in messages)


# [snippet:trim]
def trim_results(messages, keep=2, limit=200):
    """Old tool results are most of an agent's context. Shorten all but the latest `keep`."""
    out, seen = [], 0
    for m in reversed(messages):
        if m['role'] == 'tool':
            seen += 1
            if seen > keep and len(m['content']) > limit:
                m = {**m, 'content': m['content'][:limit] + f' [trimmed: {len(m["content"]) - limit} more characters]'}
        out.append(m)
    return out[::-1]

def compact(messages, summarize, keep=4):
    """Replace the middle of a long conversation with a summary. `summarize` is one more model call."""
    cut = len(messages) - keep
    while cut > 2 and messages[cut]['role'] == 'tool':
        cut -= 1                                       # never separate a result from the request before it
    if cut <= 2:
        return messages
    note = summarize(messages[2:cut])
    return messages[:2] + [{'role': 'user', 'content': 'Summary of the work so far:\n' + note}] + messages[cut:]
# [/snippet]


def shared(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def tiny_session(run, clock=False):
    """An agent session in miniature, sized to the 64-character window of our model:
    a fixed head, then one new piece of text per step. With clock=True one character
    near the start changes at every step, as a timestamp in a system prompt would."""
    head = 'First Citizen:\n'
    pieces = ['Before we', ' proceed ', 'any furth', 'er, hear ', 'me speak.']
    texts = []
    for n in range(len(pieces)):
        stamp = 'abcde'[n] if clock else 'a'
        texts.append(stamp + head + ''.join(pieces[:n + 1]))
    return [[run.stoi[c] for c in t] for t in texts]


if __name__ == '__main__':
    run = Runner()

    # ---- 1. the prefix cache, on the real model ---------------------------------------
    print('Five requests, each one the previous text plus nine characters (our model, 64-character window)\n')
    print(f'{"":34}{"tokens computed per request":>30}{"total":>8}{"time":>10}')
    rows = {}
    for label, clock, use_cache in [('no cache', False, False), ('prefix cache, text only grows', False, True),
                                    ('prefix cache, a clock at the start', True, True)]:
        session, pc, counts, spent = tiny_session(run, clock), PrefixCache(run), [], 0.0
        for ids in session:
            if not use_cache:
                pc = PrefixCache(run)                  # a server that forgets between requests
            t0 = time.perf_counter()
            logits, computed = pc.read(ids)
            spent += time.perf_counter() - t0
            counts.append(computed)
            fresh = run.new_cache()                    # the answer must not depend on the cache
            for t in ids:
                check = run.step(t, fresh)
            assert np.abs(logits - check).max() < 1e-3
        rows[label] = sum(counts)
        print(f'{label:34}{str(counts):>30}{sum(counts):>8}{1000 * spent:>8.0f} ms')
    print(f"\nReuse cuts the work by {rows['no cache'] / rows['prefix cache, text only grows']:.1f}x here. "
          f"One changing character at the start cuts it by "
          f"{rows['no cache'] / rows['prefix cache, a clock at the start']:.1f}x: nothing.")

    # ---- 2. the recorded harness session of chapter 15 ---------------------------------
    path = os.path.join(HERE, '..', 'runs', 'harness.json')
    if not os.path.exists(path):
        raise SystemExit('\nruns/harness.json not found: run  python code/ch15_harness.py --mock --save  first.')
    with open(path) as f:
        steps = json.load(f)['steps']
    texts = [''.join(text for _, text in s['segments']) for s in steps]
    print(f'\nThe harness session of chapter 15: {len(texts)} model calls (sizes in characters)\n')
    print(f'{"call":>4}{"request":>10}{"already seen":>14}{"new":>8}')
    total = new_total = 0
    for n, text in enumerate(texts):
        seen = shared(text, texts[n - 1]) if n else 0
        total += len(text); new_total += len(text) - seen
        print(f'{n + 1:>4}{len(text):>10}{seen:>14}{len(text) - seen:>8}')
    fixed = sum(len(text) for kind, text in steps[0]['segments'] if kind in ('system prompt', 'tool descriptions'))
    print(f'\nRead without a cache: {total} characters. With a prefix cache: {new_total} ({total / new_total:.1f}x less).')
    print(f'The system prompt and tool descriptions alone are {fixed} characters, sent {len(texts)} times.')

    # ---- 3. trimming: smaller requests, at the price of one cache miss ----------------
    messages = [{'role': 'system', 'content': 's' * 400}, {'role': 'user', 'content': 'task'}]
    for n in range(6):
        messages.append({'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': str(n), 'type': 'function', 'function': {'name': 'read', 'arguments': json.dumps({'path': f'file{n}.py'})}}]})
        messages.append({'role': 'tool', 'tool_call_id': str(n), 'content': f'content of file {n}: ' + 'x' * 1500})
    trimmed = trim_results(messages)
    squeezed = compact(messages, lambda old: f'Read {len(old) // 2} files; nothing unusual found.')
    a, b, c = json.dumps(messages), json.dumps(trimmed), json.dumps(squeezed)
    print('\nA longer session (six files read, 1,500 characters each):')
    print(f'  as it is             {len(a):>6} characters')
    print(f'  old results trimmed  {len(b):>6} characters, identical to the original for the first {shared(a, b)}')
    print(f'  compacted            {len(c):>6} characters, identical to the original for the first {shared(a, c)}')
    print('  After either rewrite, everything past the first changed character is computed again, once.')
