"""
ch32_helpers.py — a job split among helpers, each with a clean context.   (Chapter 32)

    python code/ch32_helpers.py        # about 5 minutes; needs the models of chapter 24

The job: "add up: 23 times 45, 17 minus 3, 8 plus 9". Our agent can only hold one
sum in its window of 73 characters, so no single context can do this job. A lead
splits it, gives each part to a helper that starts with an empty context, and
gets back one number from each. The lead here is a few lines of code.

Measured for jobs of 1 to 6 parts:
    how much text each context holds, against what one shared context would need
    how often the total is right, with three kinds of helper
"""
import json, os, sys
import numpy as np
import ch19_system as big
import ch24_posttrain as post

OUT = os.path.join(post.RUNS, 'helpers.json')


def job(rng, parts):
    """A job of several parts, and its right total. Number pairs are ones no model was trained on."""
    terms, total = [], 0
    while len(terms) < parts:
        a, b = int(rng.integers(100)), int(rng.integers(100))
        if big.held_out(a, b):
            word = list(big.WORDS)[rng.integers(3)]
            terms.append(f'{a} {word} {b}')
            total += big.TOOLS[big.WORDS[word]](a, b)
    return 'add up: ' + ', '.join(terms), total

# [snippet:helper]
def helper(run, part, rng, tries=1):
    """One helper: a fresh context, one part of the job. Returns its number (or None) and the text it used.
    With tries > 1 it is asked again until its answer rests on a tool result (the check of chapter 25)."""
    used = 0
    for _ in range(tries):
        e = post.episode(run, f'what is {part}?', rng)
        used += len(e['text'])
        if tries == 1 or e['call'] is not None:
            break
    last = e['answer'].rstrip('.').split(' ')[-1]
    return (int(last) if last.lstrip('-').isdigit() else None), used, len(e['text'])
# [/snippet]

# [snippet:lead]
def lead(run, request, rng, tries=1):
    """The lead: split the job, hand out the parts, add up what comes back. It never sees a helper's working."""
    parts = request[len('add up: '):].split(', ')
    results = [helper(run, part, rng, tries) for part in parts]        # each could run at the same time
    numbers = [r[0] for r in results]
    total = None if None in numbers else sum(numbers)                  # the lead's own tool call
    return {'total': total, 'tokens': sum(r[1] for r in results), 'largest': max(r[2] for r in results),
            'lead_context': len(request) + sum(len(str(n)) + 3 for n in numbers) + len(str(total)) + 12}
# [/snippet]


if __name__ == '__main__':
    if not os.path.exists(post.PATH['pre']):
        sys.exit('Needs the models of chapter 24. Run:  python code/ch24_posttrain.py --train')
    kinds = {'careless helper (pre-trained only)': ('pre', 1), 'careless helper, checked and asked again': ('pre', 5), 'trained helper (chapter 24)': ('rl', 1)}
    runs = {name: post.live(post.load(name)) for name in ('pre', 'rl')}
    rows = []
    print(f'{"parts":>6}{"largest helper context":>24}{"lead context":>14}{"one shared context would need":>32}   right, of 200: ' + ' | '.join(kinds))
    for parts in range(1, 7):
        row = {'parts': parts, 'right': {}}
        for kind, (model, tries) in kinds.items():
            rng, ok, largest, lead_ctx, shared, tokens = np.random.default_rng(parts), 0, 0, 0, 0, 0
            for _ in range(200):
                request, total = job(rng, parts)
                r = lead(runs[model], request, rng, tries)
                ok += r['total'] == total
                largest, lead_ctx, tokens = max(largest, r['largest']), max(lead_ctx, r['lead_context']), tokens + r['tokens']
            row['right'][kind] = ok
            row.setdefault('tokens', {})[kind] = round(tokens / 200)
            if model == 'rl':
                row.update(largest=largest, lead_context=lead_ctx, shared=len(request) + round(tokens / 200))
        rows.append(row)
        print(f"{parts:>6}{row['largest']:>24}{row['lead_context']:>14}{row['shared']:>32}   " + ' | '.join(f"{row['right'][k]:>4}" for k in kinds), flush=True)
    print('\ntokens per job:', {r['parts']: r['tokens'] for r in rows})
    with open(OUT, 'w') as f:
        json.dump({'rows': rows, 'kinds': list(kinds), 'window': post.WIDTH}, f)
