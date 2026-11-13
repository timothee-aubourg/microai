"""
ch33_selftrain.py — a model that learns from its own answers.   (Chapter 33)

    python code/ch33_selftrain.py        # three loops of four rounds, about 8 minutes; needs chapter 24's models

Nobody writes an example and nobody gives a score. Each round:

    1. the model answers 600 new questions                       (its own experience)
    2. something decides which of those answers to keep
    3. the model is trained on the answers that were kept, with the ordinary loss
    4. it is tested on 300 sums it has never seen

The loop is run three times from the same starting model. Only step 2 differs:

    check    keep an answer only if it rests on a tool result (the check of chapter 25)
    all      keep every answer
    short    keep only the shortest answers (a check that rewards the wrong thing)
"""
import json, os, sys, time
import numpy as np
import ch19_system as big
import ch24_posttrain as post

OUT = os.path.join(post.RUNS, 'selftrain.json')

# [snippet:keep]
KEEP = {
    'check': lambda e: e['call'] is not None,            # did the number come from the calculator?
    'all':   lambda e: True,                             # no judgement at all
    'short': lambda e: len(e['text']) < 50,              # "quick answers are good answers"
}
# [/snippet]

def questions(rng, n):
    out = []
    while len(out) < n:
        a, b = int(rng.integers(100)), int(rng.integers(100))
        if not big.held_out(a, b):                       # never the pairs used for testing
            out.append(f'what is {a} {list(big.WORDS)[rng.integers(3)]} {b}?')
    return out

# [snippet:loop]
def improve(keep, rounds=4, asked=600, seed=0):
    """Answer, select, train on the selection, repeat. Returns what the model does after each round."""
    model, rng = post.load('pre'), np.random.default_rng(seed)
    run = post.live(model)
    history = [dict(round=0, kept=None, **post.behaviour(run, 300), chat=post.chat(run))]
    for r in range(1, rounds + 1):
        episodes = [post.episode(run, q, rng) for q in questions(rng, asked)]        # 1. its own answers
        kept = [e for e in episodes if keep(e)]                                       # 2. the selection
        if kept:
            post.fit(model, post.rows([e['text'][1:] + '\n' for e in kept]), 200, 3e-4, seed=r, label=f'round {r}')   # 3. ordinary training
        history.append(dict(round=r, kept=len(kept), **post.behaviour(run, 300), chat=post.chat(run)))               # 4. the test
    return history
# [/snippet]


if __name__ == '__main__':
    if not os.path.exists(post.PATH['pre']):
        sys.exit('Needs the models of chapter 24. Run:  python code/ch24_posttrain.py --train')
    out = {}
    for name, keep in KEEP.items():
        t0 = time.time()
        history = improve(keep)
        out[name] = history
        print(f'\nkeep: {name}   ({time.time() - t0:.0f}s)')
        print(f'   {"round":>6}{"answers kept":>14}{"uses the calculator":>22}{"sum right":>12}{"small talk":>12}')
        for h in history:
            kept = '' if h['kept'] is None else f"{h['kept']} / 600"
            print(f"   {h['round']:>6}{kept:>14}{h['tool']:>16} / 300{h['right']:>6} / 300{h['chat']:>9} / 3", flush=True)
    with open(OUT, 'w') as f:
        json.dump(out, f)
