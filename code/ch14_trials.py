"""
ch14_trials.py — the same task, several times.                        (Chapter 14)

A model that samples its answers can pass a task once and fail it the next time.
One run per task hides that. This script runs every task ten times and counts
two different things:

    at least one of k runs passes    (written pass@k: fine when you can try again)
    all k runs pass                  (written pass^k: what a user who asks k times sees)

The model is the "as found" one of chapter 24, which is right about half the time.

    python code/ch14_trials.py
"""
import json, os
import numpy as np
import ch24_posttrain as post
import ch19_system as big

OUT = os.path.join(post.RUNS, 'trials.json')


def tasks(n=300, seed=1):
    rng, out = np.random.default_rng(seed), []
    while len(out) < n:
        a, b = int(rng.integers(100)), int(rng.integers(100))
        if big.held_out(a, b):
            out.append(f'what is {a} {list(big.WORDS)[rng.integers(3)]} {b}?')
    return out

# [snippet:trials]
def trials(run, questions, k=10, seed=0):
    """Every task, k times. One row per task, one 0 or 1 per trial."""
    rng = np.random.default_rng(seed)
    return np.array([[post.grade(q, post.episode(run, q, rng)) == 1 for _ in range(k)] for q in questions])

def at_least_one(passed, k):
    return float(passed[:, :k].any(axis=1).mean())      # pass@k

def every_time(passed, k):
    return float(passed[:, :k].all(axis=1).mean())      # pass^k
# [/snippet]


def main():
    run = post.live(post.load('pre'))
    questions = tasks()
    passed = trials(run, questions)
    p = float(passed.mean())
    out = {'tasks': len(questions), 'trials': passed.shape[1], 'rate': round(p, 4),
           'first_run': int(passed[:, 0].sum()),
           'per_task': np.bincount(passed.sum(axis=1), minlength=passed.shape[1] + 1).tolist(),
           'rows': []}
    print(f'{len(questions)} tasks x {passed.shape[1]} trials: {passed.sum()} of {passed.size} passed, {100 * p:.1f}%')
    print('tasks by number of passes out of ten:', out['per_task'])
    print(' k   at least one   (if every task had the same chance)   every time   (same)')
    for k in range(1, passed.shape[1] + 1):
        row = [k, round(at_least_one(passed, k), 4), round(1 - (1 - p) ** k, 4), round(every_time(passed, k), 4), round(p ** k, 4)]
        out['rows'].append(row)
        print(f'{k:2d}   {100 * row[1]:5.1f}%        {100 * row[2]:5.1f}%                              {100 * row[3]:5.1f}%      {100 * row[4]:5.1f}%')
    json.dump(out, open(OUT, 'w'))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
