"""
ch46_runstart.py — what every run starts with, and tidying memory between runs.   (Chapter 46)

    python code/ch46_runstart.py     # about 3 minutes; needs  python code/ch39_plan.py --train

An always-on agent starts many short runs instead of one long conversation. Two measurements:

    1. The fixed beginning. Every run starts with the same system prompt and tool descriptions,
       here those of the harness of chapter 15. With the runs of a simulated day (chapter 44),
       how much of it is read again, and how much could a prefix cache (chapter 16) save, if a
       cached copy lasts five minutes after the last run that used it?

    2. Tidying memory between runs ("dreaming"). The harness of chapter 42 kept a log of every
       fact, and showing all of it went badly. Between runs, while nobody waits, a tidy-up pass
       can rewrite the notes: keep the latest value of each name, and drop names nobody asked
       about in the last 20 sessions. Then every run shows the whole tidied list, without picking.
       Measured with the agent of chapter 39 on the people of chapter 42.
"""
import json, os, sys
import numpy as np
import ch15_harness as h15
import ch44_inbox as ib
import ch39_plan as w
import ch42_memory as mem

OUT = os.path.join(w.RUNS, 'runstart.json')
FIXED = len(h15.SYSTEM) + len(json.dumps(h15.TOOLS))     # characters every run of that harness starts with
TTL = 5 * 60                                             # a cached prefix lasts five minutes after its last use


# [snippet:cache]
def day_of_runs(seed):
    """The start and end times of the runs of one day, from the batched inbox of chapter 44."""
    events = ib.day_of_events(np.random.default_rng(seed))
    return [(s, e) for s, e, ok in ib.inbox(events, np.random.default_rng(1000 + seed), batch=True, keep_runs=True)]

def cache_hits(runs, ttl=TTL):
    """A run finds the fixed beginning in the cache if the run before it ended less than `ttl` ago."""
    return sum(1 for (s0, e0), (s1, e1) in zip(runs, runs[1:]) if s1 - e0 < ttl)
# [/snippet]


# [snippet:tidy]
def tidy(log, asked, now, unused=20):
    """The tidy-up between runs: the latest value of each name, and only names asked about
    in the last `unused` sessions (or stated in them)."""
    latest, last_seen = {}, {}
    for t, n, v in log:
        latest[n] = v
        last_seen[n] = max(last_seen.get(n, t), t)
    for t, n in asked:
        last_seen[n] = max(last_seen.get(n, t), t)
    return [(n, v) for n, v in latest.items() if now - last_seen[n] <= unused]
# [/snippet]


if __name__ == '__main__':
    out = {'fixed': FIXED, 'system': len(h15.SYSTEM), 'tools': len(json.dumps(h15.TOOLS))}
    days = [day_of_runs(d) for d in range(20)]
    n_runs = float(np.mean([len(r) for r in days]))
    hits = float(np.mean([cache_hits(r) for r in days]))
    out.update(runs=round(n_runs, 1), hits=round(hits, 1), fixed_per_day=round(n_runs * FIXED))
    print(f'fixed beginning: {FIXED} characters ({len(h15.SYSTEM)} system prompt, {out["tools"]} tool descriptions)')
    print(f'{n_runs:.1f} runs a day read it {n_runs * FIXED:,.0f} times over; '
          f'{hits:.1f} runs start within {TTL // 60} minutes of the last one and could find it cached')
    gaps = [s1 - e0 for r in days for (s0, e0), (s1, e1) in zip(r, r[1:])]
    out['median_gap'] = round(float(np.median(gaps)))
    print(f'median time between one run ending and the next starting: {np.median(gaps):.0f} s')

    if not os.path.exists(w.PATH):
        sys.exit('No trained agent yet. Run:  python code/ch39_plan.py --train')
    run = w.live()
    rows = []
    for n_names in (4, 8, 16):
        rng = np.random.default_rng(100 + n_names)
        names = list(rng.choice(list(w.NAMES), size=n_names, replace=False))
        res = {k: {'right': 0, 'read': 0, 'notes': 0} for k in ('every note', 'tidied', 'latest, picked')}
        asked_n = 0
        for person in range(8):
            log, asked = [], []
            for t, e in enumerate(mem.history(rng, names, sessions=80)):
                if e[0] == 'fact':
                    log.append((t, e[1], e[2]))
                    continue
                _, nums, ops, use, want = e
                q = w.question(nums, ops, use)
                room = w.BLOCK - len(q) - 60
                shown = {'every note': mem.notes('every note', [(n, v) for _, n, v in log], set(use.values()), room),
                         'tidied': [f'M {n} is {v}' for n, v in tidy(log, asked, t)],
                         'latest, picked': mem.notes('latest, picked', [(n, v) for _, n, v in log], set(use.values()), room)}
                while sum(len(l) + 1 for l in shown['tidied']) > room:
                    shown['tidied'] = shown['tidied'][1:]
                for k, lines in shown.items():
                    got, _, read = w.run_steps(run, w.prompt_for(nums, ops, lines, use))
                    res[k]['right'] += got == want
                    res[k]['read'] += read
                    res[k]['notes'] += len(lines)
                asked += [(t, n) for n in use.values()]
                asked_n += 1
        for k in res:
            res[k]['read'] = round(res[k]['read'] / asked_n)
            res[k]['notes'] = round(res[k]['notes'] / asked_n, 1)
        rows.append({'names': n_names, 'asked': asked_n, 'ways': res})
        print(f'{n_names:>3} names, {asked_n} questions: ' + '   '.join(f"{k} {v['right']} right, {v['notes']} notes, {v['read']} read" for k, v in res.items()), flush=True)
    out['tidy'] = rows
    with open(OUT, 'w') as f:
        json.dump(out, f)
    print('wrote', os.path.normpath(OUT))
