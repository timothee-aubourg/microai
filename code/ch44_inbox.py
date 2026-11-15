"""
ch44_inbox.py — an agent that is asleep most of the time: events, runs, and an inbox.   (Chapter 44)

    python code/ch44_inbox.py        # a few seconds; no model needed

An always-on agent is not running all the time. It sleeps, an event wakes it, it does one
run (read what is new, act, save what happened), and it sleeps again. This file simulates
one working day of events for one person and handles them in three ways:

    direct            every event starts its own run at once, even if another run is going
    inbox, one by one events are written to a table first; one worker takes them one at a time
    inbox, batched    the same table and worker, but a run takes every event that is waiting

The events are of two kinds. Messages from the person come in bursts: two or three within
half a minute. Notifications from other services (webhooks) are sent again when the sender
is not sure the first one arrived, so some arrive twice with the same id. A run sometimes
crashes part-way. Every run reads the agent's memory when it starts and writes it back when
it ends, so two runs at the same time can overwrite each other's work.

Nothing here needs a model: this is the code around it. A run's length stands for its model
passes: 20 seconds to start, plus 10 seconds per event in it.
"""
import heapq, json, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'inbox.json')
DAY = 8 * 3600                     # one working day, in seconds
START, PER_EVENT = 20, 10          # a run's length: 20 s to start, 10 s per event it handles
CRASH = 0.05                       # chance that a run crashes part-way
RESTART = 60                       # seconds before a crashed worker is replaced


# [snippet:events]
def day_of_events(rng):
    """Events of one day: (time, id, source). Messages come in bursts; some webhooks arrive twice."""
    events, n = [], 0
    for t in np.sort(rng.uniform(0, DAY, 40)):                   # 40 bursts of messages from the person
        for k in range(int(rng.integers(1, 4))):                 # one to three messages within half a minute
            n += 1
            events.append((float(t + 15 * k), f'msg-{n}', 'person'))
    for t in np.sort(rng.uniform(0, DAY, 60)):                   # 60 notifications from other services
        n += 1
        events.append((float(t), f'hook-{n}', 'webhook'))
        if rng.random() < 0.15:                                  # not sure it arrived: sent again, same id
            events.append((float(t + rng.uniform(60, 300)), f'hook-{n}', 'webhook'))
    return sorted(events)
# [/snippet]


def direct(events, rng):
    """Every event starts its own run, at once. Returns what happened to each event and to memory."""
    handled, lost, runs, overlaps, lost_writes = {}, set(), [], 0, 0
    for t, eid, _ in events:
        length = START + PER_EVENT
        end = t + length
        if rng.random() < CRASH:                                 # the run dies part-way: the event is gone
            lost.add(eid)
            runs.append((t, t + rng.uniform(0, length), False))
            continue
        handled[eid] = handled.get(eid, 0) + 1
        runs.append((t, end, True))
    # memory: a run reads it at its start and writes it back at its end. If another run read it before
    # this one wrote, and writes after, it overwrites this run's work with a copy that never saw it.
    finished = [r for r in runs if r[2]]
    for sb, eb, _ in finished:
        if any(sa < eb < ea for sa, ea, _ in finished):
            lost_writes += 1
    for s, e, _ in runs:
        overlaps = max(overlaps, sum(1 for s2, e2, _ in runs if s2 <= s < e2))
    return summary(events, handled, lost, runs, overlaps, lost_writes, waits=[START + PER_EVENT] * len(handled))


# [snippet:inbox]
def inbox(events, rng, batch, keep_runs=False):
    """Events go into a table first (duplicates by id are dropped), and one worker reads the table.
    A crashed run leaves its events in the table, and the next worker takes them again."""
    table, order = {}, []                                        # id -> {'at', 'status'}
    handled, runs, waits = {}, [], []
    arrivals = list(events)
    t, i, worker_free = 0.0, 0, 0.0
    while i < len(arrivals) or any(e['status'] == 'pending' for e in table.values()):
        while i < len(arrivals) and arrivals[i][0] <= max(t, worker_free):
            at, eid, _ = arrivals[i]
            if eid not in table:                                 # the same id twice: written once
                table[eid] = {'at': at, 'status': 'pending'}
                order.append(eid)
            i += 1
        pending = [e for e in order if table[e]['status'] == 'pending']
        if not pending:
            if i >= len(arrivals):
                break
            t = arrivals[i][0]                                   # asleep until the next event
            continue
        t = max(t, worker_free)
        take = pending if batch else pending[:1]                 # batched: every event that is waiting
        length = START + PER_EVENT * len(take)
        if rng.random() < CRASH:                                 # crash: nothing is marked done
            runs.append((t, t + length, False))
            worker_free = t + rng.uniform(0, length) + RESTART
            continue
        for e in take:
            table[e]['status'] = 'done'
            handled[e] = handled.get(e, 0) + 1
            waits.append(t + length - table[e]['at'])
        runs.append((t, t + length, True))
        worker_free = t + length
    return runs if keep_runs else summary(events, handled, set(), runs, 1, 0, waits)
# [/snippet]


def summary(events, handled, lost, runs, overlaps, lost_writes, waits):
    ids = {e[1] for e in events}
    return {'events': len(events), 'distinct': len(ids), 'handled': len(handled), 'lost': len(ids - set(handled)),
            'twice': sum(1 for v in handled.values() if v > 1), 'runs': len(runs), 'crashed': sum(1 for r in runs if not r[2]),
            'most_at_once': overlaps, 'lost_writes': lost_writes,
            'busy_minutes': round(sum(e - s for s, e, _ in runs) / 60), 'median_wait': round(float(np.median(waits)) if waits else 0),
            'slowest_wait': round(float(np.max(waits)) if waits else 0)}


if __name__ == '__main__':
    out = {}
    days = 20
    for name, fn in (('direct', lambda ev, r: direct(ev, r)), ('inbox, one by one', lambda ev, r: inbox(ev, r, batch=False)),
                     ('inbox, batched', lambda ev, r: inbox(ev, r, batch=True))):
        rows = [fn(day_of_events(np.random.default_rng(d)), np.random.default_rng(1000 + d)) for d in range(days)]
        out[name] = {k: round(float(np.mean([r[k] for r in rows])), 1) for k in rows[0]}
        r = out[name]
        print(f"{name:18} events {r['events']:.0f} ({r['distinct']:.0f} distinct)  lost {r['lost']:4.1f}  handled twice {r['twice']:4.1f}  "
              f"runs {r['runs']:5.1f}  most at once {r['most_at_once']:.1f}  memory overwritten {r['lost_writes']:4.1f}  "
              f"busy {r['busy_minutes']:.0f} min  wait median {r['median_wait']:.0f} s, slowest {r['slowest_wait']:.0f} s")
    out['days'] = days
    with open(OUT, 'w') as f:
        json.dump(out, f)
    print('averages over', days, 'simulated days; wrote', os.path.normpath(OUT))
