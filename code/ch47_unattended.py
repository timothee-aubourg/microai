"""
ch47_unattended.py — permissions when nobody is watching.   (Chapter 47)

    python code/ch47_unattended.py   # a few seconds; no model needed

An always-on agent acts with the person's access while the person is elsewhere. Every
action it asks for is allowed, blocked, or held until the person approves. Two simulations:

    1. Rules for a week of actions. Most actions are harmless reads; a few send messages,
       delete files or pay. A small share of all actions are mistakes (the agent misread the
       task, or a planted instruction steered it, chapters 17 and 43). Four policies:
           allow all       everything runs
           hold all        everything waits for the person
           by kind         reads and own writes run, sending and paying are held, deleting is blocked
           by kind + check as "by kind", but a checker (a classifier, chapter 28) looks at each
                           send and lets it run unless it looks wrong; only flagged ones are held
       Measured: approvals asked of the person, mistakes that ran, and of those, the ones that
       cannot be undone.

    2. Waiting for an answer. When an action is held, the run can wait for the person, keeping
       the worker busy, or end at once and come back when the answer arrives as a new event in
       the inbox (a deferred call). The events of chapter 44, with one event in twenty needing an
       approval that comes after 20 minutes on average.
"""
import json, os
import numpy as np
import ch44_inbox as ib

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'unattended.json')

# kind: share of actions, can it be undone, chance that an action of this kind is a mistake
KINDS = {'read': (0.70, True, 0.02), 'write': (0.15, True, 0.03), 'send': (0.10, False, 0.04),
         'delete': (0.03, False, 0.05), 'pay': (0.02, False, 0.05)}
CHECK_CATCH, CHECK_FALSE = 0.90, 0.05      # the checker flags 90% of wrong sends and 5% of right ones

# [snippet:policy]
def decide(policy, kind, flagged):
    """allow, block or hold, for one action."""
    if policy == 'allow all':
        return 'allow'
    if policy == 'hold all':
        return 'hold'
    rule = {'read': 'allow', 'write': 'allow', 'send': 'hold', 'delete': 'block', 'pay': 'hold'}[kind]
    if policy == 'by kind + check' and kind == 'send':
        return 'hold' if flagged else 'allow'                 # the checker settles the ordinary sends
    return rule
# [/snippet]


def week(policy, rng, per_day=200):
    asked = ran_wrong = lost_for_good = blocked_right = 0
    kinds = list(KINDS)
    p = [KINDS[k][0] for k in kinds]
    for _ in range(7 * per_day):
        kind = kinds[rng.choice(len(kinds), p=p)]
        undo, wrong_rate = KINDS[kind][1], KINDS[kind][2]
        wrong = rng.random() < wrong_rate
        flagged = rng.random() < (CHECK_CATCH if wrong else CHECK_FALSE)
        d = decide(policy, kind, flagged)
        if d == 'hold':
            asked += 1                                        # the person stops a wrong one, approves a right one
        elif d == 'allow' and wrong:
            ran_wrong += 1
            lost_for_good += not undo
        elif d == 'block' and not wrong:
            blocked_right += 1
    return {'asked': asked, 'ran_wrong': ran_wrong, 'lost_for_good': lost_for_good, 'blocked_right': blocked_right}


# [snippet:defer]
def approvals(events, rng, deferred, need=0.05, answer_mean=20 * 60):
    """One worker, one event per run (chapter 44). Some runs need an approval. Waiting: the worker
    sits until the person answers. Deferred: the run ends at once; the answer comes back later as a
    new event, which then takes one more short run."""
    queue = [(t, 'event') for t, _, _ in events]
    queue.sort()
    free, waits, idle = 0.0, [], 0.0
    while queue:
        t, kind = queue.pop(0)
        start = max(t, free)
        length = ib.START + ib.PER_EVENT
        if kind == 'event' and rng.random() < need:
            answer = float(rng.exponential(answer_mean))
            if deferred:
                queue.append((start + length + answer, 'answer'))   # comes back as an event
                queue.sort()
            else:
                length += answer                                  # the worker waits, doing nothing else
                idle += answer
        free = start + length
        waits.append(start - t)                                   # time this event sat in the inbox
    return {'median_wait': float(np.median(waits)), 'p95_wait': float(np.percentile(waits, 95)),
            'slowest_wait': float(np.max(waits)), 'idle_minutes': idle / 60}
# [/snippet]


if __name__ == '__main__':
    out = {'kinds': {k: list(v) for k, v in KINDS.items()}, 'policies': {}, 'approvals': {}}
    for policy in ('allow all', 'hold all', 'by kind', 'by kind + check'):
        rows = [week(policy, np.random.default_rng(s)) for s in range(20)]
        r = {k: round(float(np.mean([x[k] for x in rows])), 1) for k in rows[0]}
        out['policies'][policy] = r
        print(f"{policy:16} asks the person {r['asked']:6.1f} times a week; mistakes that ran {r['ran_wrong']:5.1f}, "
              f"of which cannot be undone {r['lost_for_good']:4.1f}; right actions blocked {r['blocked_right']:4.1f}")
    for deferred in (False, True):
        rows = [approvals(ib.day_of_events(np.random.default_rng(d)), np.random.default_rng(500 + d), deferred) for d in range(20)]
        r = {k: round(float(np.mean([x[k] for x in rows])), 1) for k in rows[0]}
        out['approvals']['deferred' if deferred else 'waiting'] = r
        print(f"{'deferred' if deferred else 'waiting':9} other events wait: median {r['median_wait']:.0f} s, 95% under {r['p95_wait']:.0f} s, "
              f"slowest {r['slowest_wait'] / 60:.0f} min; worker idle {r['idle_minutes']:.0f} min a day")
    with open(OUT, 'w') as f:
        json.dump(out, f)
    print('wrote', os.path.normpath(OUT))
