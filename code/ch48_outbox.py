"""
ch48_outbox.py — what an agent working in the background sends to the person, and how often.   (Chapter 48)

    python code/ch48_outbox.py       # a few seconds; no model needed

An always-on agent does not stream its thinking to anyone. A run ends in one of three ways:
it sends a message, it decides that nothing needs sending, or it asks for an approval
(chapter 47). The messages go into an outbox, and a dispatcher delivers them.

What to send is a question about the person's attention, which is limited. This file uses a
small model of it, with every assumption stated:

    - a message is read with probability   trust x 1 / (1 + messages in the last 24 hours / 6)
    - reading a message that mattered raises trust by 0.04; one that did not lowers it by 0.03
    - trust starts at 0.6 and stays between 0.05 and 1

Over four weeks the agent has 2 things a day that matter (a reply arrived, a payment failed)
and 20 that do not much (a task finished as planned, a small update). Four ways to send:

    everything      every item, as it happens
    digest          what matters at once; everything else in one message at the end of the day
    only important  what matters, nothing else
    start quiet     only what matters at first; a daily digest is added once the person has read
                    at least 8 of the last 10 messages, and removed again if that drops below 5

Measured: messages sent, items that mattered and were read, and the trust at the end.
"""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'outbox.json')
DAYS, IMPORTANT, ROUTINE = 28, 2, 20


# [snippet:attention]
def read_it(rng, person, now):
    """Does the person read a message sent at `now`? Trust, and how many messages came in the last day."""
    recent = sum(1 for t in person['sent'] if now - t < 24)
    return rng.random() < person['trust'] / (1 + recent / 6)

def deliver(rng, person, now, matters):
    """The dispatcher sends one message; the person reads it or not, and trust moves."""
    seen = read_it(rng, person, now)
    person['sent'].append(now)
    person['opened'].append(seen)
    if seen:
        person['trust'] = min(1.0, max(0.05, person['trust'] + (0.04 if matters else -0.03)))
    return seen
# [/snippet]


# [snippet:ways]
def month(way, rng):
    person = {'trust': 0.6, 'sent': [], 'opened': []}
    digest_on = way in ('digest', 'everything')
    read_important = sent = routine_read = 0
    for day in range(DAYS):
        items = sorted([(day * 24 + float(rng.uniform(8, 20)), True) for _ in range(IMPORTANT)] +
                       [(day * 24 + float(rng.uniform(8, 20)), False) for _ in range(ROUTINE)])
        pending = 0
        for t, matters in items:
            if matters:
                read_important += deliver(rng, person, t, True); sent += 1
            elif way == 'everything':
                routine_read += deliver(rng, person, t, False); sent += 1
            elif digest_on:
                pending += 1                                      # kept for the end of the day
        if pending and way != 'everything':
            routine_read += pending * deliver(rng, person, day * 24 + 20, False); sent += 1   # one digest
        if way == 'start quiet':                                  # earn attention, and give it back
            last = person['opened'][-10:]
            if len(last) == 10 and sum(last) >= 8:
                digest_on = True
            elif len(last) == 10 and sum(last) < 5:
                digest_on = False
    return {'sent': sent, 'important_read': read_important, 'important': DAYS * IMPORTANT,
            'routine_read': routine_read, 'trust': person['trust']}
# [/snippet]


if __name__ == '__main__':
    out = {}
    for way in ('everything', 'digest', 'only important', 'start quiet'):
        rows = [month(way, np.random.default_rng(s)) for s in range(50)]
        r = {k: round(float(np.mean([x[k] for x in rows])), 2 if k == 'trust' else 1) for k in rows[0]}
        out[way] = r
        print(f"{way:15} sent {r['sent']:6.1f}  things that mattered read {r['important_read']:5.1f} of {r['important']:.0f}  "
              f"other items read {r['routine_read']:6.1f}  trust at the end {r['trust']:.2f}")
    with open(OUT, 'w') as f:
        json.dump(out, f)
    print('averages over 50 simulated months; wrote', os.path.normpath(OUT))
