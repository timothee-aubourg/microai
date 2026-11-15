"""
ch45_timers.py — an agent that wakes itself: follow-ups, schedules, and a scheduler.   (Chapter 45)

    python code/ch45_timers.py       # a few seconds; no model needed

Events from outside are not the only reason to wake. An agent sets its own timers:

    follow-up   a deadline it gives itself once: "if nobody has replied in two hours, look again"
    schedule    something that repeats: "every morning at 9, send the summary"

A scheduler looks at the stored timers every few minutes, and for each one that is due it
writes an event into the inbox of chapter 44. Three questions, each simulated over a week:

    1. how late do timers fire, depending on how often the scheduler looks?
    2. what happens to follow-ups when the worker restarts, if timers are kept in its memory
       instead of in a store of their own?
    3. the scheduler was down for five hours. An hourly schedule missed five times. Fire all five
       when it comes back, or once?
"""
import json, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'timers.json')
WEEK = 7 * 24 * 60                     # minutes


# [snippet:scheduler]
def poll(timers, now, missed='once'):
    """One look by the scheduler. Returns the events to write to the inbox, and updates the timers.
    A follow-up fires once and is done. A schedule fires and moves to its next time in the future."""
    events = []
    for tm in timers:
        if tm['done'] or tm['due'] > now:
            continue
        if tm['every'] is None:                                  # a follow-up
            events.append((tm['name'], tm['due']))
            tm['done'] = True
            continue
        late = 0
        while tm['due'] <= now:                                  # a schedule: how many times was it due?
            late += 1
            if missed == 'all':
                events.append((tm['name'], tm['due']))           # one event per missed time
            tm['due'] += tm['every']
        if missed == 'once':
            events.append((tm['name'], now))                     # one event, whatever was missed
    return events
# [/snippet]


def lateness(every_min, rng):
    """Follow-ups due at random times; the scheduler looks every `every_min` minutes."""
    due = rng.uniform(0, WEEK, 500)
    fired = np.ceil(due / every_min) * every_min
    return float(np.mean(fired - due)), float(np.max(fired - due))


def restarts(rng, keep):
    """A week of follow-ups (30 a day, two minutes to a day ahead) and two restarts a day.
    keep='memory': timers live in the worker and vanish when it restarts. keep='store': they survive."""
    timers, fired, made = [], 0, 0
    restart_at = set(int(x) for x in rng.uniform(0, WEEK, 14))
    for minute in range(WEEK):
        if rng.random() < 30 / (24 * 60):
            timers.append({'name': 'follow-up', 'due': minute + rng.uniform(2, 24 * 60), 'every': None, 'done': False})
            made += 1
        if minute in restart_at and keep == 'memory':
            timers = [t for t in timers if t['done']]            # everything pending was only in the worker
        if minute % 5 == 0:
            fired += len(poll(timers, minute))
    return made, fired, sum(1 for t in timers if not t['done'])


def outage(missed):
    """An hourly schedule; the scheduler is down from hour 50 to hour 55 (it looks every 5 minutes)."""
    timers = [{'name': 'hourly summary', 'due': 0, 'every': 60, 'done': False}]
    sent = []
    for minute in range(0, 4 * 24 * 60, 5):
        if 50 * 60 <= minute < 55 * 60:
            continue                                            # down: nobody looks
        for e in poll(timers, minute, missed):
            sent.append(minute)
    after = [m for m in sent if m == 55 * 60]
    return len(sent), len(after)


if __name__ == '__main__':
    out = {'lateness': [], 'restarts': {}, 'outage': {}}
    rng = np.random.default_rng(45)
    for every in (1, 5, 15, 60):
        mean, worst = lateness(every, rng)
        out['lateness'].append({'every': every, 'mean': round(mean, 1), 'worst': round(worst, 1)})
        print(f'scheduler looks every {every:>2} min: a timer fires {mean:5.1f} min late on average, at worst {worst:5.1f}')
    for keep in ('memory', 'store'):
        made, fired, waiting = restarts(np.random.default_rng(7), keep)
        lost = made - fired - waiting
        out['restarts'][keep] = {'made': made, 'fired': fired, 'still_waiting': waiting, 'lost': lost}
        print(f'timers kept in {keep:6}: {made} follow-ups set in a week, {fired} fired, {waiting} still waiting at the end, {lost} lost')
    for missed in ('all', 'once'):
        total, burst = outage(missed)
        out['outage'][missed] = {'sent': total, 'at_once': burst}
        print(f'after five hours down, missed times fired "{missed}": {burst} summaries at the moment it came back ({total} in four days)')
    with open(OUT, 'w') as f:
        json.dump(out, f)
    print('wrote', os.path.normpath(OUT))
