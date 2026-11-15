"""
ch42_memory.py — memory that lasts from one session to the next: what to keep, and what to show.   (Chapter 42)

    python code/ch42_memory.py      # about 3 minutes; needs  python code/ch39_plan.py --train

A person works with the agent over many sessions. In some they state a fact ("n is 42"),
and later they change it ("n is now 50"). In others they ask a question that uses the
names: "what is n times 3?". The model knows nothing between sessions: whatever it should
remember, the harness has to write down and put back in the context, as lines "M n is 42"
before the question (chapter 39 trained the agent to read them).

Four ways for the harness to keep and show its notes:

    nothing        no notes at all
    every note     every fact ever stated, in order, old values included; cut to fit the window
    latest         one note per name, the newest value replaces the old one
    latest, picked one note per name, and only the notes whose name is in the question

Measured as the number of names grows: right answers, and characters read per question.
"""
import json, os, sys, time
import numpy as np
import ch39_plan as w

OUT = os.path.join(w.RUNS, 'memory_notes.json')


def history(rng, names, sessions=60):
    """A person's sessions: facts stated and changed, and questions that use them."""
    facts, events = {}, []
    for _ in range(sessions):
        if not facts or rng.random() < 0.4:
            name = names[rng.integers(len(names))]
            facts[name] = int(rng.integers(100))
            events.append(('fact', name, facts[name]))      # stated for the first time, or changed
        else:
            known = sorted(facts)
            links = int(rng.integers(1, 3))
            nums, ops, _ = w.chain(rng, links, test=True)
            k = 1                                              # one name per question: the test is of the notes, not of the lookup
            pick = rng.choice(len(known), size=k, replace=False)
            slots = rng.choice(links + 1, size=k, replace=False)
            use = {int(s): known[int(p)] for s, p in zip(slots, pick)}
            for s, n in use.items():
                nums[s] = facts[n]
            values, v = [], nums[0]
            for op, x in zip(ops, nums[1:]):
                v = w.TOOLS[w.WORDS[op]](v, x)
                values.append(v)
            if all(abs(x) < 10000 for x in values):
                events.append(('ask', nums, ops, use, values[-1]))
    return events


# [snippet:keep]
def notes(way, log, question_names, room):
    """The note lines the harness puts before the question. `log` is every fact stated so far, in order."""
    if way == 'nothing':
        return []
    if way == 'every note':
        lines = [f'M {n} is {v}' for n, v in log]               # old values stay, next to the new ones
    else:
        latest = {}
        for n, v in log:
            latest[n] = v                                      # a new value replaces the old one
        lines = [f'M {n} is {v}' for n, v in latest.items()
                 if way == 'latest' or n in question_names]    # 'latest, picked': only what the question names
    while lines and sum(len(l) + 1 for l in lines) > room:
        lines = lines[1:]                                      # does not fit: the oldest go first
    return lines
# [/snippet]


if __name__ == '__main__':
    if not os.path.exists(w.PATH):
        sys.exit('No trained agent yet. Run:  python code/ch39_plan.py --train')
    run, t0 = w.live(), time.time()
    ways = ('nothing', 'every note', 'latest', 'latest, picked')
    rows = []
    print(f'{"names":>6}  ' + '  '.join(f'{x:>22}' for x in ways))
    for n_names in (2, 4, 8, 16):
        rng = np.random.default_rng(n_names)
        names = list(rng.choice(list(w.NAMES), size=n_names, replace=False))
        row = {'names': n_names, 'ways': {x: {'right': 0, 'read': 0} for x in ways}, 'asked': 0}
        for person in range(8):
            log = []
            for e in history(rng, names):
                if e[0] == 'fact':
                    log.append((e[1], e[2]))
                    continue
                _, nums, ops, use, want = e
                q = w.question(nums, ops, use)
                room = w.BLOCK - len(q) - 60                     # what the window leaves after the question and its working
                row['asked'] += 1
                for x in ways:
                    got, calls, read = w.run_steps(run, w.prompt_for(nums, ops, notes(x, log, set(use.values()), room), use))
                    row['ways'][x]['right'] += got == want
                    row['ways'][x]['read'] += read
        for x in ways:
            row['ways'][x]['read'] = round(row['ways'][x]['read'] / row['asked'])
        rows.append(row)
        print(f'{n_names:>6}  ' + '  '.join(f"{row['ways'][x]['right']:>4}/{row['asked']:<4} read {row['ways'][x]['read']:>4}" for x in ways), flush=True)
    with open(OUT, 'w') as f:
        json.dump({'rows': rows, 'seconds': round(time.time() - t0)}, f)
    print(f'{round(time.time() - t0)} s')
