"""
ch40_recover.py — when a step fails: stop, retry, tell the model, and know when to give up.   (Chapter 40)

    python code/ch40_recover.py      # about 4 minutes; needs  python code/ch39_plan.py --train

The agent of chapter 39 answers chains of two or three links, one call at a time. Its tool
is now unreliable: each call fails with probability 0.2 and returns an error instead of a
number. Four ways for the harness to handle a failed call:

    stop          give up on the whole question at the first error
    retry quietly the harness tries the call again, up to three times, and the model never knows
    tell, clearly the error goes into the context as "R error busy", the wording seen in training
    tell, vaguely the error goes into the context as "R failed", a wording the model never saw

Then the tool goes down for good, and every call fails. Without a limit the agent asks again
and again. A loop guard stops it when it sends the same call three times in a row.

Last, side effects: each step writes its result to a ledger. When a job fails half-way, the
ledger keeps half a job, unless the harness writes only at the end, or undoes what it wrote.
"""
import json, os, sys, time
import numpy as np
import ch39_plan as w

OUT = os.path.join(w.RUNS, 'recover.json')


class GiveUp(Exception):
    pass


# [snippet:ways]
def tool_for(way, rng, fail=0.2):
    """What the harness does when the model sends a call, for each way of handling a failure."""
    def call(name, a, b):
        tries = 3 if way == 'retry quietly' else 1
        for _ in range(tries):
            if rng.random() >= fail:
                return str(w.TOOLS[name](a, b))               # the call worked
        if way == 'stop':
            raise GiveUp()                                    # the whole question fails
        if way == 'retry quietly':
            raise GiveUp()                                    # three failures in a row: give up too
        return w.BUSY if way == 'tell, clearly' else 'failed' # the model reads the error and decides
    return call
# [/snippet]

def attempt(run, nums, ops, call, max_calls=8):
    """One question. Returns the answer (or None), the model passes it took, and the calls it wrote."""
    sent = []
    def counted(name, a, b):
        sent.append((name, a, b))
        return call(name, a, b)
    try:
        got, calls, read = w.run_steps(run, w.prompt_for(nums, ops), max_calls=max_calls, on_result=counted)
    except GiveUp:
        return None, len(sent), len(sent)                     # every call written was a pass of the model
    return got, len(calls) + 1, len(calls)


# [snippet:guard]
def guarded(call, repeats=3):
    """A loop guard: stop when the model sends the same call `repeats` times in a row."""
    last = []
    def wrapped(name, a, b):
        last.append((name, a, b))
        if len(last) >= repeats and len(set(last[-repeats:])) == 1:
            raise GiveUp()                                    # it is going round in circles
        return call(name, a, b)
    return wrapped
# [/snippet]


# [snippet:ledger]
def job_with_ledger(run, nums, ops, values, rng, fail, mode):
    """Each step of the job writes its result to a ledger. Returns what the ledger holds afterwards.
    'as you go' writes at once; 'at the end' keeps the writes aside and adds them only if the job finished;
    'undo' writes at once and removes its own writes if the job fails."""
    ledger, pending = [], []
    def call(name, a, b):
        if rng.random() < fail:
            raise GiveUp()                                    # this step fails, and so does the job
        r = w.TOOLS[name](a, b)
        (ledger if mode != 'at the end' else pending).append(r)
        return str(r)
    start = len(ledger)
    try:
        got, _, _ = w.run_steps(run, w.prompt_for(nums, ops), on_result=call)
        ok = got == values[-1]
    except GiveUp:
        ok = False
    if mode == 'at the end' and ok:
        ledger += pending                                     # commit: all or nothing
    if mode == 'undo' and not ok:
        del ledger[start:]                                    # roll back what this job wrote
    return ledger, ok
# [/snippet]


if __name__ == '__main__':
    if not os.path.exists(w.PATH):
        sys.exit('No trained agent yet. Run:  python code/ch39_plan.py --train')
    run, t0, out = w.live(), time.time(), {}
    rng = np.random.default_rng(1)
    qs = [w.chain(rng, 2 + i % 2, test=True) for i in range(200)]

    base = sum(w.run_steps(run, w.prompt_for(n, o))[0] == v[-1] for n, o, v in qs)
    print(f'no failures at all: {base} right of {len(qs)}')
    out['base'] = base
    out['ways'] = {}
    for way in ('stop', 'retry quietly', 'tell, clearly', 'tell, vaguely'):
        r = np.random.default_rng(2)
        right = passes = 0
        for nums, ops, values in qs:
            got, p, _ = attempt(run, nums, ops, tool_for(way, r))
            right += got == values[-1]
            passes += p
        out['ways'][way] = {'right': int(right), 'passes': round(passes / len(qs), 2)}
        print(f'{way:15} right {right:>3} of {len(qs)}   model passes per question {passes / len(qs):.2f}', flush=True)

    # the tool is down for good: how many passes before the agent stops?
    down = {}
    for name, guard, limit in (('no limit', False, 40), ('step limit 8', False, 8), ('loop guard', True, 40)):
        spent = []
        for nums, ops, values in qs[:50]:
            call = tool_for('tell, clearly', np.random.default_rng(0), fail=1.0)
            got, p, written = attempt(run, nums, ops, guarded(call) if guard else call, max_calls=limit)
            spent.append(written)                                  # calls the model wrote before the harness stopped
        down[name] = {'mean': round(float(np.mean(spent)), 1), 'max': int(max(spent))}
        print(f'tool down, {name:13}: {np.mean(spent):.1f} calls written on average before it stops, at most {max(spent)}', flush=True)
    out['down'] = down

    # side effects: 200 jobs, each step failing with probability 0.1
    ledgers = {}
    for mode in ('as you go', 'at the end', 'undo'):
        r = np.random.default_rng(3)
        finished = half = 0
        for nums, ops, values in qs:
            ledger, ok = job_with_ledger(run, nums, ops, values, r, 0.1, mode)
            finished += ok
            half += (not ok) and len(ledger) > 0                 # a failed job that left writes behind
        ledgers[mode] = {'finished': int(finished), 'half': int(half)}
        print(f'ledger, {mode:10}: {finished} jobs finished, {half} failed jobs left writes behind', flush=True)
    out['ledger'] = ledgers
    out['n'] = len(qs)
    out['seconds'] = round(time.time() - t0)
    with open(OUT, 'w') as f:
        json.dump(out, f)
    print(f"{out['seconds']} s")
