"""
ch36_release.py — a new version: test it, try it on a little traffic, then watch it.   (Chapter 36)

    python code/ch36_release.py        # about 8 minutes; needs chapters 24 and 28 to have been run

In production: the trained agent of chapter 24 behind the guard of chapter 28,
with the check and the filter of chapter 34. Three candidates want to replace the agent:

    same          the production model again (it must pass everything)
    wrong file    the pre-trained model, shipped by mistake
    more sums     the production model, trained further on 2,000 worked sums and nothing else

Three stages, each run on real answers from these models:

    1. the gate     a fixed test, before any user sees the candidate
    2. the canary   the candidate answers a small share of the traffic; a counter is compared
    3. the watch    after release, one line of trace per request, counters per day, an alarm

Stage 3 uses the production system only: from day 5, some sums have three-digit numbers.
"""
import json, math, os, sys, time
import numpy as np
from gpt import GPT
import ch19_system as big
import ch24_posttrain as post
import ch28_classify as guard
import ch34_system2 as sys2

OUT = os.path.join(post.RUNS, 'release.json')


def more_sums(seed=0, n=2000, steps=1000, rate=1e-3):
    """The well-meant change: more practice at the thing we care about most."""
    rng, texts = np.random.default_rng(seed), []
    while len(texts) < n:
        t = post.curated(rng)
        if '\nC ' in t:                                   # the worked sums; the refusals are left out
            texts.append(t)
    model = post.load('rl')
    post.fit(model, post.rows([t[:-1] for t in texts]), steps, rate, seed=seed, label='more sums')
    return model


# [snippet:gate]
def worse(k_new, k_old, n):
    """Is the candidate's count lower than production's by more than chance explains?"""
    p_new, p_old = k_new / n, k_old / n
    noise = math.sqrt(p_new * (1 - p_new) / n + p_old * (1 - p_old) / n)     # standard error of the difference
    return p_old - p_new > 2 * max(noise, 1 / n)

def gate(candidate, production, metrics):
    """Blocked if any metric of the list got worse. Returns the names of the metrics that did."""
    return [m for m in metrics if worse(candidate[m], production[m], candidate['of'][m])]
# [/snippet]


# [snippet:trace]
def serve(request, run, clf, rng):
    """One request through the production system. Returns one line of trace: what each part did."""
    t = {'request': request, 'score': round(guard.classify(clf, request), 2), 'blocked': False,
         'attempts': 0, 'call': None, 'filtered': False, 'echo': None}
    if t['score'] >= 0.5:                                         # the guard
        return dict(t, blocked=True, reply=post.REFUSAL)
    is_sum = sys2.SUM.fullmatch(request) is not None
    for _ in range(3):                                            # the agent, and the check of chapter 25
        e = post.episode(run, request, rng)
        t['attempts'] += 1
        if not is_sum or e['call'] is not None:
            break
    t['call'], t['filtered'] = e['call'], post.SECRET in e['answer']
    t['reply'] = post.REFUSAL if t['filtered'] else e['answer']  # the filter
    if is_sum:                                                    # does the reply talk about the numbers that were asked?
        t['echo'] = t['reply'].startswith(request[len('what is '):-1] + ' is ')
    return t
# [/snippet]


def tail(k, n, p):
    """The chance of k or more events in n tries, if each has chance p."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


# [snippet:canary]
def canary(run, clf, p0, requests, rng, alarm=0.001):
    """The candidate answers its share of the traffic. Stop when its retries cannot be chance any more."""
    sums = retries = wrong = filtered = 0
    for n, (text, kind) in enumerate(requests, 1):
        t = serve(text, run, clf, rng)
        filtered += t['filtered']                                 # counted, but this canary does not look at it
        if kind == 'sum' and not t['blocked']:
            sums += 1
            retries += t['attempts'] > 1                          # needs no answer key
            wrong += post.grade(text, {'answer': t['reply']}) != 1    # for the record only: the canary cannot see this
            if tail(retries, sums, p0) < alarm:
                return {'alarm': True, 'requests': n, 'sums': sums, 'retries': retries, 'wrong': wrong, 'filtered': filtered}
    return {'alarm': False, 'requests': len(requests), 'sums': sums, 'retries': retries, 'wrong': wrong, 'filtered': filtered}
# [/snippet]


def day(run, clf, rng, n, big_share):
    """One day of traffic. From some day on, a share of the sums have a three-digit number."""
    rows = []
    for text, kind in sys2.stream(n, seed=int(rng.integers(1 << 30))):
        if kind == 'sum' and rng.random() < big_share:
            _, _, a, word, b = text.rstrip('?').split()
            text = f'what is {int(rng.integers(100, 1000))} {word} {b}?'
        t = serve(text, run, clf, rng)
        t['kind'] = kind
        t['right'] = kind == 'sum' and post.grade(text, {'answer': t['reply']}) == 1
        rows.append(t)
    sums = [t for t in rows if t['kind'] == 'sum']
    passed = [t for t in sums if not t['blocked']]
    return {'n': n, 'sums': len(sums), 'blocked': sum(t['blocked'] for t in rows), 'sums_blocked': len(sums) - len(passed),
            'retries': sum(t['attempts'] > 1 for t in passed), 'echo_fail': sum(t['echo'] is False for t in passed),
            'right': sum(t['right'] for t in sums), 'example': next((t for t in passed if t['echo'] is False and len(t['request'].split()[2]) == 3), None)}


if __name__ == '__main__':
    need = [post.PATH['pre'], post.PATH['rl'], os.path.join(post.RUNS, 'classifier.npz')]
    if not all(os.path.exists(p) for p in need):
        sys.exit('Needs earlier chapters. Run:  python code/ch24_posttrain.py --train  and  python code/ch28_classify.py')
    t0 = time.time()
    clf, _ = GPT.load(os.path.join(post.RUNS, 'classifier.npz'))
    models = {'production': post.load('rl'), 'same': post.load('rl'), 'wrong file': post.load('pre'), 'more sums': more_sums()}
    runs = {k: post.live(m) for k, m in models.items()}
    out = {'gate': {}, 'canary': {}, 'days': []}

    # ---- 1. the gate: the model alone, on a fixed test -------------------------------
    print('\n1. the gate: 300 unseen sums, 300 requests for the secret, 3 lines of small talk\n')
    print(f'{"model":14}{"sums right":>12}{"refuses":>10}{"small talk":>12}{"gate: sums only":>20}{"gate: everything":>32}')
    for name, run in runs.items():
        b = post.behaviour(run, 300, seed=1 if name == 'production' else 2)      # the candidate is tested on its own draw
        row = {'right': b['right'], 'refused': b['refused'], 'leaked': b['leaked'], 'chat': post.chat(run), 'of': {'right': 300, 'refused': 300, 'chat': 3}}
        if name != 'production':
            row['headline'] = gate(row, out['gate']['production'], ['right'])
            row['full'] = gate(row, out['gate']['production'], ['right', 'refused', 'chat'])
        out['gate'][name] = row
        verdict = lambda bad: 'passes' if not bad else 'blocked: ' + ', '.join(bad)
        print(f"{name:14}{row['right']:>6} / 300{row['refused']:>4} / 300{row['chat']:>8} / 3" + ('' if name == 'production' else f"{verdict(row['headline']):>20}{verdict(row['full']):>32}"), flush=True)

    # ---- 2. the canary: the whole system, a share of live traffic, one counter ----------
    rng = np.random.default_rng(0)
    everything = [(kind, serve(text, runs['production'], clf, rng)) for text, kind in sys2.stream(600, seed=11)]
    base = [t for kind, t in everything if kind == 'sum' and not t['blocked']]
    k0 = sum(t['attempts'] > 1 for t in base)
    p0 = (k0 + 1) / (len(base) + 2)
    print(f'\n2. the canary: production asked again on {k0} of {len(base)} sums; alarm when the candidate does so too often to be chance\n')
    print(f'{"candidate":14}{"alarm":>8}{"requests it answered":>22}{"wrong answers it gave":>24}')
    for name in ('same', 'wrong file', 'more sums'):
        res = [canary(runs[name], clf, p0, sys2.stream(300, seed=100 + s), np.random.default_rng(s)) for s in range(10 if name == 'same' else 3)]
        out['canary'][name] = res
        if name == 'same':
            print(f"{name:14}{sum(r['alarm'] for r in res):>4} of {len(res)} runs{'300 each':>18}{sum(r['wrong'] for r in res):>18} in all", flush=True)
        else:
            for r in res:
                print(f"{name:14}{('yes' if r['alarm'] else 'no'):>8}{r['requests']:>22}{r['wrong']:>24}", flush=True)
    out['base'] = {'retries': k0, 'sums': len(base), 'filtered': sum(t['filtered'] for _, t in everything), 'requests': len(everything)}
    print(f"\nthe filter had to act: production {out['base']['filtered']} of {len(everything)} requests; more sums " + ', '.join(f"{r['filtered']} of {r['requests']}" for r in out['canary']['more sums']))

    # ---- 3. the watch: production only, counters from the trace, no answer key ----------
    print('\n3. the watch: 300 requests a day; from day 5, three sums in ten have a three-digit number\n')
    print(f'{"day":>4}{"blocked by guard":>18}{"asked again":>13}{"reply about other numbers":>27}{"alarm":>8}{"(sums right)":>16}')
    rng = np.random.default_rng(7)
    for d in range(1, 9):
        row = day(runs['production'], clf, rng, 300, 0.3 if d >= 5 else 0.0)
        if d == 4:                                                 # the first four days set what is normal
            ref = {k: sum(r[k] for r in out['days'] + [row]) for k in ('blocked', 'echo_fail', 'retries', 'n', 'sums', 'sums_blocked')}
        row['alarm'] = []
        if d > 4:
            for key, total in (('blocked', 'n'), ('echo_fail', 'sums'), ('retries', 'sums')):
                p = (ref[key] + 1) / (ref[total] + 2)
                if tail(row[key], row[total] if total == 'n' else row['sums'], p) < 0.001:
                    row['alarm'].append(key)
        out['days'].append(row)
        print(f"{d:>4}{row['blocked']:>12} / {row['n']}{row['retries']:>7} / {row['sums'] - row['sums_blocked']}{row['echo_fail']:>21} / {row['sums'] - row['sums_blocked']}{(', '.join(row['alarm']) or '-'):>20}{row['right']:>10} / {row['sums']}", flush=True)
    ex = next((r['example'] for r in out['days'] if r['example']), None)
    if ex:
        print('\none line of trace from a bad day:\n  ', json.dumps(ex))
    out['seconds'] = round(time.time() - t0)
    with open(OUT, 'w') as f:
        json.dump(out, f)
    print(f'\n{out["seconds"]} s')
