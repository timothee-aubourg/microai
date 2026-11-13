"""
ch27_serve.py — one model, many people asking at once.   (Chapter 27)

    python code/ch27_serve.py        # about 2 minutes

    1. batches      one pass of the model on 1, 2, 4 ... 32 texts at once: how long does it take?
    2. big matrix   the same question for a matrix too large for the cache (the regime of full-size models)
    3. a queue      requests arriving at random, served one at a time or all together, using the times of parts 1 and 2
"""
import json, os, time
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')                      # one core, so the numbers mean the same on any machine
import numpy as np
from gpt import GPT

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
SIZES = [1, 2, 4, 8, 16, 32]
LENGTH = 48                                               # tokens each request asks for


# [snippet:batch]
def pass_seconds(model, batch, repeats=7):
    """One pass of the model over `batch` texts of 64 tokens at once. Best of several timings."""
    x = np.random.default_rng(0).integers(0, model.vocab_size, (batch, model.block_size))
    best = 1e9
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.forward(x)
        best = min(best, time.perf_counter() - t0)
    return best
# [/snippet]

def matrix_seconds(w, batch, repeats=3):
    """One huge matrix times `batch` vectors: the weights are read once, whatever the batch."""
    x = np.ones((w.shape[1], batch), dtype=np.float32)
    best = 1e9
    for _ in range(repeats):
        t0 = time.perf_counter()
        w @ x
        best = min(best, time.perf_counter() - t0)
    return best


# [snippet:queue]
def serve(arrivals, step_seconds, together):
    """Requests arrive at the given times; each wants LENGTH tokens. Returns when each one finished.
    together = 1: one request at a time. together = 16: every pass advances up to 16 requests by one token."""
    waiting, active, done, clock = list(range(len(arrivals))), {}, {}, 0.0
    while waiting or active:
        while waiting and arrivals[waiting[0]] <= clock and len(active) < together:
            active[waiting.pop(0)] = LENGTH                    # a free place: the next request joins, even mid-way
        if not active:
            clock = arrivals[waiting[0]]                       # nothing to do: wait for the next arrival
            continue
        clock += step_seconds(len(active))                     # one pass, for everyone who is active
        for r in list(active):
            active[r] -= 1
            if active[r] == 0:
                done[r] = clock
                del active[r]
    return np.array([done[r] - arrivals[r] for r in range(len(arrivals))])
# [/snippet]


if __name__ == '__main__':
    model, _ = GPT.load(os.path.join(RUNS, 'micro.npz'))
    out = {}

    print('1. one pass of the course model (345,504 weights) on several texts at once, one core')
    times = {b: pass_seconds(model, b) for b in SIZES}
    print(f'   {"texts":>6}{"ms per pass":>14}{"tokens per second, all":>25}{"tokens per second, each":>26}')
    for b in SIZES:
        print(f'   {b:>6}{1000 * times[b]:>14.2f}{b / times[b]:>25,.0f}{1 / times[b]:>26,.0f}')
    out['batch'] = [[b, round(1000 * times[b], 3)] for b in SIZES]

    print('\n2. one matrix of 1 GB (16,384 x 16,384 numbers) times several vectors at once')
    w = np.ones((16384, 16384), dtype=np.float32)
    big = {b: matrix_seconds(w, b) for b in (1, 2, 4, 8, 16)}
    for b, t in big.items():
        print(f'   {b:>6} vectors{1000 * t:>10.0f} ms{b / t:>10.1f} results per second')
    out['matrix'] = [[b, round(1000 * t, 1)] for b, t in big.items()]
    del w

    def queue(label, sizes, seconds):
        step = lambda n: float(np.interp(n, sizes, seconds))        # between measured sizes: a straight line
        alone = 1 / (LENGTH * seconds[0])                           # requests per second that one-at-a-time can finish
        print(f'\n3. {label}: requests of {LENGTH} tokens arriving at random. One at a time, {alone:.2f} are finished per second.')
        print(f'   {"arriving per second":>22}{"one at a time: typical, slowest 5%":>38}{"up to 16 together: typical, slowest 5%":>42}')
        rows = []
        for load in (0.5, 0.9, 1.5, 3.0):
            arrivals = np.cumsum(np.random.default_rng(0).exponential(1 / (load * alone), 2000))
            a, b = serve(arrivals, step, 1), serve(arrivals, step, 16)
            rows.append({'load': load, 'rate': round(load * alone, 2), 'one': [round(float(np.median(a)), 2), round(float(np.percentile(a, 95)), 2)],
                         'many': [round(float(np.median(b)), 2), round(float(np.percentile(b, 95)), 2)]})
            print(f"   {rows[-1]['rate']:>22.2f}{rows[-1]['one'][0]:>24.2f} s, {rows[-1]['one'][1]:>8.2f} s{rows[-1]['many'][0]:>28.2f} s, {rows[-1]['many'][1]:>8.2f} s")
        return {'alone': round(alone, 2), 'rows': rows}
    out['queue_small'] = queue('the course model', SIZES, [times[b] for b in SIZES])
    out['queue_large'] = queue('a model the size of the 1 GB matrix', list(big), list(big.values()))
    with open(os.path.join(RUNS, 'serve.json'), 'w') as f:
        json.dump(out, f)
