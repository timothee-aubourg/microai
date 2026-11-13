"""
ch18_hardware.py — why this arithmetic wants a particular kind of machine.  (Chapter 18)

    python code/ch18_hardware.py

A language model is a long list of multiplications and additions. This script
measures, on your own processor, the three facts that decide which hardware
runs it well:

  1. the same arithmetic is hundreds of times faster when it is laid out as
     one large, regular block of work than as one step at a time;
  2. regular work can be done for many inputs at once at little extra cost;
  3. writing one token means reading every weight once, so for a large model
     the limit is how fast memory can deliver the weights.

No graphics card is used or needed. The script measures a CPU; the chapter
then says what changes on a GPU and why.
"""
import time
import numpy as np
from ch08_sample import Runner


def best(fn, repeat=5):
    """Fastest of a few runs: the least disturbed measurement."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
    return min(times)


# [snippet:loops]
def multiply_one_at_a_time(A, x):
    """A matrix times a vector, written out: for each row, multiply and add along it."""
    out = []
    for row in A:
        total = 0.0
        for a, b in zip(row, x):
            total += a * b                 # one multiply-add: the unit of work of a neural network
        out.append(total)
    return out
# [/snippet]


if __name__ == '__main__':
    rng = np.random.default_rng(0)

    # ---- 1. one step at a time, or one block of work ---------------------------------
    n = 512
    A, x = rng.standard_normal((n, n)).astype(np.float32), rng.standard_normal(n).astype(np.float32)
    rows, vec = A.tolist(), x.tolist()
    slow = best(lambda: multiply_one_at_a_time(rows, vec), repeat=3)
    fast = best(lambda: A @ x, repeat=50)
    assert np.allclose(multiply_one_at_a_time(rows, vec), A @ x, rtol=1e-3, atol=1e-3)
    work = n * n
    print(f'1. {n} x {n} matrix times a vector: {work:,} multiply-adds')
    print(f'   one at a time (Python loop)   {1e3 * slow:9.2f} ms   {work / slow / 1e6:10.1f} million multiply-adds per second')
    print(f'   as one block  (numpy)         {1e3 * fast:9.4f} ms   {work / fast / 1e6:10.1f} million multiply-adds per second')
    print(f'   same numbers, same result, {slow / fast:,.0f} times faster\n')

    # ---- 2. many inputs at once ------------------------------------------------------
    print('2. the same matrix applied to B inputs at once (time per input)')
    for B in (1, 8, 64, 512):
        X = rng.standard_normal((B, n)).astype(np.float32)
        t = best(lambda: X @ A, repeat=20)
        print(f'   B = {B:4d}   {1e6 * t / B:8.2f} microseconds per input   {B * work / t / 1e9:6.2f} billion multiply-adds per second')
    print()

    # ---- 3. writing a token reads every weight ---------------------------------------
    run = Runner()
    weights = sum(w.size for w in run.w.values())
    ids = [run.stoi[c] for c in 'First Citizen:\nBefore we proceed any further, hear me speak.']
    def write():
        cache = run.new_cache()
        for t in ids:
            run.step(t, cache)
    per_token = best(write) / len(ids)
    print(f'3. our model: {weights:,} weights, {weights * 4 / 1e6:.1f} MB as 32-bit numbers')
    print(f'   one token takes {1e3 * per_token:.2f} ms here: {1 / per_token:,.0f} tokens per second')
    big = np.ones(64_000_000, dtype=np.float32)                      # 256 MB: far too large for the processor's cache
    t = best(lambda: float(big.sum()), repeat=5)
    rate = big.nbytes / t / 1e9
    print(f'   one core of this machine reads memory at about {rate:.0f} GB per second')
    for name, gb in [('our model', weights * 4 / 1e9), ('an 8-billion-weight model at 4.85 bits per weight', 8e9 * 4.85 / 8 / 1e9)]:
        print(f'   {name}: {gb:.3g} GB to read per token, so at most {rate / gb:,.0f} tokens per second at that rate')
    print('   Our model is small enough to sit in the processor\'s cache, so memory is not its limit.')
    print('   For the large one, memory speed is the limit: see the chapter for the same sum on other hardware.')
