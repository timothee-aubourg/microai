"""
ch12_lora.py — teach a frozen model a new habit with a small add-on.    (Chapter 12)

    python code/ch12_lora.py

The base model stays frozen. Next to each weight matrix W we train two thin
matrices A (d x r) and B (r x d) and use  W + A @ B  instead. Here the "new
habit" is deliberately silly so you can see it at a glance: WRITE IN CAPITALS.
"""
import os
import numpy as np
from engine import Tensor
from gpt import GPT, load_text

HERE = os.path.dirname(os.path.abspath(__file__))
RANK, ALPHA, STEPS, LR, BATCH = 4, 8, 600, 3e-3, 32

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    model, chars = GPT.load(os.path.join(HERE, '..', 'runs', 'micro.npz'))
    stoi = {c: i for i, c in enumerate(chars)}
    text = load_text().upper()                           # the target "style"
    data = np.array([stoi[c] for c in text])
    split = int(0.9 * len(data)); train, val = data[:split], data[split:]
    T = model.block_size

    def batch(src, n):
        s = rng.integers(0, len(src) - T - 1, n)
        return np.stack([src[i:i + T] for i in s]), np.stack([src[i + 1:i + T + 1] for i in s])
    val_batches = [batch(val, 64) for _ in range(4)]

    # [snippet:lora]
    frozen = {k: t.data for k, t in model.p.items()}             # never updated
    targets = [k for k in frozen if '.attn.' in k or '.mlp.' in k]
    A = {k: Tensor(rng.normal(0, 0.02, (frozen[k].shape[0], RANK))) for k in targets}
    B = {k: Tensor(np.zeros((RANK, frozen[k].shape[1]))) for k in targets}   # zero: start as a no-op

    def apply_adapter():
        for k in targets:                                        # W_eff = W + (alpha/r) * A @ B
            model.p[k] = Tensor(frozen[k]) + (A[k] @ B[k]) * (ALPHA / RANK)
    # [/snippet]

    def evaluate():
        apply_adapter()
        return float(np.mean([model.forward(x, y)[1].data for x, y in val_batches]))

    trainable = sum(t.data.size for t in list(A.values()) + list(B.values()))
    total = sum(v.size for v in frozen.values())
    before = evaluate()
    print(f'base model: {total:,} frozen parameters.  adapter: {trainable:,} trainable ({100 * trainable / total:.1f}%)')
    print(f'loss on CAPITALS text before: {before:.3f}')

    params = list(A.values()) + list(B.values())
    m = [np.zeros_like(p.data) for p in params]; v = [np.zeros_like(p.data) for p in params]
    for step in range(STEPS):
        apply_adapter()
        x, y = batch(train, BATCH)
        _, loss = model.forward(x, y)
        for p in params:
            p.grad = None
        loss.backward()                                          # gradients reach A and B through W_eff
        for i, p in enumerate(params):                           # Adam, on the adapter only
            m[i] = 0.9 * m[i] + 0.1 * p.grad
            v[i] = 0.99 * v[i] + 0.01 * p.grad ** 2
            p.data -= LR * (m[i] / (1 - 0.9 ** (step + 1))) / (np.sqrt(v[i] / (1 - 0.99 ** (step + 1))) + 1e-8)
        if step % 100 == 0:
            print(f'  step {step:4d}  loss {float(loss.data):.3f}', flush=True)
    after = evaluate()
    print(f'loss on CAPITALS text after : {after:.3f}')

    # Merging: fold the adapter into W. Same outputs, zero extra cost at inference.
    x, y = val_batches[0]
    with_adapter = model.forward(x)[0].data
    for k in targets:
        model.p[k] = Tensor(frozen[k] + (A[k].data @ B[k].data) * (ALPHA / RANK))
    merged = model.forward(x)[0].data
    print(f'merged vs adapter outputs differ by at most {np.abs(with_adapter - merged).max():.1e}')

    def sample(n=200, seed=1):
        r = np.random.default_rng(seed); idx = [stoi['\n']]
        for _ in range(n):
            z = model.forward(np.array([idx[-T:]]))[0].data[0, -1] / 0.8
            p = np.exp(z - z.max()); p /= p.sum()
            idx.append(int(r.choice(len(p), p=p)))
        return ''.join(chars[i] for i in idx[1:])
    print('--- sample with the adapter merged in ---\n' + sample())

    np.savez(os.path.join(HERE, '..', 'runs', 'lora.npz'),
             __meta__=np.array([RANK, ALPHA, trainable, total, before, after]),
             **{f'{k}.A': A[k].data for k in targets}, **{f'{k}.B': B[k].data for k in targets})
