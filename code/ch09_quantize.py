"""
ch09_quantize.py — fewer bits per weight, almost the same model.        (Chapter 9)

    python code/ch09_quantize.py

Round every weight to one of 2^bits levels and measure what it costs in loss.
This is "round to nearest with one scale per block of 32 weights" — the idea
behind the simplest GGUF formats (Q8_0, Q4_0). Real K-quants are smarter, but
the trade-off you see here is the real one.
"""
import os
import numpy as np
from engine import Tensor
from gpt import GPT, load_text

HERE = os.path.dirname(os.path.abspath(__file__))


# [snippet:quantize]
def quantize(w, bits, block=32):
    """Return w rounded to `bits` bits per weight, with one float16 scale per block."""
    flat = w.reshape(-1)
    pad = (-len(flat)) % block
    blocks = np.concatenate([flat, np.zeros(pad, flat.dtype)]).reshape(-1, block)
    if bits == 1:                                    # 1 bit: keep only the sign
        scale = np.abs(blocks).mean(axis=1, keepdims=True).astype(np.float16)
        deq = np.sign(blocks) * scale
    else:
        qmax = 2 ** (bits - 1) - 1                   # e.g. 4 bits -> integers in [-7, 7]
        scale = (np.abs(blocks).max(axis=1, keepdims=True) / qmax).astype(np.float16)
        scale = np.maximum(scale, np.float16(1e-8))
        q = np.clip(np.round(blocks / scale), -qmax, qmax)   # what gets stored: small ints
        deq = q * scale                                      # what the runtime computes with
    return deq.reshape(-1)[:len(flat)].reshape(w.shape).astype(np.float32)
# [/snippet]


def val_loss(model, data, n=16):
    """Loss on n windows spread evenly over the held-out text."""
    T = model.block_size
    starts = np.linspace(0, len(data) - T - 1, n).astype(int)
    x = np.stack([data[s:s + T] for s in starts])
    y = np.stack([data[s + 1:s + T + 1] for s in starts])
    return float(model.forward(x, y)[1].data)


if __name__ == '__main__':
    model, chars = GPT.load(os.path.join(HERE, '..', 'runs', 'micro.npz'))
    text = load_text()
    stoi = {c: i for i, c in enumerate(chars)}
    val = np.array([stoi[c] for c in text[int(0.9 * len(text)):]])
    original = {k: t.data.copy() for k, t in model.p.items()}
    matrices = [k for k, v in original.items() if v.ndim == 2]
    n_w = sum(original[k].size for k in matrices)
    base = val_loss(model, val)
    print(f'{n_w:,} weights in {len(matrices)} matrices.  float32 val loss: {base:.4f}\n')
    print('format              bits/weight   size (KB)   val loss    change')

    def report(name, bpw):
        l = val_loss(model, val)
        print(f'{name:18s}  {bpw:11.2f}  {n_w * bpw / 8 / 1024:10.1f}  {l:9.4f}  {l - base:+8.4f}')
        return l

    report('float32', 32)
    for k in matrices:
        model.p[k] = Tensor(original[k].astype(np.float16))
    report('float16', 16)
    for bits in (8, 6, 5, 4, 3, 2, 1):
        for k in matrices:
            model.p[k] = Tensor(quantize(original[k], bits))
        report(f'{bits}-bit, block 32', bits + 16 / 32)        # + one 16-bit scale per 32 weights
    print()
    for bits in (4, 3):
        for k in matrices:
            model.p[k] = Tensor(quantize(original[k], bits, block=original[k].size))
        report(f'{bits}-bit, one scale', bits)
