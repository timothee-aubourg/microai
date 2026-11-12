"""
train.py — teach the GPT to write, one gradient step at a time.        (Chapter 7)

    python code/train.py               # ~10-20 min on a laptop CPU, numpy only
    python code/train.py --steps 300   # a quick taste

Writes runs/micro.npz (the weights) and runs/log.json (loss curve + samples).
"""
import argparse, json, os, time
import numpy as np
from gpt import GPT, load_text

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=5000)
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs'))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(0)

# ---- data: the text becomes one long array of integers -----------------------
text = load_text()
chars = sorted(set(text))                       # our "tokenizer": one id per character
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int64)
split = int(0.9 * len(data))
train, val = data[:split], data[split:]

model = GPT(vocab_size=len(chars))
T = model.block_size
print(f'{len(text):,} characters, vocab {len(chars)}, model {model.num_params():,} parameters')

# [snippet:batch]
def get_batch(source, batch_size):
    """Random windows of text. The target is the same window shifted by one character."""
    starts = rng.integers(0, len(source) - T - 1, batch_size)
    x = np.stack([source[s:s + T] for s in starts])
    y = np.stack([source[s + 1:s + T + 1] for s in starts])
    return x, y
# [/snippet]

val_batches = [get_batch(val, 64) for _ in range(4)]     # fixed, so the curve is comparable
def val_loss():
    return float(np.mean([model.forward(x, y)[1].data for x, y in val_batches]))

def sample(n=160, temperature=0.8, seed=0):
    r = np.random.default_rng(seed)
    idx = [stoi['\n']]
    for _ in range(n):
        logits, _ = model.forward(np.array([idx[-T:]]))
        z = logits.data[0, -1] / temperature
        p = np.exp(z - z.max()); p /= p.sum()
        idx.append(int(r.choice(len(p), p=p)))
    return ''.join(chars[i] for i in idx[1:])

# ---- the optimiser: Adam = gradient descent with per-parameter step sizes ---------
params = model.parameters()
m = [np.zeros_like(p.data) for p in params]
v = [np.zeros_like(p.data) for p in params]
b1, b2, eps, warmup = 0.9, 0.99, 1e-8, 100

def lr_at(step):                                 # warm up, then cosine decay
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, args.steps - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))

log = {'config': dict(vocab=len(chars), block=T, n_layer=model.n_layer, n_head=model.n_head,
                      d_model=model.d_model, params=model.num_params(), steps=args.steps, batch=args.batch),
       'train': [], 'val': [], 'samples': []}
sample_at = {0, 10, 30, 100, 300, 1000, 2000, 3000, args.steps}
t0, smooth = time.time(), None

# [snippet:loop]
for step in range(args.steps + 1):
    if step % 250 == 0 or step == args.steps:
        log['val'].append([step, val_loss()])
    if step in sample_at:
        log['samples'].append([step, sample()])
    if step == args.steps:
        break

    x, y = get_batch(train, args.batch)
    _, loss = model.forward(x, y)          # 1. predict, and measure the surprise
    for p in params:
        p.grad = None
    loss.backward()                        # 2. which way should every weight move?

    gnorm = float(np.sqrt(sum(float((p.grad ** 2).sum()) for p in params)))
    clip = min(1.0, 1.0 / (gnorm + 1e-6))  # never take a wild step
    lr = float(lr_at(step))
    for i, p in enumerate(params):         # 3. move each weight a little (Adam)
        g = p.grad * clip
        m[i] = b1 * m[i] + (1 - b1) * g
        v[i] = b2 * v[i] + (1 - b2) * g * g
        mhat, vhat = m[i] / (1 - b1 ** (step + 1)), v[i] / (1 - b2 ** (step + 1))
        p.data -= lr * mhat / (np.sqrt(vhat) + eps)
# [/snippet]

    l = float(loss.data)
    smooth = l if smooth is None else 0.95 * smooth + 0.05 * l
    if step % 10 == 0:
        log['train'].append([step, l])
    if step % 100 == 0:
        print(f'step {step:5d}  loss {l:.3f}  (smoothed {smooth:.3f})  lr {lr:.1e}  {time.time() - t0:.0f}s', flush=True)

log['seconds'] = time.time() - t0
model.save(os.path.join(args.out, 'micro.npz'), chars)
with open(os.path.join(args.out, 'log.json'), 'w') as f:
    json.dump(log, f)
print(f"final val loss {log['val'][-1][1]:.3f} in {log['seconds']:.0f}s")
print('--- sample ---\n' + sample(400, seed=1))
