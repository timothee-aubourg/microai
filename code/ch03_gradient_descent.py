"""
ch03_gradient_descent.py — learning is rolling downhill.               (Chapter 3)

    python code/ch03_gradient_descent.py

Part 1 fits a line with hand-derived gradients.
Part 2 does the same with the autograd engine (no hand derivation).
Part 3 checks the engine against brute-force finite differences on a whole GPT.
"""
import numpy as np
from engine import Tensor
from gpt import GPT

# ---- Part 1: gradient descent by hand ---------------------------------------
# [snippet:by_hand]
xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
ys = 2.0 * xs - 1.0 + np.array([0.3, -0.2, 0.1, -0.3, 0.2, 0.0])   # a noisy line

w, b, lr = 0.0, 0.0, 0.05
for step in range(200):
    err = w * xs + b - ys            # prediction error on every point
    loss = (err ** 2).mean()
    dw = (2 * err * xs).mean()       # dLoss/dw, derived with pen and paper
    db = (2 * err).mean()            # dLoss/db
    w -= lr * dw                     # step against the gradient
    b -= lr * db
print(f'by hand : w={w:.3f} b={b:.3f} loss={loss:.4f}   (truth: w=2, b=-1)')
# [/snippet]

# ---- Part 2: the same, but the engine finds the gradients ----------------------
Tensor.dtype = np.float64
w, b = Tensor(0.0), Tensor(0.0)
for step in range(200):
    err = w * xs + b - ys
    loss = (err * err).mean()
    w.grad = b.grad = None
    loss.backward()
    w.data -= 0.05 * w.grad
    b.data -= 0.05 * b.grad
print(f'autograd: w={float(w.data):.3f} b={float(b.data):.3f} loss={float(loss.data):.4f}')

# ---- Part 3: is backward() right? Nudge a weight, watch the loss ----------------
# [snippet:gradcheck]
rng = np.random.default_rng(0)
model = GPT(vocab_size=11, block_size=8, n_layer=2, n_head=2, d_model=8)
x, y = rng.integers(0, 11, (3, 8)), rng.integers(0, 11, (3, 8))
_, loss = model.forward(x, y)
loss.backward()

worst = 0.0
for name, p in model.p.items():
    i = tuple(rng.integers(0, n) for n in p.shape)   # one random entry per tensor
    old, h = p.data[i], 1e-5
    p.data[i] = old + h; up = float(model.forward(x, y)[1].data)
    p.data[i] = old - h; down = float(model.forward(x, y)[1].data)
    p.data[i] = old
    numeric = (up - down) / (2 * h)                  # slope measured by brute force
    worst = max(worst, abs(numeric - p.grad[i]))
print(f'gradient check on a 2-layer GPT: worst |numeric - backprop| = {worst:.2e}')
assert worst < 1e-6
# [/snippet]
