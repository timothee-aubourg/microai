"""
engine.py — a tiny reverse-mode autograd engine on top of numpy.      (Chapter 3)

Everything a GPT needs to learn fits in ~150 lines:
  * a Tensor remembers which operation produced it and from which parents;
  * every operation knows its own local derivative;
  * backward() walks the graph in reverse and applies the chain rule.

No PyTorch. If you understand this file you understand what `loss.backward()` does.
"""
import numpy as np


def _unbroadcast(g, shape):
    """Sum a gradient back down to `shape` (undo numpy broadcasting)."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


# [snippet:tensor]
class Tensor:
    dtype = np.float32          # switch to float64 for gradient checking

    def __init__(self, data, parents=()):
        self.data = np.asarray(data, dtype=Tensor.dtype)
        self.grad = None        # dLoss/dself, filled in by backward()
        self._parents = parents
        self._backward = None   # given dLoss/dself, pushes gradient to the parents

    def _accumulate(self, g):   # gradients from several paths ADD UP
        self.grad = g if self.grad is None else self.grad + g

    def backward(self):
        order, seen = [], set()
        def visit(t):           # topological order: parents before children
            if id(t) not in seen:
                seen.add(id(t))
                for p in t._parents:
                    visit(p)
                order.append(t)
        visit(self)
        self.grad = np.ones_like(self.data)        # dLoss/dLoss = 1
        for t in reversed(order):                  # chain rule, output -> inputs
            if t._backward is not None:
                t._backward(t.grad)            # hand the output's gradient to the op
# [/snippet]

    @property
    def shape(self):
        return self.data.shape

    # ---- arithmetic -----------------------------------------------------
    # [snippet:ops]
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))
        def _backward(g):        # d(a+b)/da = 1, d(a+b)/db = 1
            self._accumulate(_unbroadcast(g, self.shape))
            other._accumulate(_unbroadcast(g, other.shape))
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))
        def _backward(g):        # d(a*b)/da = b, d(a*b)/db = a
            self._accumulate(_unbroadcast(g * other.data, self.shape))
            other._accumulate(_unbroadcast(g * self.data, other.shape))
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))
        def _backward(g):        # the matrix version of the product rule
            self._accumulate(_unbroadcast(g @ np.swapaxes(other.data, -1, -2), self.shape))
            if other.data.ndim == 2:   # weight matrix shared by the whole batch
                a2 = self.data.reshape(-1, self.shape[-1])
                other._accumulate(a2.T @ g.reshape(-1, g.shape[-1]))
            else:
                other._accumulate(_unbroadcast(np.swapaxes(self.data, -1, -2) @ g, other.shape))
        out._backward = _backward
        return out
    # [/snippet]

    def __pow__(self, k):       # k is a plain number
        out = Tensor(self.data ** k, (self,))
        def _backward(g):
            self._accumulate(g * k * self.data ** (k - 1))
        out._backward = _backward
        return out

    def __neg__(self): return self * -1.0
    def __sub__(self, other): return self + (-other if isinstance(other, Tensor) else -np.asarray(other))
    def __truediv__(self, other): return self * (other ** -1 if isinstance(other, Tensor) else 1.0 / other)
    __radd__ = __add__
    __rmul__ = __mul__

    # ---- reductions and shapes -------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))
        def _backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.shape).astype(Tensor.dtype))
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,))
        def _backward(g):
            self._accumulate(g.reshape(self.shape))
        out._backward = _backward
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), (self,))
        def _backward(g):
            self._accumulate(g.transpose(*np.argsort(axes)))
        out._backward = _backward
        return out

    def __getitem__(self, idx):         # used for embedding lookup: table[token_ids]
        out = Tensor(self.data[idx], (self,))
        def _backward(g):
            table = np.zeros_like(self.data)
            np.add.at(table, idx, g)     # a row used twice receives two gradients
            self._accumulate(table)
        out._backward = _backward
        return out

    # ---- nonlinearities ----------------------------------------------------
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))
        def _backward(g):
            self._accumulate(g * (1 - t * t))
        out._backward = _backward
        return out

    def gelu(self):                     # smooth ReLU used in GPTs (tanh approximation)
        x = self.data
        c = float(np.sqrt(2 / np.pi))
        u = c * (x + 0.044715 * x * x * x)
        t = np.tanh(u)
        out = Tensor(0.5 * x * (1 + t), (self,))
        def _backward(g):
            du = c * (1 + 3 * 0.044715 * x * x)
            self._accumulate(g * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du))
        out._backward = _backward
        return out

    def masked_fill(self, mask, value):
        out = Tensor(np.where(mask, value, self.data), (self,))
        def _backward(g):
            self._accumulate(np.where(mask, 0, g))
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, (self,))
        def _backward(g):
            self._accumulate(s * (g - (g * s).sum(axis=axis, keepdims=True)))
        out._backward = _backward
        return out


# [snippet:cross_entropy]
def cross_entropy(logits, targets):
    """Mean of -log p(correct token). logits: (N, vocab) Tensor, targets: (N,) ints."""
    z = logits.data - logits.data.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))     # log-softmax
    n = len(targets)
    out = Tensor(-logp[np.arange(n), targets].mean(), (logits,))
    def _backward(g):
        p = np.exp(logp)                 # the predicted probabilities...
        p[np.arange(n), targets] -= 1    # ...minus the truth: "p - y"
        logits._accumulate(g * p / n)
    out._backward = _backward
    return out
# [/snippet]
