"""
gpt.py — a complete GPT in ~100 lines of numpy.            (Chapters 4, 5, 6)

The same architecture family as the multi-gigabyte model you will run locally
in chapter 10 — only the numbers (width, depth, vocabulary) are smaller.

    tokens -> embeddings -> [ attention -> MLP ] x n_layer -> logits
"""
import numpy as np
from engine import Tensor, cross_entropy


class GPT:
    def __init__(self, vocab_size, block_size=64, n_layer=3, n_head=4, d_model=96, seed=1337):
        self.vocab_size, self.block_size = vocab_size, block_size
        self.n_layer, self.n_head, self.d_model = n_layer, n_head, d_model
        rng = np.random.default_rng(seed)
        d = d_model
        w = lambda *shape, std=0.02: Tensor(rng.normal(0, std, shape))
        res_std = 0.02 / np.sqrt(2 * n_layer)     # keep the residual stream calm at init
        # [snippet:params]
        p = {}
        p['wte'] = w(vocab_size, d)               # token embedding  (also the output layer)
        p['wpe'] = w(block_size, d)               # position embedding
        for i in range(n_layer):
            p[f'h{i}.ln1.g'] = Tensor(np.ones(d));  p[f'h{i}.ln1.b'] = Tensor(np.zeros(d))
            p[f'h{i}.attn.wq'] = w(d, d)          # what am I looking for?
            p[f'h{i}.attn.wk'] = w(d, d)          # what do I contain?
            p[f'h{i}.attn.wv'] = w(d, d)          # what do I hand over if chosen?
            p[f'h{i}.attn.wo'] = w(d, d, std=res_std)
            p[f'h{i}.ln2.g'] = Tensor(np.ones(d));  p[f'h{i}.ln2.b'] = Tensor(np.zeros(d))
            p[f'h{i}.mlp.w1'] = w(d, 4 * d)       # expand...
            p[f'h{i}.mlp.w2'] = w(4 * d, d, std=res_std)   # ...and project back
        p['ln_f.g'] = Tensor(np.ones(d));  p['ln_f.b'] = Tensor(np.zeros(d))
        self.p = p
        # [/snippet]

    def parameters(self):
        return list(self.p.values())

    def num_params(self):
        return sum(t.data.size for t in self.parameters())

    # [snippet:layernorm]
    @staticmethod
    def layernorm(x, g, b, eps=1e-5):
        """Rescale each token vector to mean 0, variance 1, then let the model re-stretch it."""
        xc = x - x.mean(axis=-1, keepdims=True)
        var = (xc * xc).mean(axis=-1, keepdims=True)
        return xc * (var + eps) ** -0.5 * g + b
    # [/snippet]

    # [snippet:attention]
    def attention(self, x, i, trace=None):
        B, T, d = x.shape
        h, hd = self.n_head, d // self.n_head
        split = lambda t: t.reshape(B, T, h, hd).transpose(0, 2, 1, 3)   # (B, heads, T, hd)
        q = split(x @ self.p[f'h{i}.attn.wq'])
        k = split(x @ self.p[f'h{i}.attn.wk'])
        v = split(x @ self.p[f'h{i}.attn.wv'])
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(hd))     # every query · every key
        future = np.triu(np.ones((T, T), dtype=bool), k=1)               # no peeking ahead
        weights = scores.masked_fill(future, -1e9).softmax(axis=-1)      # each row sums to 1
        if trace is not None:
            trace.setdefault('attn', []).append(weights.data)
        mixed = weights @ v                                              # weighted average of values
        mixed = mixed.transpose(0, 2, 1, 3).reshape(B, T, d)             # glue the heads back
        return mixed @ self.p[f'h{i}.attn.wo']
    # [/snippet]

    # [snippet:forward]
    def forward(self, idx, targets=None, trace=None):
        """idx: (B, T) array of token ids -> logits (B, T, vocab), and the loss if targets given."""
        B, T = idx.shape
        p = self.p
        x = p['wte'][idx] + p['wpe'][np.arange(T)]          # what the token is + where it is
        for i in range(self.n_layer):
            # tokens talk to each other...
            x = x + self.attention(self.layernorm(x, p[f'h{i}.ln1.g'], p[f'h{i}.ln1.b']), i, trace)
            # ...then each token thinks on its own
            hidden = (self.layernorm(x, p[f'h{i}.ln2.g'], p[f'h{i}.ln2.b']) @ p[f'h{i}.mlp.w1']).gelu()
            x = x + hidden @ p[f'h{i}.mlp.w2']
            if trace is not None:
                trace.setdefault('resid', []).append(x.data)
        x = self.layernorm(x, p['ln_f.g'], p['ln_f.b'])
        logits = x @ p['wte'].transpose(1, 0)               # compare with every token embedding
        if targets is None:
            return logits, None
        loss = cross_entropy(logits.reshape(B * T, self.vocab_size), targets.reshape(B * T))
        return logits, loss
    # [/snippet]

    # ---- save / load ---------------------------------------------------------
    def save(self, path, chars):
        np.savez(path, __chars__=np.array(list(chars)),
                 __config__=np.array([self.vocab_size, self.block_size, self.n_layer, self.n_head, self.d_model]),
                 **{k: t.data for k, t in self.p.items()})

    @staticmethod
    def load(path):
        z = np.load(path)
        V, block, L, H, d = [int(v) for v in z['__config__']]
        m = GPT(V, block, L, H, d)
        for k in m.p:
            m.p[k] = Tensor(z[k])
        return m, ''.join(z['__chars__'])


def load_text(path=None):
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'shakespeare.txt')
    with open(path, encoding='utf-8') as f:
        return f.read()
