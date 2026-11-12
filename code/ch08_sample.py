"""
ch08_sample.py — from probabilities to text, fast.                      (Chapter 8)

    python code/ch08_sample.py --prompt "ROMEO:" --temperature 0.8 --n 400

Inference only: no autograd, just numpy. Two things live here:
  1. sampling  — temperature, top-k, top-p: how one token is drawn from the logits;
  2. KV cache  — why generating token n does not have to redo tokens 1..n-1.
"""
import argparse, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# [snippet:sample]
def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0, rng=np.random.default_rng()):
    if temperature == 0:
        return int(np.argmax(logits))                 # greedy: always the favourite
    z = logits / temperature                          # <1 sharpens, >1 flattens
    if top_k:
        z = np.where(z < np.sort(z)[-top_k], -np.inf, z)   # keep the k best
    p = np.exp(z - z.max()); p /= p.sum()             # softmax
    if top_p < 1.0:                                   # nucleus: smallest set worth top_p
        order = np.argsort(-p)
        before = np.cumsum(p[order]) - p[order]       # probability mass ranked ahead of each token
        p[order[before >= top_p]] = 0
        p /= p.sum()
    return int(rng.choice(len(p), p=p))
# [/snippet]


class Runner:
    """The GPT forward pass again, weights only, with an optional key/value cache."""

    def __init__(self, path=os.path.join(HERE, '..', 'runs', 'micro.npz')):
        z = np.load(path)
        self.vocab, self.block, self.L, self.H, self.d = [int(v) for v in z['__config__']]
        self.chars = ''.join(z['__chars__'])
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.w = {k: z[k].astype(np.float32) for k in z.files if not k.startswith('__')}

    @staticmethod
    def ln(x, g, b):
        xc = x - x.mean(-1, keepdims=True)
        return xc / np.sqrt((xc * xc).mean(-1, keepdims=True) + 1e-5) * g + b

    def full(self, ids):
        """Recompute everything for the whole sequence. Returns logits (T, vocab)."""
        w, T, H, hd = self.w, len(ids), self.H, self.d // self.H
        x = w['wte'][ids] + w['wpe'][:T]
        for i in range(self.L):
            a = self.ln(x, w[f'h{i}.ln1.g'], w[f'h{i}.ln1.b'])
            split = lambda m: (a @ m).reshape(T, H, hd).transpose(1, 0, 2)
            q, k, v = split(w[f'h{i}.attn.wq']), split(w[f'h{i}.attn.wk']), split(w[f'h{i}.attn.wv'])
            s = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
            s = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -1e9, s)
            s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
            x = x + (s @ v).transpose(1, 0, 2).reshape(T, self.d) @ w[f'h{i}.attn.wo']
            h = self.ln(x, w[f'h{i}.ln2.g'], w[f'h{i}.ln2.b']) @ w[f'h{i}.mlp.w1']
            h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h * h * h)))
            x = x + h @ w[f'h{i}.mlp.w2']
        return self.ln(x, w['ln_f.g'], w['ln_f.b']) @ w['wte'].T

    # [snippet:kvcache]
    def step(self, token, cache):
        """Process ONE new token. `cache` holds the keys and values of all earlier tokens."""
        w, H, hd = self.w, self.H, self.d // self.H
        pos = cache['pos']
        x = w['wte'][token] + w['wpe'][pos]
        for i in range(self.L):
            a = self.ln(x, w[f'h{i}.ln1.g'], w[f'h{i}.ln1.b'])
            q = (a @ w[f'h{i}.attn.wq']).reshape(H, hd)            # only the NEW token asks a question
            cache['k'][i].append((a @ w[f'h{i}.attn.wk']).reshape(H, hd))
            cache['v'][i].append((a @ w[f'h{i}.attn.wv']).reshape(H, hd))
            K = np.stack(cache['k'][i], axis=1)                    # (H, pos+1, hd): old keys are reused
            Vv = np.stack(cache['v'][i], axis=1)
            s = np.einsum('hd,htd->ht', q, K) / np.sqrt(hd)        # one row of scores, not a T x T table
            s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
            x = x + np.einsum('ht,htd->hd', s, Vv).reshape(self.d) @ w[f'h{i}.attn.wo']
            h = self.ln(x, w[f'h{i}.ln2.g'], w[f'h{i}.ln2.b']) @ w[f'h{i}.mlp.w1']
            h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h * h * h)))
            x = x + h @ w[f'h{i}.mlp.w2']
        cache['pos'] += 1
        return self.ln(x, w['ln_f.g'], w['ln_f.b']) @ w['wte'].T
    # [/snippet]

    def new_cache(self):
        return {'pos': 0, 'k': [[] for _ in range(self.L)], 'v': [[] for _ in range(self.L)]}

    # [snippet:generate]
    def generate(self, prompt, n, on_token=None, **sampling):
        ids = [self.stoi[c] for c in prompt]
        cache = self.new_cache()
        for t in ids[-self.block:]:                    # "prompt processing": fill the cache
            logits = self.step(t, cache)
        for _ in range(n):
            nxt = sample_next(logits, **sampling)      # draw one token...
            ids.append(nxt)
            if on_token:
                on_token(self.chars[nxt])
            if cache['pos'] == self.block:             # context window full: keep the recent half
                cache = self.new_cache()
                for t in ids[-self.block // 2:-1]:
                    self.step(t, cache)
            logits = self.step(nxt, cache)             # ...and feed it back in
        return ''.join(self.chars[i] for i in ids)
    # [/snippet]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default='\n')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top_k', type=int, default=0)
    ap.add_argument('--top_p', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    run = Runner()

    # 1. the cache changes the cost, not the answer
    ids = [run.stoi[c] for c in 'First Citizen:\nBefore we proceed any further, hear me speak.'[:run.block]]
    full = run.full(ids)
    cache = run.new_cache()
    cached = np.stack([run.step(t, cache) for t in ids])
    print(f'max |full - cached| over {len(ids)} positions: {np.abs(full - cached).max():.1e}')
    assert np.abs(full - cached).max() < 1e-3

    t0 = time.time()
    for n in range(1, len(ids) + 1):
        run.full(ids[:n])                       # naive generation: redo the whole prefix each time
    t_naive = time.time() - t0
    t0 = time.time(); cache = run.new_cache()
    for t in ids:
        run.step(t, cache)
    t_cached = time.time() - t0
    print(f'{len(ids)} tokens: recompute-everything {t_naive * 1000:.0f} ms, with KV cache {t_cached * 1000:.0f} ms\n')

    # 2. write something
    rng = np.random.default_rng(args.seed)
    print(args.prompt, end='', flush=True)
    run.generate(args.prompt, args.n, on_token=lambda c: print(c, end='', flush=True),
                 temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, rng=rng)
    print()
