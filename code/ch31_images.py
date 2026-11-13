"""
ch31_images.py — the agent reads a number from a picture.   (Chapter 31)

    python code/ch31_images.py --train     # about 6 minutes
    python code/ch31_images.py             # the measurements

The agent of chapter 19, with one change to what it can be given. A number in
the question may arrive as a small picture: each digit is 5 x 3 pixels. A
picture cannot be looked up in the embedding table, so each digit's 15 pixels
are multiplied by a new matrix that turns them into a vector of the same size
as a token's. From there on, the model is the same model.
"""
import argparse, json, math, os, sys, time
import numpy as np
from engine import Tensor, cross_entropy
from gpt import GPT
from ch13_constrain import moves, START
import ch19_system as big

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, '..', 'runs')
PATH, OUT = os.path.join(RUNS, 'eyes.npz'), os.path.join(RUNS, 'images.json')
CHARS = big.CHARS + ['#']                      # '#' marks a place where a picture goes
STOI = {c: i for i, c in enumerate(CHARS)}
WIDTH = 73
GLYPHS = {d: np.array([int(c) for c in rows], dtype=float) for d, rows in {
    '0': '111101101101111', '1': '010110010010111', '2': '111001111100111', '3': '111001111001111', '4': '101101111001001',
    '5': '111100111001111', '6': '111100111101111', '7': '111001010010010', '8': '111101111101111', '9': '111101111001111'}.items()}


# [snippet:patches]
def picture(number, rng, noise):
    """A number as pixels: one row of 15 pixels per digit. `noise` is the chance that a pixel is flipped."""
    out = np.stack([GLYPHS[d] for d in str(number)])
    flip = rng.random(out.shape) < noise
    return np.where(flip, 1 - out, out)

def encode(text, pictures):
    """Text to token ids, and for each position its 15 pixels (zeros where there is no picture)."""
    ids = np.array([STOI[c] for c in text])
    pixels = np.zeros((len(text), 15))
    pixels[ids == STOI['#']] = np.concatenate(pictures) if pictures else 0
    return ids, pixels
# [/snippet]


class EyesGPT(GPT):
    """The GPT of chapter 6, plus one matrix that turns 15 pixels into a token-sized vector."""
    def __init__(self, **kw):
        super().__init__(len(CHARS), block_size=WIDTH, n_layer=2, n_head=4, d_model=64, **kw)
        self.p['patch.w'] = Tensor(np.random.default_rng(5).normal(0, 0.02, (15, self.d_model)))

    # [snippet:forward]
    def forward(self, idx, targets=None, pixels=None):
        B, T = idx.shape
        p = self.p
        x = p['wte'][idx] + p['wpe'][np.arange(T)]
        if pixels is not None:
            x = x + Tensor(pixels) @ p['patch.w']              # the only new line: pixels become part of the input
        for i in range(self.n_layer):
            x = x + self.attention(self.layernorm(x, p[f'h{i}.ln1.g'], p[f'h{i}.ln1.b']), i)
            x = x + (self.layernorm(x, p[f'h{i}.ln2.g'], p[f'h{i}.ln2.b']) @ p[f'h{i}.mlp.w1']).gelu() @ p[f'h{i}.mlp.w2']
        logits = self.layernorm(x, p['ln_f.g'], p['ln_f.b']) @ p['wte'].transpose(1, 0)
        if targets is None:
            return logits, None
        return logits, cross_entropy(logits.reshape(B * T, self.vocab_size), targets.reshape(B * T))
    # [/snippet]


def conversation(rng, test=False, noise=0.05):
    """One training conversation. The first number of the question is a picture."""
    while True:
        a, b = int(rng.integers(100)), int(rng.integers(100))
        if big.held_out(a, b) == test:
            break
    word = list(big.WORDS)[rng.integers(3)]
    result = big.TOOLS[big.WORDS[word]](a, b)
    text = f'U what is {"#" * len(str(a))} {word} {b}?\nC {big.WORDS[word]} {a} {b}\nR {result}\nA {a} {word} {b} is {result}.\n\n'
    return text, [picture(a, rng, noise)], (a, word, b, result)

def train(steps=4000, batch=32, lr=3e-3):
    rng = np.random.default_rng(0)
    model = EyesGPT()
    ids, pix = [], []
    for _ in range(20000):
        text, pics, _ = conversation(rng)
        i, p = encode(('\n' + text).ljust(WIDTH, '\n'), pics)
        ids.append(i); pix.append(p)
    ids, pix = np.array(ids), np.array(pix)
    params = model.parameters()
    m = [np.zeros_like(p.data) for p in params]
    v = [np.zeros_like(p.data) for p in params]
    t0 = time.time()
    for step in range(steps):
        pick = rng.integers(0, len(ids), batch)
        _, loss = model.forward(ids[pick, :-1], ids[pick, 1:], pix[pick, :-1])
        for p in params:
            p.grad = None
        loss.backward()
        rate = lr * min(1, (step + 1) / 100) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        norm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in params))
        for i, p in enumerate(params):
            g = p.grad * min(1.0, 1.0 / (norm + 1e-6))
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.99 * v[i] + 0.01 * g * g
            p.data -= rate * (m[i] / (1 - 0.9 ** (step + 1))) / (np.sqrt(v[i] / (1 - 0.99 ** (step + 1))) + 1e-8)
        if step % 500 == 0 or step == steps - 1:
            print(f'  step {step:5d}  loss {float(loss.data):.3f}  {time.time() - t0:.0f}s', flush=True)
    np.savez(PATH, __chars__=np.array(CHARS), __config__=np.array([len(CHARS), WIDTH, 2, 4, 64]), **{k: t.data for k, t in model.p.items()})

def load():
    model, z = EyesGPT(), np.load(PATH)
    for k in model.p:
        model.p[k] = Tensor(z[k])
    return model

def answer(model, question, pictures, blind=False):
    """The harness of chapter 19, greedy. Returns (call, final sentence)."""
    ids, pixels = encode('\nU ' + question + '\n', pictures)
    if blind:
        pixels = pixels * 0
    ids, pixels = list(ids), list(pixels)
    def write(fmt=None, limit=40):
        line, state = '', START
        while not line.endswith('\n') and len(line) < limit and len(ids) < WIDTH:
            z = model.forward(np.array([ids]), None, np.array([pixels]))[0].data[0, -1]
            if fmt:
                legal = moves(fmt, state)
                if not legal:
                    break
                mask = np.full(len(CHARS), -np.inf)
                mask[[STOI[c] for c in legal]] = 0
                z = z + mask
            c = CHARS[int(z.argmax())]
            if fmt:
                state = legal[c]
            line += c
            ids.append(STOI[c]); pixels.append(np.zeros(15))
        return line
    call = None
    if write([['C ', 'A ']]) == 'C ':
        call = write(big.CALL).strip()
        name, a, b = call.split()
        for c in f'R {big.TOOLS[name](int(a), int(b))}\nA ':
            ids.append(STOI[c]); pixels.append(np.zeros(15))
    return call, write().strip()

def grade(model, noise, n=300, seed=1, blind=False):
    rng, calls, right = np.random.default_rng(seed), 0, 0
    for _ in range(n):
        text, pics, (a, word, b, result) = conversation(rng, test=True, noise=noise)
        call, final = answer(model, text.split('\n')[0][2:], pics, blind)
        calls += call == f'{big.WORDS[word]} {a} {b}'
        right += call == f'{big.WORDS[word]} {a} {b}' and final == f'{a} {word} {b} is {result}.'
    return calls, right


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    args = ap.parse_args()
    if args.train:
        train()
    if not os.path.exists(PATH):
        sys.exit('No model yet. Run:  python code/ch31_images.py --train')
    model = load()
    print(f'{sum(t.data.size for t in model.p.values()):,} weights, of which {model.p["patch.w"].data.size} turn pixels into a vector\n')
    rows = []
    print(f'{"pixels flipped":>15}{"read the number and called the tool right":>44}{"whole answer right":>21}   (300 unseen sums, first number given as a picture)')
    for noise in (0.0, 0.05, 0.1, 0.2, 0.3):
        calls, right = grade(model, noise)
        rows.append({'noise': noise, 'calls': calls, 'right': right})
        print(f'{100 * noise:>14.0f}%{calls:>38} / 300{right:>15} / 300', flush=True)
    calls, right = grade(model, 0.0, blind=True)
    print(f'{"picture hidden":>15}{calls:>38} / 300{right:>15} / 300')
    with open(OUT, 'w') as f:
        json.dump({'rows': rows, 'blind': {'calls': calls, 'right': right}, 'weights': sum(t.data.size for t in model.p.values())}, f)
