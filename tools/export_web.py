"""
export_web.py — pack the trained model for the website.

    python tools/export_web.py

Writes docs/assets/weights.js (float16 weights, base64) and docs/assets/data.js
(training log, embedding map, text excerpts, reference numbers used by the
browser tests). Run after code/train.py and code/ch12_lora.py.
If runs/evals.json exists (written by code/ch14_eval.py --save), it is copied
to docs/assets/evals.js for chapter 14.
"""
import base64, json, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'code'))
from engine import Tensor
from gpt import GPT, load_text

model, chars = GPT.load(os.path.join(ROOT, 'runs', 'micro.npz'))
text = load_text()
stoi = {c: i for i, c in enumerate(chars)}
T = model.block_size

# ---- weights as float16 --------------------------------------------------------
blob, tensors = b'', []
def add(name, arr):
    global blob
    tensors.append({'name': name, 'shape': list(arr.shape), 'offset': len(blob) // 2})
    blob += arr.astype('<f2').tobytes()
for k, t in model.p.items():
    add(k, t.data)
adapter = None
lora_path = os.path.join(ROOT, 'runs', 'lora.npz')
if os.path.exists(lora_path):
    z = np.load(lora_path)
    rank, alpha, trainable, total, before, after = [float(v) for v in z['__meta__']]
    adapter = {'rank': int(rank), 'scale': alpha / rank, 'trainable': int(trainable), 'total': int(total),
               'loss_before': before, 'loss_after': after}
    for k in z.files:
        if not k.startswith('__'):
            add('lora.' + k, z[k])
weights = {'config': {'vocab': model.vocab_size, 'block': T, 'n_layer': model.n_layer, 'n_head': model.n_head,
                      'd_model': model.d_model, 'params': model.num_params()},
           'chars': chars, 'tensors': tensors, 'adapter': adapter, 'b64': base64.b64encode(blob).decode()}
with open(os.path.join(ROOT, 'docs', 'assets', 'weights.js'), 'w') as f:
    f.write('window.MICRO_WEIGHTS = ' + json.dumps(weights) + ';\n')

# ---- reference numbers (float16 weights, exactly what the browser gets) ---------------
for k in model.p:
    model.p[k] = Tensor(model.p[k].data.astype(np.float16))
val_text = text[int(0.9 * len(text)):]
held = val_text[:8 * T + 1]                                   # a contiguous unseen passage (chapter 1)
starts = np.linspace(0, len(val_text) - T - 1, 16).astype(int)  # the same 16 windows as code/ch09_quantize.py
windows = [val_text[s:s + T + 1] for s in starts]
x = np.stack([[stoi[c] for c in w[:-1]] for w in windows])
y = np.stack([[stoi[c] for c in w[1:]] for w in windows])
held_loss = float(model.forward(x, y)[1].data)
prompt = 'First Citizen:\nBefore we proceed'
logits = model.forward(np.array([[stoi[c] for c in prompt]]))[0].data[0, -1]

# ---- embedding map: PCA of the token embedding table ------------------------------
E = model.p['wte'].data.astype(np.float64)
Ec = E - E.mean(0)
U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
xy = Ec @ Vt[:2].T
pca = [[chars[i], round(float(xy[i, 0]), 4), round(float(xy[i, 1]), 4)] for i in range(len(chars))]
explained = float((S[:2] ** 2).sum() / (S ** 2).sum())

with open(os.path.join(ROOT, 'runs', 'log.json')) as f:
    log = json.load(f)
log['train'] = [[s, round(l, 4)] for s, l in log['train'][::2]]
log['val'] = [[s, round(l, 4)] for s, l in log['val']]

sys.path.insert(0, os.path.join(ROOT, 'code'))
from ch11_rag import load_chunks
notes = [{'title': c.splitlines()[0], 'text': ' '.join(c.splitlines()[1:]).strip()} for c in load_chunks()]

data = {'log': log, 'notes': notes, 'pca': pca, 'pca_explained': explained,
        'corpus': text[:60000], 'heldout': held, 'windows': windows, 'heldout_loss': held_loss,
        'parity': {'prompt': prompt, 'logits': [round(float(v), 4) for v in logits]}}
with open(os.path.join(ROOT, 'docs', 'assets', 'data.js'), 'w') as f:
    f.write('window.MICRO_DATA = ' + json.dumps(data) + ';\n')
# ---- evaluation results with their traces (chapter 14) --------------------------------
evals_path = os.path.join(ROOT, 'runs', 'evals.json')
if os.path.exists(evals_path):
    with open(evals_path) as f:
        evals = json.load(f)
    with open(os.path.join(ROOT, 'docs', 'assets', 'evals.js'), 'w') as f:
        f.write('window.MICRO_EVALS = ' + json.dumps(evals) + ';\n')

print(f'weights.js {os.path.getsize(os.path.join(ROOT, "docs/assets/weights.js")) / 1024:.0f} KB, '
      f'data.js {os.path.getsize(os.path.join(ROOT, "docs/assets/data.js")) / 1024:.0f} KB, '
      f'held-out loss {held_loss:.4f}, PCA explains {100 * explained:.0f}%')
