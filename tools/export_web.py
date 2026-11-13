"""
export_web.py — pack the trained model for the website.

    python tools/export_web.py

Writes docs/assets/weights.js (float16 weights, base64) and docs/assets/data.js
(training log, embedding map, text excerpts, reference numbers used by the
browser tests). Run after code/train.py and code/ch12_lora.py.
Optional inputs, each copied to docs/assets/ when present: runs/evals.json
(ch14_eval.py --save), runs/harness.json (ch15_harness.py --mock --save),
runs/guard.json (ch17_guard.py --save), runs/agent.npz (ch19_system.py --train),
runs/moe.json (ch20_moe.py --train), runs/think_*.npz (ch21_think.py --train),
runs/tiny.bin (ch23_tiny.py --train), and runs/post_*.npz with runs/posttrain.json
(ch24_posttrain.py --train).
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

# ---- recorded sessions for chapters 15 and 17 ------------------------------------------
for name, var in [('harness', 'MICRO_HARNESS'), ('guard', 'MICRO_GUARD'), ('moe', 'MICRO_MOE'), ('selftrain', 'MICRO_SELF'), ('release', 'MICRO_RELEASE')]:
    src = os.path.join(ROOT, 'runs', name + '.json')
    if os.path.exists(src):
        with open(src) as f:
            payload = json.load(f)
        with open(os.path.join(ROOT, 'docs', 'assets', name + '.js'), 'w') as f:
            f.write(f'window.{var} = ' + json.dumps(payload) + ';\n')

# ---- the agent trained in chapter 19, in the same layout as weights.js -----------------
agent_path = os.path.join(ROOT, 'runs', 'agent.npz')
if os.path.exists(agent_path):
    z = np.load(agent_path)
    V, block, L, H, d = [int(v) for v in z['__config__']]
    ablob, atensors = b'', []
    for k in z.files:
        if not k.startswith('__'):
            atensors.append({'name': k, 'shape': list(z[k].shape), 'offset': len(ablob) // 2})
            ablob += z[k].astype('<f2').tobytes()
    agent = {'config': {'vocab': V, 'block': block, 'n_layer': L, 'n_head': H, 'd_model': d, 'params': len(ablob) // 2},
             'chars': ''.join(z['__chars__']), 'tensors': atensors, 'adapter': None,
             'b64': base64.b64encode(ablob).decode()}
    with open(os.path.join(ROOT, 'docs', 'assets', 'agent.js'), 'w') as f:
        f.write('window.MICRO_AGENT = ' + json.dumps(agent) + ';\n')

# ---- the two models of chapter 21 (answer directly, or write the steps first) ----------
def pack(npz_path):
    z = np.load(npz_path)
    V, block, L, H, d = [int(v) for v in z['__config__']]
    blob_, tensors_ = b'', []
    for k in z.files:
        if not k.startswith('__'):
            tensors_.append({'name': k, 'shape': list(z[k].shape), 'offset': len(blob_) // 2})
            blob_ += z[k].astype('<f2').tobytes()
    return {'config': {'vocab': V, 'block': block, 'n_layer': L, 'n_head': H, 'd_model': d, 'params': len(blob_) // 2},
            'chars': ''.join(z['__chars__']), 'tensors': tensors_, 'adapter': None, 'b64': base64.b64encode(blob_).decode()}

think = {name: os.path.join(ROOT, 'runs', f'think_{name}.npz') for name in ('direct', 'steps')}
if all(os.path.exists(p) for p in think.values()):
    with open(os.path.join(ROOT, 'docs', 'assets', 'think.js'), 'w') as f:
        f.write('window.MICRO_THINK = ' + json.dumps({name: pack(p) for name, p in think.items()}) + ';\n')

# ---- chapter 24: the model after each stage of training, and the record of the reinforcement stage ----
post = {name: os.path.join(ROOT, 'runs', f'post_{name}.npz') for name in ('pre', 'sft', 'rl')}
post_facts = os.path.join(ROOT, 'runs', 'posttrain.json')
if all(os.path.exists(p) for p in post.values()) and os.path.exists(post_facts):
    facts = json.load(open(post_facts))
    keep = ('n', 'tool', 'right', 'refused', 'leaked', 'chat')
    with open(os.path.join(ROOT, 'docs', 'assets', 'post.js'), 'w') as f:
        f.write('window.MICRO_POST = ' + json.dumps({'models': {name: pack(p) for name, p in post.items()}, 'curve': facts['rl']['curve'],
                                                     'stats': {name: {k: facts[name][k] for k in keep} for name in post}}) + ';\n')

# ---- chapter 28: the classifier, and the requests it is tested on ----
clf = os.path.join(ROOT, 'runs', 'classifier.npz')
if os.path.exists(clf):
    sys.path.insert(0, os.path.join(ROOT, 'code'))
    import ch28_classify
    with open(os.path.join(ROOT, 'docs', 'assets', 'clf.js'), 'w') as f:
        f.write('window.MICRO_CLF = ' + json.dumps({'model': pack(clf), 'test': ch28_classify.requests('test', seed=1)}) + ';\n')

# ---- chapter 26: the small draft model ----
draft = os.path.join(ROOT, 'runs', 'draft.npz')
if os.path.exists(draft):
    with open(os.path.join(ROOT, 'docs', 'assets', 'draft.js'), 'w') as f:
        f.write('window.MICRO_DRAFT = ' + json.dumps(pack(draft)) + ';\n')

# ---- chapter 31: the agent that reads its first number from a picture ----
eyes = os.path.join(ROOT, 'runs', 'eyes.npz')
if os.path.exists(eyes):
    with open(os.path.join(ROOT, 'docs', 'assets', 'eyes.js'), 'w') as f:
        f.write('window.MICRO_EYES = ' + json.dumps(pack(eyes)) + ';\n')

# ---- chapter 35: the guard after three repairs, and what the attackers achieved ----
hard = os.path.join(ROOT, 'runs', 'classifier_hard.npz')
red = os.path.join(ROOT, 'runs', 'redteam.json')
if os.path.exists(hard) and os.path.exists(red):
    sys.path.insert(0, os.path.join(ROOT, 'code'))
    import ch35_redteam
    with open(os.path.join(ROOT, 'docs', 'assets', 'redteam.js'), 'w') as f:
        f.write('window.MICRO_RED = ' + json.dumps({'hard': pack(hard), 'words': ch35_redteam.WORDS, 'starts': ch35_redteam.STARTS,
                                                    'facts': json.load(open(red))}) + ';\n')

# ---- chapters 37 and 38: the agent that reads its tools from a list, and the catalogue of chapter 38 ----
tools = os.path.join(ROOT, 'runs', 'tools.npz')
if os.path.exists(tools):
    sys.path.insert(0, os.path.join(ROOT, 'code'))
    import ch38_skills
    with open(os.path.join(ROOT, 'docs', 'assets', 'tools.js'), 'w') as f:
        f.write('window.MICRO_TOOLS = ' + json.dumps({'model': pack(tools), 'catalogue': ch38_skills.CATALOGUE, 'room': ch38_skills.ROOM}) + ';\n')

# ---- the packed model of chapter 23: the file itself, byte for byte ----
tiny = os.path.join(ROOT, 'runs', 'tiny.bin')
if os.path.exists(tiny):
    with open(os.path.join(ROOT, 'docs', 'assets', 'tiny.js'), 'w') as f:
        f.write('window.MICRO_TINY = ' + json.dumps({'b64': base64.b64encode(open(tiny, 'rb').read()).decode()}) + ';\n')

print(f'weights.js {os.path.getsize(os.path.join(ROOT, "docs/assets/weights.js")) / 1024:.0f} KB, '
      f'data.js {os.path.getsize(os.path.join(ROOT, "docs/assets/data.js")) / 1024:.0f} KB, '
      f'held-out loss {held_loss:.4f}, PCA explains {100 * explained:.0f}%')
