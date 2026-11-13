"""
ch29_distill.py — a large model teaches a small one.   (Chapter 29)

    python code/ch29_distill.py        # four trainings and their measurements, about 10 minutes

The teacher is the model of chapter 7 (345,504 weights). The student has
16,608. It is trained twice from the same starting weights, on the same
batches, with one difference in what it is told at each position:

    from text      the one character that came next in the text
    from teacher   the teacher's whole list of probabilities for the next character

and that pair is trained twice more with only 5,000 characters of text to learn from.
"""
import json, math, os, time
import numpy as np
from engine import Tensor
from gpt import GPT
import ch26_draft as draft

OUT = os.path.join(draft.RUNS, 'distill.json')


# [snippet:soft]
def soft_loss(logits, target):
    """The loss of chapter 1 against a whole list of probabilities instead of one right answer.
    target: one row of probabilities per position (the teacher's). A row with a single 1 gives the usual loss."""
    z = logits.data - logits.data.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    out = Tensor(-(target * logp).sum() / len(target), (logits,))
    def _backward(g):
        logits._accumulate(g * (np.exp(logp) - target) / len(target))   # "p - y" again, with y a list of probabilities
    out._backward = _backward
    return out
# [/snippet]

def probabilities(model, x):
    z = model.forward(x)[0].data
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

# [snippet:teach]
def train_student(teacher, data, chars, from_teacher, steps=3000, batch=32, lr=3e-3):
    rng = np.random.default_rng(0)                            # the same batches whichever way it is taught
    student = GPT(len(chars), n_layer=1, n_head=2, d_model=32)
    T, params = student.block_size, student.parameters()
    m = [np.zeros_like(p.data) for p in params]
    v = [np.zeros_like(p.data) for p in params]
    for step in range(steps):
        starts = rng.integers(0, len(data) - T - 1, batch)
        x, y = np.stack([data[s:s + T] for s in starts]), np.stack([data[s + 1:s + T + 1] for s in starts])
        if from_teacher:
            logits, _ = student.forward(x)
            target = probabilities(teacher, x)                # what the teacher expects next, at every position
            loss = soft_loss(logits.reshape(-1, len(chars)), target.reshape(-1, len(chars)))
        else:
            _, loss = student.forward(x, y)                   # what the text actually had next
        for p in params:
            p.grad = None
        loss.backward()
        rate = lr * min(1, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for i, p in enumerate(params):
            m[i] = 0.9 * m[i] + 0.1 * p.grad
            v[i] = 0.99 * v[i] + 0.01 * p.grad * p.grad
            p.data -= rate * (m[i] / (1 - 0.9 ** (step + 1))) / (np.sqrt(v[i] / (1 - 0.99 ** (step + 1))) + 1e-8)
    return student
# [/snippet]

def measure(student, teacher, val, windows=200):
    """On unseen text: the student's loss, and how often its first choice is the teacher's."""
    T = student.block_size
    starts = np.linspace(0, len(val) - T - 1, windows).astype(int)
    x, y = np.stack([val[s:s + T] for s in starts]), np.stack([val[s + 1:s + T + 1] for s in starts])
    ps, pt = probabilities(student, x), probabilities(teacher, x)
    loss = float(-np.log(np.take_along_axis(ps, y[..., None], -1)).mean())
    return loss, float((ps.argmax(-1) == pt.argmax(-1)).mean())


if __name__ == '__main__':
    data, val, chars = draft.corpus()
    teacher, _ = GPT.load(os.path.join(draft.RUNS, 'micro.npz'))
    t_loss, _ = measure(teacher, teacher, val)
    print(f'teacher: {sum(p.data.size for p in teacher.parameters()):,} weights, loss on unseen text {t_loss:.3f}\n')
    big = draft.Counted(os.path.join(draft.RUNS, 'micro.npz'))
    starts = np.linspace(0, len(val) - draft.PROMPT - 1, 100).astype(int)
    prompts = [list(val[s:s + draft.PROMPT]) for s in starts]
    rows = []
    print(f'{"text to learn from":22}{"taught by":>12}{"loss on unseen text":>22}{"same first choice as teacher":>31}{"drafts kept (chapter 26)":>27}')
    for label, source in (('all of it (1,003,854)', data), ('5,000 characters', data[:5000])):
        for who in ('text', 'teacher'):
            t0 = time.time()
            student = train_student(teacher, source, chars, who == 'teacher')
            seconds = time.time() - t0
            loss, agree = measure(student, teacher, val)
            path = os.path.join(draft.RUNS, '_student.npz')
            student.save(path, chars)
            small = draft.Counted(path)
            os.remove(path)
            kept = drafted = 0
            for p in prompts:                                   # used as the draft model of chapter 26, drafts of 2
                _, a, b = draft.speculate(big, small, p, draft.LENGTH, 2)
                kept, drafted = kept + a, drafted + b
            rows.append({'data': label, 'by': who, 'loss': round(loss, 3), 'agree': round(agree, 3), 'kept': round(kept / drafted, 3), 'seconds': round(seconds)})
            print(f'{label:22}{who:>12}{loss:>22.3f}{100 * agree:>29.1f} %{100 * kept / drafted:>25.1f} %   ({seconds:.0f}s)', flush=True)
    with open(OUT, 'w') as f:
        json.dump({'teacher_loss': round(t_loss, 3), 'rows': rows, 'chars': int(len(data))}, f)
