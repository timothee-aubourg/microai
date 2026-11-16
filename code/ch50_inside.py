"""
ch50_inside.py — reading and changing what a model computes.          (Chapter 50)

The agent of chapter 19 reads "what is 12 times 30?" and writes "mul 12 30".
Somewhere between the two, its 108,032 weights decide on "mul". This script finds
where, with three experiments that touch no weight:

    swap    replace ONE activation with the one from another question, and see
            whether the answer changes. Done for every place and every layer.
    test    turn what the swap suggests into predictions about new inputs,
            and check them: one holds, one does not.
    steer   add a direction to the activations and watch the choice move.

    python code/ch50_inside.py
"""
import json, os
import numpy as np
from ch08_sample import Runner
import ch19_system as big

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'runs', 'inside.json')


# ---- a runner with one place to read or change the activations -------------------------
class Open(Runner):
    """The cached forward pass of chapter 8, with a hook on what goes in and after every block."""
    edit = None

    # [snippet:hook]
    def step(self, token, cache):
        w, H, hd = self.w, self.H, self.d // self.H
        pos = cache['pos']
        x = w['wte'][token] + w['wpe'][pos]
        seen = cache.setdefault('seen', [[] for _ in range(self.L + 1)])
        if self.edit is not None:
            x = self.edit(0, pos, x)                      # stage 0: what goes in
        seen[0].append(x)
        for i in range(self.L):
            a = self.ln(x, w[f'h{i}.ln1.g'], w[f'h{i}.ln1.b'])
            q = (a @ w[f'h{i}.attn.wq']).reshape(H, hd)
            cache['k'][i].append((a @ w[f'h{i}.attn.wk']).reshape(H, hd))
            cache['v'][i].append((a @ w[f'h{i}.attn.wv']).reshape(H, hd))
            K, V = np.stack(cache['k'][i], axis=1), np.stack(cache['v'][i], axis=1)
            s = np.einsum('hd,htd->ht', q, K) / np.sqrt(hd)
            s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
            x = x + np.einsum('ht,htd->hd', s, V).reshape(self.d) @ w[f'h{i}.attn.wo']
            h = self.ln(x, w[f'h{i}.ln2.g'], w[f'h{i}.ln2.b']) @ w[f'h{i}.mlp.w1']
            h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h * h * h)))
            x = x + h @ w[f'h{i}.mlp.w2']
            if self.edit is not None:
                x = self.edit(i + 1, pos, x)              # stage i + 1: after block i. Return x, or another x
            seen[i + 1].append(x)                         # every activation is kept, to be read later
        cache['pos'] += 1
        return self.ln(x, w['ln_f.g'], w['ln_f.b']) @ w['wte'].T
    # [/snippet]


def ask(run, question, edit=None):
    """One question through the agent, always taking the favourite character. Returns (role, call, answer)."""
    run.edit = edit
    cache = run.new_cache()
    logits = big.feed(run, cache, '\nU ' + question + '\n')
    role, logits = big.write_line(run, cache, logits, fmt=[['C ', 'A ']], temperature=0)
    call = None
    if role == 'C ':
        call, logits = big.write_line(run, cache, logits, fmt=big.CALL, temperature=0)
        name, a, b = call.split()
        logits = big.feed(run, cache, f'R {big.TOOLS[name](int(a), int(b))}\nA ')
    text, _ = big.write_line(run, cache, logits, temperature=0)
    run.edit = None
    return role.strip(), (call or '').strip(), text.strip()

def activations(run, text):
    """Every activation the model computes while it reads `text`: [stage][place] -> 64 numbers.
    Stage 0 is what goes into the first block; stage 1 and 2 are what comes out of each block."""
    run.edit = None
    cache = run.new_cache()
    big.feed(run, cache, text)
    return cache['seen']

def pairs(n, seed):
    """Two-digit numbers the model never saw together in training."""
    rng, out = np.random.default_rng(seed), []
    while len(out) < n:
        a, b = int(rng.integers(10, 100)), int(rng.integers(10, 100))
        if big.held_out(a, b):
            out.append((a, b))
    return out


# ---- 1. swap one activation -------------------------------------------------------------
# [snippet:swap]
def swap_grid(run, word, other, numbers):
    """Ask "a WORD b". At one place and one stage, put in the activation from "a OTHER b".
    Count how often the tool that is called becomes the other word's tool."""
    places = len(f'\nU what is 12 {word} 30?\nC ')
    flipped, kept = np.zeros((run.L + 1, places), int), np.zeros((run.L + 1, places), int)
    for a, b in numbers:
        donor = activations(run, f'\nU what is {a} {other} {b}?\nC ')
        for stage in range(run.L + 1):
            for place in range(places):
                edit = lambda i, pos, x: donor[stage][place] if (i == stage and pos == place) else x
                _, call, _ = ask(run, f'what is {a} {word} {b}?', edit)
                if call.startswith(big.WORDS[other]):
                    flipped[stage, place] += 1
                    kept[stage, place] += call.split()[1:] == [str(a), str(b)]
    return flipped, kept
# [/snippet]


# ---- 2. a prediction about new inputs ---------------------------------------------------
def calls(run, word, numbers):
    out = {'add': 0, 'sub': 0, 'mul': 0, 'none': 0}
    for a, b in numbers:
        _, call, _ = ask(run, f'what is {a} {word} {b}?')
        out[call.split()[0] if call else 'none'] += 1
    return out


# ---- 3. a direction ----------------------------------------------------------------------
# [snippet:direction]
def mean_at(run, word, numbers, stage, tail='C '):
    """The average activation, at the place where the tool name is about to be written."""
    return np.mean([activations(run, f'\nU what is {a} {word} {b}?\n{tail}')[stage][-1] for a, b in numbers], axis=0)

def steer(run, word, other, direction, stage, strength, numbers):
    """Add strength x direction at that one place, and count the calls that change tool."""
    flipped = kept = 0
    for a, b in numbers:
        place = len(f'\nU what is {a} {word} {b}?\nC ') - 1
        edit = lambda i, pos, x: x + strength * direction if (i == stage and pos == place) else x
        _, call, _ = ask(run, f'what is {a} {word} {b}?', edit)
        if call.startswith(big.WORDS[other]):
            flipped += 1
            kept += call.split()[1:] == [str(a), str(b)]
    return flipped, kept
# [/snippet]


def main():
    run = Open(os.path.join(HERE, '..', 'runs', 'agent.npz'))
    last = run.L                                            # the stage after the last block
    out = {'n': 100, 'layers': run.L, 'width': run.d}
    test, fit = pairs(100, 3), pairs(200, 11)

    # the swap grid, both ways
    out['text'] = '\nU what is 12 times 30?\nC '
    out['grid'] = {}
    for word, other in (('times', 'minus'), ('minus', 'times')):
        flipped, kept = swap_grid(run, word, other, test)
        out['grid'][f'{word}>{other}'] = {'flipped': flipped.tolist(), 'kept': kept.tolist()}
        for stage in range(run.L + 1):
            print(f'{word}>{other} stage {stage}:', ' '.join(f'{p}{out["text"][p]!r}:{flipped[stage, p]}' for p in range(flipped.shape[1]) if flipped[stage, p]))

    # where do the two runs differ at all? A swap can only matter there.
    one, two = activations(run, '\nU what is 12 times 30?\nC '), activations(run, '\nU what is 12 minus 30?\nC ')
    out['differ'] = [[int(np.abs(np.array(one[s][p]) - np.array(two[s][p])).max() > 1e-6) for p in range(26)] for s in range(run.L + 1)]
    print('cells where the two runs differ, by stage:', [sum(row) for row in out['differ']])

    # the narrow claim: between "times" and "minus", the third letter decides. Every mix of the two words.
    out['words'] = {}
    for word in ['times', 'minus', 'plus', 'tines', 'tinus', 'timus', 'mimes', 'mimus', 'mines', 'over', 'and', 'by', 'divided by']:
        out['words'][word] = calls(run, word, test)
        print(f'{word:10s}', out['words'][word])

    # the broad claim: "it reads only the third letter". Change ONE letter of "times" to each other letter it knows.
    known = sorted(set(c for c in ''.join(big.conversation(np.random.default_rng(i)) for i in range(400)) if c.islower()))
    out['known'], out['scan'] = ''.join(known), []
    for place in range(5):
        row = {}
        for letter in known:
            if letter != 'times'[place]:
                row[letter] = calls(run, 'times'[:place] + letter + 'times'[place + 1:], test)['mul']
        out['scan'].append(row)
        print(f'place {place + 1} ({"times"[place]}): mul kept for', sum(v > 50 for v in row.values()), 'of', len(row), 'letters:', ''.join(k for k, v in row.items() if v > 50),
              '| mixed:', {k: v for k, v in row.items() if 0 < v < 100})

    # a direction at the place where the tool name starts
    means = {w: [mean_at(run, w, fit, stage) for stage in range(run.L + 1)] for w in big.WORDS}
    out['steer'], out['means'] = {}, {w: [np.round(m, 4).tolist() for m in ms] for w, ms in means.items()}
    strengths = [0, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.5]
    for word, other in (('times', 'minus'), ('minus', 'times'), ('plus', 'times'), ('times', 'plus')):
        for stage in range(1, run.L + 1):
            d = means[other][stage] - means[word][stage]
            row = [steer(run, word, other, d, stage, s, test) for s in strengths]
            out['steer'][f'{word}>{other}@{stage}'] = {'strengths': strengths, 'flipped': [r[0] for r in row], 'kept': [r[1] for r in row],
                                                      'length': round(float(np.linalg.norm(d)), 2)}
            print(f'steer {word}>{other} stage {stage} |d|={np.linalg.norm(d):.1f}', [r[0] for r in row], 'numbers kept', [r[1] for r in row])
    out['size'] = [round(float(np.mean([np.linalg.norm(means[w][stage]) for w in big.WORDS])), 2) for stage in range(run.L + 1)]

    # is it one of the 64 numbers, or a direction across them?
    A = np.array([activations(run, f'\nU what is {a} times {b}?\nC ')[last][-1] for a, b in test])
    B = np.array([activations(run, f'\nU what is {a} minus {b}?\nC ')[last][-1] for a, b in test])
    d = means['minus'][last] - means['times'][last]
    mid = (means['minus'][last] + means['times'][last]) / 2
    right = lambda pa, pb: int((pa < 0).sum() + (pb > 0).sum())
    along = right((A - mid) @ d, (B - mid) @ d)
    single = sorted((max(right(A[:, j] - mid[j], B[:, j] - mid[j]), right(mid[j] - A[:, j], mid[j] - B[:, j])) for j in range(run.d)), reverse=True)
    out['one_number'] = {'direction': along, 'best': single[0], 'median': single[run.d // 2], 'perfect': sum(s == 200 for s in single), 'of': 200}
    print('direction separates', along, 'of 200; best single number', single[0], 'median', single[run.d // 2], 'perfect ones', out['one_number']['perfect'])
    share = (d ** 2) / (d ** 2).sum()
    out['spread'] = {'largest': round(float(share.max()), 3), 'top8': round(float(np.sort(share)[-8:].sum()), 3)}
    print('share of the direction carried by its largest number', out['spread'])

    # the choice between a tool and a direct answer, at the end of the request
    sums_mean = np.mean([activations(run, f'\nU what is {a} {w} {b}?\n')[last][-1] for (a, b), w in zip(fit, list(big.WORDS) * 67)], axis=0)
    chat_mean = np.mean([activations(run, f'\nU {q}\n')[last][-1] for q, _ in big.CHAT], axis=0)
    d = chat_mean - sums_mean
    out['direct'] = {'strengths': [0, 0.25, 0.5, 0.75, 1], 'direct': [], 'right': [], 'shown': []}
    for s in out['direct']['strengths']:
        direct = right_n = 0
        for k, (a, b) in enumerate(test):
            word = list(big.WORDS)[k % 3]
            q = f'what is {a} {word} {b}?'
            place = len('\nU ' + q + '\n') - 1
            role, call, text = ask(run, q, lambda i, pos, x: x + s * d if (i == last and pos == place) else x)
            direct += role == 'A'
            right_n += text == f'{a} {word} {b} is {big.TOOLS[big.WORDS[word]](a, b)}.'
            if s == 1 and k < 6:
                out['direct']['shown'].append([q, role, call, text])
        out['direct']['direct'].append(direct); out['direct']['right'].append(right_n)
    out['direct']['mean_chat'], out['direct']['mean_sums'] = np.round(chat_mean, 4).tolist(), np.round(sums_mean, 4).tolist()
    print('direct answers', out['direct']['direct'], 'right', out['direct']['right'])
    for row in out['direct']['shown']:
        print('   ', row)

    json.dump(out, open(OUT, 'w'))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
