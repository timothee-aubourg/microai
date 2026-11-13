"""
ch25_infer.py — a better answer from the same weights.   (Chapter 25)

    python code/ch25_infer.py        # about 5 minutes; needs the models of chapter 24

Chapter 24 changed a model's behaviour by training it. This file changes
nothing in the model. It takes the PRE-TRAINED model of chapter 24, the one
that guesses half its sums, and changes only what the harness does when it
asks:

    once      ask one time, take the answer                       (the baseline)
    vote      ask N times, keep the answer given most often
    check     ask again until the answer rests on a tool result
    lean      add a number to one score before the model chooses
    filter    for the secret: ask again while the reply contains it

Each method is graded on the same 300 unseen sums, and its cost is counted in
tokens the model had to process.
"""
import json, os, sys, collections
import numpy as np
import ch19_system as big
import ch24_posttrain as post

OUT = os.path.join(post.RUNS, 'infer.json')


def sums(n=300, seed=1):
    rng, out = np.random.default_rng(seed), []
    while len(out) < n:
        a, b = int(rng.integers(100)), int(rng.integers(100))
        if big.held_out(a, b):
            out.append(f'what is {a} {list(big.WORDS)[rng.integers(3)]} {b}?')
    return out

# [snippet:vote]
def vote(run, question, rng, n):
    """Ask n times. Different guesses disagree with each other; answers that are right agree."""
    episodes = [post.episode(run, question, rng) for _ in range(n)]
    answer, _ = collections.Counter(e['answer'] for e in episodes).most_common(1)[0]
    return answer, sum(len(e['text']) for e in episodes)
# [/snippet]

# [snippet:check]
def check(run, question, rng, tries=5):
    """Accept an answer only if it rests on a tool result. Otherwise ask again."""
    tokens = 0
    for attempt in range(1, tries + 1):
        e = post.episode(run, question, rng)
        tokens += len(e['text'])
        if e['call'] is not None:                 # the check: did the number come from the calculator?
            break
    return e['answer'], tokens, attempt
# [/snippet]

# [snippet:lean]
def lean(run, question, rng, amount):
    """One number added to the score of "C" at the moment the model chooses between calling and answering."""
    e = post.episode(run, question, rng, lean=amount)
    return e['answer'], len(e['text'])
# [/snippet]

def keep_secret(run, ask, rng, tries=10):
    """An output filter: a reply that contains the secret is thrown away and the question asked again."""
    tokens = 0
    for attempt in range(1, tries + 1):
        e = post.episode(run, ask, rng)
        tokens += len(e['text'])
        if post.SECRET not in e['answer']:
            return e['answer'], tokens, attempt
    return None, tokens, tries                    # every try leaked: the harness answers nothing


if __name__ == '__main__':
    if not os.path.exists(post.PATH['pre']):
        sys.exit('Needs the models of chapter 24. Run:  python code/ch24_posttrain.py --train')
    run = post.live(post.load('pre'))
    questions, rows = sums(), []
    right = lambda q, answer: post.grade(q, {'answer': answer}) == 1

    def measure(name, method, extra=None):
        rng, ok, tokens, more = np.random.default_rng(0), 0, 0, 0
        for q in questions:
            out = method(q, rng)
            ok, tokens = ok + right(q, out[0]), tokens + out[1]
            more += out[2] if len(out) > 2 else 0
        row = {'name': name, 'right': ok, 'n': len(questions), 'tokens': round(tokens / len(questions))}
        if extra:
            row[extra] = round(more / len(questions), 2)
        rows.append(row)
        print(f"{name:44}{ok:>4} / {len(questions)} right   {row['tokens']:>5} tokens per question" + (f"   {row[extra]} {extra}" if extra else ''), flush=True)

    print('the pre-trained model of chapter 24, weights untouched\n')
    measure('ask once', lambda q, rng: vote(run, q, rng, 1))
    for n in (3, 5, 9, 15):
        measure(f'ask {n} times and vote', lambda q, rng, n=n: vote(run, q, rng, n))
    measure('ask again until a tool was used (5 tries)', lambda q, rng: check(run, q, rng), 'tries')
    for amount in (1, 2, 4):
        measure(f'lean toward the tool by {amount}', lambda q, rng, a=amount: lean(run, q, rng, a))

    rng, kept, gave_up, tokens, tries = np.random.default_rng(0), 0, 0, 0, 0
    for i in range(300):
        answer, t, k = keep_secret(run, post.ASKS[i % 2], rng)
        kept, gave_up, tokens, tries = kept + (answer == post.REFUSAL), gave_up + (answer is None), tokens + t, tries + k
    secret = {'declined': kept, 'silent': gave_up, 'leaked': 0, 'n': 300, 'tokens': round(tokens / 300), 'tries': round(tries / 300, 2)}
    print(f"\nthe secret, with an output filter (10 tries): declined {kept} / 300, no answer {gave_up} / 300, given away 0 / 300, {secret['tries']} tries and {secret['tokens']} tokens per request")

    # what no method here can do: a question outside what the model learned
    rng, hard, ok = np.random.default_rng(0), [f'what is {100 + 7 * i} plus {i}?' for i in range(50)], 0
    for q in hard:
        a, word, b = q.rstrip('?').split()[2:]
        ok += vote(run, q, rng, 15)[0] == f'{a} {word} {b} is {int(a) + int(b)}.'
    print(f'sums with a three-digit number, 15 votes each: {ok} / {len(hard)} right')
    with open(OUT, 'w') as f:
        json.dump({'rows': rows, 'secret': secret, 'outside': {'right': ok, 'n': len(hard)}}, f)
