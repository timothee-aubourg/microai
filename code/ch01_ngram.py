"""
ch01_ngram.py — the simplest language model: count what follows what.   (Chapter 1)

    python code/ch01_ngram.py

A language model is anything that answers: "given the text so far, how likely
is each possible next token?"  Here the answer is a lookup table of counts.
Watch the validation loss: more context helps, until the table is mostly empty.
"""
import math, random
from collections import Counter, defaultdict
from gpt import load_text

text = load_text()
split = int(0.9 * len(text))
train, val = text[:split], text[split:split + 20000]
V = len(set(text))

# [snippet:fit]
def fit(order):
    """counts[context][next_char] = how often next_char followed context."""
    counts = defaultdict(Counter)
    for i in range(order, len(train)):
        counts[train[i - order:i]][train[i]] += 1
    return counts

def prob(counts, context, ch, alpha=0.01):
    """P(ch | context), with a tiny bit of probability reserved for the unseen."""
    c = counts.get(context, {})
    return (c.get(ch, 0) + alpha) / (sum(c.values()) + alpha * V)

def loss(counts, order, s):
    """Cross-entropy: the average surprise, -log p(what actually came next)."""
    return -sum(math.log(prob(counts, s[i - order:i], s[i])) for i in range(order, len(s))) / (len(s) - order)
# [/snippet]

# [snippet:sample]
def sample(counts, order, n=120, seed=0):
    rnd = random.Random(seed)
    out = train[:order]                        # seed with the first few characters
    for _ in range(n):
        c = counts.get(out[len(out) - order:], None)
        if not c:                              # never saw this context: the table has no answer
            out += rnd.choice(train)
            continue
        chars, weights = zip(*c.items())
        out += rnd.choices(chars, weights)[0]  # roll the weighted dice
    return out[order:]
# [/snippet]

if __name__ == '__main__':
    print(f'uniform guessing over {V} characters costs ln({V}) = {math.log(V):.3f} nats per character\n')
    print('order  contexts-seen  possible-contexts   train-loss  val-loss')
    for order in range(0, 7):
        counts = fit(order)
        print(f'{order:5d}  {len(counts):13,d}  {V ** order:17,d}   {loss(counts, order, train[:20000]):10.3f}  {loss(counts, order, val):8.3f}')
    for order in (0, 1, 3, 5):
        print(f'\n--- order {order} ---\n{sample(fit(order), order)}')
