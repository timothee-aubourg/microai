"""
ch02_bpe.py — byte-pair encoding: how text becomes tokens.              (Chapter 2)

    python code/ch02_bpe.py

Start from raw bytes (256 symbols, so ANY text is encodable), then repeatedly
glue the most frequent adjacent pair into a new token. That is the whole idea
behind the tokenizers of GPT-style models.
"""
from collections import Counter
from gpt import load_text

# [snippet:bpe]
def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in ids with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

def train_bpe(text, n_merges):
    ids = list(text.encode('utf-8'))            # start: one token per byte
    merges = {}                                 # (a, b) -> new token id, in learned order
    for k in range(n_merges):
        pairs = Counter(zip(ids, ids[1:]))      # count adjacent pairs
        if not pairs:
            break
        best = max(pairs, key=pairs.get)        # the most frequent pair...
        merges[best] = 256 + k                  # ...becomes a new token
        ids = merge(ids, best, 256 + k)
    return merges

def encode(text, merges):
    ids = list(text.encode('utf-8'))
    for pair, new_id in merges.items():         # replay the merges in the order learned
        ids = merge(ids, pair, new_id)
    return ids

def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]
    return b''.join(vocab[i] for i in ids).decode('utf-8', errors='replace')
# [/snippet]

if __name__ == '__main__':
    corpus = load_text()[:50000]
    merges = train_bpe(corpus, 300)
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), n in merges.items():
        vocab[n] = vocab[a] + vocab[b]

    print('first merges :', [vocab[n].decode() for n in list(merges.values())[:12]])
    print('later merges :', [vocab[n].decode() for n in list(merges.values())[-12:]])

    s = 'Speak, speak. We are accounted poor citizens, the patricians good.'
    ids = encode(s, merges)
    print(f'\n"{s}"')
    print(' | '.join(vocab[i].decode(errors='replace') for i in ids))
    print(f'{len(s)} characters -> {len(ids)} tokens  ({len(s) / len(ids):.2f} characters per token)')
    assert decode(ids, merges) == s

    emoji = 'naïve café 🤖'                      # never seen in training: still encodable, thanks to bytes
    assert decode(encode(emoji, merges), merges) == emoji
    print(f'\nunseen text "{emoji}" -> {len(encode(emoji, merges))} tokens, round-trips exactly')
