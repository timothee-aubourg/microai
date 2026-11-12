"""
ch11_rag.py — give the model an open book.                             (Chapter 11)

    python code/ch11_rag.py "why divide attention scores by sqrt(d)?"

Retrieval-augmented generation in four steps:
  1. cut documents into chunks          3. find the chunks nearest to the question
  2. turn every chunk into a vector     4. paste them into the prompt

The documents here are this course's own notes (data/notes.md).
Set MICROAI_EMBED_MODEL to use a real embedding model from your local server;
otherwise a tiny word-count embedding (TF-IDF) is used so the script runs anywhere.
"""
import math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# [snippet:chunk]
def load_chunks(path=os.path.join(HERE, '..', 'data', 'notes.md')):
    """One chunk per '## ' section. Real systems cut by size with overlap; the idea is the same."""
    with open(path, encoding='utf-8') as f:
        sections = re.split(r'^## ', f.read(), flags=re.M)[1:]
    return [s.strip() for s in sections]
# [/snippet]


class TfidfEmbedder:
    """A stand-in for a learned embedding model: rare shared words = similar texts."""
    def __init__(self, docs):
        words = [set(self.words(d)) for d in docs]
        self.vocab = {w: i for i, w in enumerate(sorted(set().union(*words)))}
        self.idf = np.ones(len(self.vocab))
        for w, i in self.vocab.items():
            df = sum(w in ws for ws in words)
            self.idf[i] = math.log((1 + len(docs)) / (1 + df)) + 1    # rare words count more

    @staticmethod
    def words(text):
        return re.findall(r'[a-z0-9]+', text.lower())

    def __call__(self, texts):
        out = np.zeros((len(texts), len(self.vocab)))
        for r, t in enumerate(texts):
            for w in self.words(t):
                if w in self.vocab:
                    out[r, self.vocab[w]] += self.idf[self.vocab[w]]
        return out


def make_embedder(chunks):
    if os.environ.get('MICROAI_EMBED_MODEL'):
        from llm import embed
        return lambda texts: np.array(embed(texts))
    return TfidfEmbedder(chunks)


# [snippet:retrieve]
def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

class Index:
    def __init__(self, chunks, embedder):
        self.chunks, self.embedder = chunks, embedder
        self.vectors = normalize(embedder(chunks))          # computed once, reused for every question

    def search(self, question, k=2):
        q = normalize(self.embedder([question]))[0]
        scores = self.vectors @ q                           # cosine similarity with every chunk
        best = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in best]

def build_prompt(question, hits):
    context = '\n\n'.join(f'[{n + 1}] {chunk}' for n, (_, chunk) in enumerate(hits))
    return ('Answer the question using only the notes below. '
            'Cite the note number. If the notes do not contain the answer, say so.\n\n'
            f'NOTES\n{context}\n\nQUESTION\n{question}')
# [/snippet]


if __name__ == '__main__':
    question = ' '.join(sys.argv[1:]) or 'Why are attention scores divided by the square root of the dimension?'
    chunks = load_chunks()
    index = Index(chunks, make_embedder(chunks))
    hits = index.search(question)
    print(f'{len(chunks)} chunks indexed. Nearest to the question:')
    for score, chunk in hits:
        print(f'  {score:.3f}  {chunk.splitlines()[0]}')
    prompt = build_prompt(question, hits)
    print(f'\n--- prompt sent to the model ({len(prompt)} characters) ---\n{prompt}\n')
    try:
        from llm import chat
        print('--- answer ---\n' + chat([{'role': 'user', 'content': prompt}], temperature=0.2)['content'])
    except OSError as e:
        print(f'(no model server reachable: {e}. Retrieval above still ran; see ch10 to start a server.)')
