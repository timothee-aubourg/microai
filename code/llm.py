"""
llm.py — talk to a model server on your own machine.                   (Chapter 10)

Thirty lines of standard library. Works with anything that speaks the
OpenAI-style HTTP API, which today includes Ollama (port 11434),
llama.cpp's llama-server (port 8080) and LM Studio (port 1234).

    export MICROAI_BASE_URL=http://localhost:11434/v1     # default: Ollama
    export MICROAI_MODEL=<a model you pulled>             # default: first one the server lists
"""
import json, os, urllib.request

BASE_URL = os.environ.get('MICROAI_BASE_URL', 'http://localhost:11434/v1').rstrip('/')


# [snippet:http]
def _post(path, payload, stream=False):
    req = urllib.request.Request(BASE_URL + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json', 'Authorization': 'Bearer local'})
    resp = urllib.request.urlopen(req, timeout=600)
    return resp if stream else json.loads(resp.read())

def default_model():
    if os.environ.get('MICROAI_MODEL'):
        return os.environ['MICROAI_MODEL']
    with urllib.request.urlopen(BASE_URL + '/models', timeout=10) as r:
        return json.loads(r.read())['data'][0]['id']       # whatever is installed

def chat(messages, tools=None, model=None, **options):
    """Send the whole conversation, get back the assistant's next message (a dict)."""
    payload = {'model': model or default_model(), 'messages': messages, **options}
    if tools:
        payload['tools'] = tools
    return _post('/chat/completions', payload)['choices'][0]['message']

def stream(messages, model=None, **options):
    """Same, but yield the text piece by piece as the model samples it."""
    payload = {'model': model or default_model(), 'messages': messages, 'stream': True, **options}
    for line in _post('/chat/completions', payload, stream=True):
        line = line.decode().strip()
        if line.startswith('data: ') and line != 'data: [DONE]':
            delta = json.loads(line[6:])['choices'][0]['delta']
            if delta.get('content'):
                yield delta['content']
# [/snippet]

def embed(texts, model=None):
    """Turn each text into a vector with an embedding model served locally."""
    model = model or os.environ.get('MICROAI_EMBED_MODEL') or default_model()
    data = _post('/embeddings', {'model': model, 'input': texts})['data']
    return [d['embedding'] for d in data]
