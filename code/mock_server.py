"""
mock_server.py — a fake model server, for testing the plumbing without a model.

    python code/mock_server.py            # listens on http://localhost:8765/v1
    MICROAI_BASE_URL=http://localhost:8765/v1 python code/ch11_agent.py

It speaks the same HTTP dialect as Ollama / llama-server, but there is no
intelligence here: replies are canned, tool calls are chosen by regex, and
"embeddings" are hashed word counts. It exists so that every script in this
repo can be run and tested end to end on a machine with no model installed.
"""
import hashlib, json, re, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer


def fake_reply(payload):
    messages, tools = payload['messages'], payload.get('tools')
    last = messages[-1]
    if last['role'] == 'tool':
        return {'role': 'assistant', 'content': f"[mock] The tool returned: {last['content'][:200]}"}
    if tools and last['role'] == 'user':
        names = [t['function']['name'] for t in tools]
        expr = re.search(r'\d[\d\s\.\+\-\*/\(\)%]*[\+\-\*/][\d\s\.\+\-\*/\(\)]*\d', last['content'])
        if expr and 'calculator' in names:
            call = ('calculator', {'expression': expr.group(0).strip()})
        else:
            call = (names[-1] if 'search_notes' not in names else 'search_notes', {'query': last['content']})
        return {'role': 'assistant', 'content': None, 'tool_calls': [{
            'id': 'call_1', 'type': 'function',
            'function': {'name': call[0], 'arguments': json.dumps(call[1])}}]}
    return {'role': 'assistant', 'content': f"[mock] I am not a real model. You said: {str(last['content'])[:300]}"}


def fake_embedding(text, dim=256):
    v = [0.0] * dim
    for word in re.findall(r'[a-z0-9]+', text.lower()):
        v[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    return v


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip('/').endswith('/models'):
            return self._json({'object': 'list', 'data': [{'id': 'mock-model', 'object': 'model'}]})
        self.send_error(404)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path.endswith('/embeddings'):
            texts = payload['input'] if isinstance(payload['input'], list) else [payload['input']]
            return self._json({'data': [{'index': i, 'embedding': fake_embedding(t)} for i, t in enumerate(texts)]})
        if not self.path.endswith('/chat/completions'):
            return self.send_error(404)
        msg = fake_reply(payload)
        if not payload.get('stream'):
            return self._json({'choices': [{'index': 0, 'message': msg, 'finish_reason': 'stop'}]})
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        for piece in re.findall(r'\S+\s*', msg['content'] or ''):
            chunk = {'choices': [{'index': 0, 'delta': {'content': piece}}]}
            self.wfile.write(f'data: {json.dumps(chunk)}\n\n'.encode())
            self.wfile.flush()
            time.sleep(0.02)
        self.wfile.write(b'data: [DONE]\n\n')


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'mock model server on http://localhost:{port}/v1  (ctrl-c to stop)')
    HTTPServer(('127.0.0.1', port), Handler).serve_forever()
