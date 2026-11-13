"""
mock_server.py — a fake model server, for testing the plumbing without a model.

    python code/mock_server.py            # listens on http://localhost:8765/v1
    MICROAI_BASE_URL=http://localhost:8765/v1 python code/ch11_agent.py

It speaks the same HTTP dialect as Ollama / llama-server, but there is no
intelligence here: replies are canned, tool calls are chosen by regex, and
"embeddings" are hashed word counts. When the request offers a `bash` tool it
plays a fixed coding routine instead (see fake_coder). It exists so that every script in this
repo can be run and tested end to end on a machine with no model installed.
"""
import hashlib, json, re, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer


def tool_call(name, args):
    return {'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': 'call_1', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}


def fake_coder(messages):
    """A scripted stand-in for a coding model (chapters 15 to 17). It plays one routine:
    run the tests, read the file they blame, fix one known bug, run the tests again.
    It also obeys any line of the form  AI AGENT: {"tool": ..., "args": ...}  that it
    finds in a tool result. A real model is harder to fool; chapter 17 needs one that is easy."""
    results = [m['content'] for m in messages if m['role'] == 'tool']
    asked = [c['function'] for m in messages if m.get('tool_calls') for c in m['tool_calls']]
    names = [c['name'] for c in asked]
    for text in results:                                   # obey planted instructions, once each
        for line in re.findall(r'AI AGENT: (\{.*\})', text):
            order = json.loads(line)
            if json.dumps(order['args']) not in [c['arguments'] for c in asked]:
                return tool_call(order['tool'], order['args'])
    task = next(m['content'] for m in messages if m['role'] == 'user')
    test = (re.search(r'`([^`]+)`', task) or [None, 'python test_stats.py'])[1]
    if 'read' not in names:
        if not results:
            return tool_call('bash', {'command': test})    # 1. look at the failure
        blamed = re.search(r'FAIL in (\S+\.py)', ' '.join(results))
        return tool_call('read', {'path': blamed.group(1) if blamed else 'stats.py'})    # 2. read the code
    if 'edit' not in names:
        bug = re.search(r'/ \(len\((\w+)\) - 1\)', ' '.join(results))
        if bug:                                            # 3. fix the one bug this script knows
            return tool_call('edit', {'path': 'stats.py', 'old': bug.group(0), 'new': f'/ len({bug.group(1)})'})
        return {'role': 'assistant', 'content': '[mock] I could not find the bug this script knows how to fix.'}
    if not results[-1].startswith('exit code 0'):
        if [c['arguments'] for c in asked].count(json.dumps({'command': test})) < 2:
            return tool_call('bash', {'command': test})    # 4. check the fix
        return {'role': 'assistant', 'content': '[mock] The tests still fail after my change.'}
    return {'role': 'assistant', 'content': '[mock] Fixed. mean() divided by len(xs) - 1; it now divides by len(xs), and the tests pass.'}


def fake_reply(payload):
    messages, tools = payload['messages'], payload.get('tools')
    last = messages[-1]
    if tools and any(t['function']['name'] == 'bash' for t in tools):
        return fake_coder(messages)
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
