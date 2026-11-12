"""
ch11_agent.py — an agent is a while-loop around a language model.       (Chapter 11)

    python code/ch11_agent.py "What is 17% of 2340, and what do the notes say about KV cache?"

The model cannot DO anything. It can only write text. So we make a deal:
if it writes a specially formatted request ("call calculator with ..."),
our program runs the tool and appends the result to the conversation.
Then we ask the model again. That is the entire trick.
"""
import ast, json, operator, sys
from llm import chat
from ch11_rag import load_chunks, Index, make_embedder

# ---- two tools ------------------------------------------------------------------
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg}

def calculator(expression):
    """Arithmetic only. Never eval() text written by a model."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.operand))
        raise ValueError('only numbers and + - * / ** % are allowed')
    return ev(ast.parse(expression.replace('%', '/100*') if expression.strip().endswith('%') else expression, mode='eval').body)

_index = None
def search_notes(query):
    global _index
    if _index is None:
        chunks = load_chunks()
        _index = Index(chunks, make_embedder(chunks))
    return '\n\n'.join(chunk for _, chunk in _index.search(query, k=2))

REGISTRY = {'calculator': calculator, 'search_notes': search_notes}

# [snippet:tools]
# What the model is told about the tools: a name, a description, a JSON schema.
TOOLS = [
    {'type': 'function', 'function': {
        'name': 'calculator',
        'description': 'Evaluate an arithmetic expression, e.g. "0.17 * 2340".',
        'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression']}}},
    {'type': 'function', 'function': {
        'name': 'search_notes',
        'description': 'Search the course notes on local LLMs. Returns the most relevant passages.',
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}},
]
# [/snippet]

# [snippet:loop]
def run_agent(question, max_steps=6, verbose=True):
    messages = [
        {'role': 'system', 'content': 'Use the tools when they help. Answer briefly when you are done.'},
        {'role': 'user', 'content': question},
    ]
    for _ in range(max_steps):
        reply = chat(messages, tools=TOOLS, temperature=0)
        messages.append(reply)
        if not reply.get('tool_calls'):            # plain text: the model is finished
            return reply['content']
        for call in reply['tool_calls']:           # the model asked for something
            name, args = call['function']['name'], call['function']['arguments']
            args = json.loads(args) if isinstance(args, str) else args
            try:
                result = str(REGISTRY[name](**args))
            except Exception as e:                 # errors go back to the model: it can retry
                result = f'error: {e}'
            if verbose:
                print(f'  tool  {name}({args}) -> {result[:80]!r}')
            messages.append({'role': 'tool', 'tool_call_id': call.get('id', ''), 'content': result})
    return 'Stopped: too many steps.'              # a loop needs a brake
# [/snippet]

if __name__ == '__main__':
    question = ' '.join(sys.argv[1:]) or 'What is 17 * 2340 / 100?'
    try:
        print('answer:', run_agent(question))
    except OSError as e:
        sys.exit(f'No model server reachable ({e}). Start one (chapter 10) or run code/mock_server.py.')
