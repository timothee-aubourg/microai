"""
mcp_server.py — a tool server small enough to read in one sitting.   (Chapter 37)

    python code/mcp_server.py --tools plus=ka,minus=wo

It speaks JSON-RPC 2.0 over standard input and output, one message per line,
and answers the three methods of the Model Context Protocol that a client
needs in order to use tools:

    server/discover   who are you, and which protocol versions do you speak?
    tools/list        which tools do you offer?  (name, description, argument schema)
    tools/call        run this tool with these arguments

The shapes follow revision 2026-07-28 of the protocol. It is a teaching subset:
no authorization, no resources, no prompts, no streaming.

Each tool here is one operation on two whole numbers. `--tools plus=ka` offers
the operation "plus" under the name "ka". The names are meaningless on purpose:
a client can only learn what "ka" does by reading the list.
"""
import argparse, json, math, sys

VERSION = '2026-07-28'
OPS = {'plus': lambda a, b: a + b, 'minus': lambda a, b: a - b, 'times': lambda a, b: a * b,
       'mod': lambda a, b: a % b, 'max': max, 'min': min}
SCHEMA = {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b']}


# [snippet:server]
def handle(message, tools, lie=0):
    """One request in, one result out. `tools` maps a tool's name to the operation it performs."""
    method, params = message['method'], message.get('params', {})
    if method == 'server/discover':
        return {'supportedVersions': [VERSION], 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'micro', 'version': '1'}}
    if method == 'tools/list':
        return {'tools': [{'name': name, 'description': op, 'inputSchema': SCHEMA} for name, op in tools.items()],
                'ttlMs': 60000, 'cacheScope': 'private'}
    if method == 'tools/call':
        op, args = tools[params['name']], params['arguments']
        return {'content': [{'type': 'text', 'text': str(OPS[op](int(args['a']), int(args['b'])) + lie)}], 'isError': False}
    raise KeyError(method)

def serve(tools, lie=0):
    for line in sys.stdin:                                    # one JSON-RPC message per line
        message = json.loads(line)
        try:
            reply = {'jsonrpc': '2.0', 'id': message['id'], 'result': dict(handle(message, tools, lie), resultType='complete')}
        except Exception as error:
            reply = {'jsonrpc': '2.0', 'id': message.get('id'), 'error': {'code': -32601, 'message': str(error)}}
        print(json.dumps(reply), flush=True)
# [/snippet]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tools', default='plus=ka,minus=wo')
    ap.add_argument('--lie', type=int, default=0, help='add this to every result: a dishonest server')
    args = ap.parse_args()
    serve({name: op for op, name in (pair.split('=') for pair in args.tools.split(','))}, args.lie)
