"""
ch10_chat.py — a chat in your terminal, served from your own machine.   (Chapter 10)

    ollama pull <model>          # or start llama-server / LM Studio
    python code/ch10_chat.py

No model yet?  `python code/mock_server.py` starts a fake server so you can see
the plumbing work:  MICROAI_BASE_URL=http://localhost:8765/v1 python code/ch10_chat.py
"""
import sys
from llm import stream, default_model, BASE_URL

# [snippet:chat]
def main():
    print(f'model: {default_model()} at {BASE_URL}   (ctrl-c to quit)')
    messages = [{'role': 'system', 'content': 'You are a concise, helpful assistant.'}]
    while True:
        try:
            user = input('\nyou> ')
        except (EOFError, KeyboardInterrupt):
            break
        messages.append({'role': 'user', 'content': user})
        reply = ''
        for piece in stream(messages, temperature=0.7):   # tokens arrive as they are sampled
            reply += piece
            print(piece, end='', flush=True)
        print()
        # The server keeps no memory. The "conversation" exists only because
        # we send the whole transcript again on every turn.
        messages.append({'role': 'assistant', 'content': reply})
# [/snippet]

if __name__ == '__main__':
    try:
        main()
    except OSError as e:
        sys.exit(f'Could not reach a model server at {BASE_URL} ({e}).\n'
                 'Start Ollama / llama-server, or run `python code/mock_server.py` and set MICROAI_BASE_URL.')
