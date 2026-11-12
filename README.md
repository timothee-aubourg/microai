# microAI

**Build and understand a local language model from first principles.**

An interactive course in twelve chapters. Each chapter starts from a question,
gives you an experiment to play with in the browser, and ends with the few
lines of code that implement the idea. One small GPT, trained from scratch with
numpy, runs through the whole course: you watch it learn, look inside its
attention heads, sample from it, quantize it, and fine-tune it, all with the
real weights.

By the end you can re-implement every piece of a minimal local AI system and
you know what each component of a full-size one is doing.

## Read the course

Open `docs/index.html` in a browser. It needs no server and no internet
connection.

To publish it as a website with GitHub Pages: push this repository, then go to
**Settings → Pages → Build and deployment**, choose **Deploy from a branch**,
branch `main`, folder `/docs`, and save. The course appears at
`https://<your-user>.github.io/<repo-name>/` a minute later.

| | Chapter | The question |
|---|---|---|
| 1 | Guess the next character | How can a file of numbers write a sentence? |
| 2 | Cutting text into tokens | Why not letters, and why not words? |
| 3 | Learning by rolling downhill | How do you tune 345,504 numbers at once? |
| 4 | Meaning as direction | How can arithmetic on ID numbers find that `a` and `e` are alike? |
| 5 | Letting tokens look at each other | Why a softmax, and why divide by a square root? |
| 6 | The whole machine | What computes, and how do you stack it? |
| 7 | Watching it learn | What does training look like, step by step? |
| 8 | Rolling the dice | Why not always pick the most likely token? |
| 9 | Fitting it in your RAM | How does a 16 GB model run on a 16 GB laptop? |
| 10 | A real model on your machine | What is in the file, and what does the runtime do? |
| 11 | Memory, books and hands | How do you build a system from a frozen, forgetful text predictor? |
| 12 | Teaching new habits | Can you change a model without retraining all of it? |

## Run the code

The only dependency is numpy (Python 3.9 or later).

```bash
pip install numpy

python code/ch01_ngram.py              # a language model made of counts
python code/ch02_bpe.py                # a tokenizer in 30 lines
python code/ch03_gradient_descent.py   # gradient descent, autograd, gradient check
python code/train.py                   # train the GPT (about 15 min on a laptop CPU)
python code/ch08_sample.py --prompt "ROMEO:" --temperature 0.8
python code/ch09_quantize.py           # round the weights, measure the damage
python code/ch12_lora.py               # train a small adapter on the frozen model
```

The trained weights are included (`runs/micro.npz`, `runs/lora.npz`), so every
script works before you train anything yourself.

Chapters 10 and 11 talk to a model server on your own machine. With
[Ollama](https://ollama.com) installed and a model pulled:

```bash
python code/ch10_chat.py                                   # chat in the terminal
python code/ch11_rag.py "why divide attention scores by sqrt(d)?"
python code/ch11_agent.py "What is 17 * 2340 / 100?"
```

For llama.cpp's `llama-server` or LM Studio, set `MICROAI_BASE_URL` (see
`code/llm.py`). No model installed? A fake server lets you watch the plumbing:

```bash
python code/mock_server.py &
export MICROAI_BASE_URL=http://localhost:8765/v1
python code/ch11_agent.py "What is 17 * 2340 / 100?"
```

## What is in the repository

```
code/
  engine.py                 reverse-mode autograd on numpy arrays (ch 3)
  gpt.py                    the GPT: embeddings, attention, blocks (ch 4-6)
  train.py                  the training loop (ch 7)
  ch01_ngram.py             count-based language model, loss
  ch02_bpe.py               byte-pair encoding
  ch03_gradient_descent.py  descent by hand, with autograd, and a gradient check
  ch08_sample.py            temperature, top-k, top-p, KV cache
  ch09_quantize.py          quantization to n bits with per-block scales
  llm.py                    30-line client for a local model server (ch 10)
  ch10_chat.py              terminal chat
  ch11_rag.py               retrieval-augmented generation
  ch11_agent.py             tool calling and the agent loop
  ch12_lora.py              LoRA fine-tuning of the frozen model
  mock_server.py            fake model server for testing without a model
data/
  shakespeare.txt           training text (public domain, 1.1 MB)
  notes.md                  short course notes, used as documents by ch 11
docs/                       the course website (plain HTML, CSS and JavaScript)
  assets/micro.js           helpers and a full GPT forward pass in JavaScript
  assets/weights.js         the trained weights, float16
  assets/data.js            training log, embedding map, text excerpts
runs/                       trained weights and the training log
tools/
  export_web.py             pack weights and logs for the website
  build_site.py             copy code snippets from code/ into the pages
  test_parity.js            check that the browser model matches the Python one
```

## Changing things

The code shown in the chapters is copied from `code/` by a script, so the pages
always show what actually runs. After editing code or retraining:

```bash
python code/train.py          # optional: retrain (edit sizes at the top of gpt.py / train.py)
python code/ch12_lora.py      # optional: retrain the adapter
python tools/export_web.py    # weights and logs -> docs/assets
python tools/build_site.py    # code snippets -> docs/*.html
node tools/test_parity.js     # browser model == Python model?
```

To train on your own text, replace `data/shakespeare.txt` with any plain-text
file of a few hundred kilobytes or more and run the five commands above. A few
sentences in chapters 4, 5, 7 and 9 describe what this particular model learned
and quote its measured numbers; after retraining, check them against yours.

## How this was checked

- `ch03_gradient_descent.py` compares backpropagation with finite differences
  on a small GPT; they agree to about 1e-9.
- `ch08_sample.py` asserts that the KV-cache path gives the same outputs as
  recomputing everything.
- `tools/test_parity.js` confirms that the JavaScript model in the browser
  reproduces the Python model's outputs and held-out loss.
- Every page was loaded in a headless browser at phone and desktop widths with
  no script errors, and each experiment was exercised.
- The chat, retrieval and agent scripts were run end to end against
  `mock_server.py`. They follow the OpenAI-compatible API that Ollama
  documents, but **they have not been run against a real Ollama, llama.cpp or
  LM Studio server**. If one behaves differently, the fix belongs in
  `code/llm.py`.

## Scope

This course covers the path from next-token prediction to a working local
system. It deliberately leaves out reinforcement-learning fine-tuning (policy
gradients, PPO, GRPO), multimodal models and evaluation methodology; chapter 12
sketches where those begin.

## Sources and credits

The teaching approach is inspired by David Louapre
([Science Étonnante](https://www.youtube.com/@ScienceEtonnante)), Grant
Sanderson ([3Blue1Brown](https://www.3blue1brown.com/)) and Andrej Karpathy
([micrograd](https://github.com/karpathy/micrograd),
[nanoGPT](https://github.com/karpathy/nanoGPT)). All text, code and visuals
here are original.

- Vaswani et al., *Attention Is All You Need*, 2017: <https://arxiv.org/abs/1706.03762>
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021: <https://arxiv.org/abs/2106.09685>
- llama.cpp (GGUF, quantization formats and their bits per weight): <https://github.com/ggml-org/llama.cpp>
- Ollama, OpenAI-compatible API: <https://docs.ollama.com/api/openai-compatibility>
- Training text: the "tiny Shakespeare" file from <https://github.com/karpathy/char-rnn>

Facts about tools and file formats were checked in November 2026. Model names
and tool versions change quickly; the arithmetic does not.

## License

MIT. See `LICENSE`.
