# microAI

**Build and understand a local language model, and the agent around it, from first principles.**

An interactive course in thirty-four chapters, in thirteen parts. Each chapter starts
from a question and a ten-second guess, gives you an experiment to play with in
the browser, shows the few lines of code that implement the idea, and ends with
one line to keep. One small GPT, trained from scratch with numpy, runs through
the first twelve chapters: you watch it learn, look inside its attention heads,
sample from it, quantize it, and fine-tune it, all with the real weights.
Chapters 13 to 19 build what surrounds a model: structured output, tracing and
evaluation, the harness, context and caching, security, hardware, and finally a
complete tool-using agent that you train yourself in five minutes. Chapters 20
and 21 add two ideas that full-size models rely on, each trained and measured
at toy size: a mixture of experts, and thinking tokens. Chapters 22 and 23 fit
the model to the machine: every kind of memory a model can live in, and the
chapter 19 agent retrained with three-valued weights into a file of 20
kilobytes that runs without numpy. Chapter 24 runs the training that comes
after pre-training, on written examples and then on rewards alone. Chapter 25
improves the same model's answers at inference, with its weights fixed.
Chapters 26 and 27 cover speed and scale: speculative decoding, and serving
many requests with one model. Chapters 28 and 29 cover small models that work
for a large one: classifiers and probes that guard it, and distillation.
Chapters 30 and 31 look inside current models: rotary positions, and pictures
as input. Chapters 32 to 34 take the agent further: helpers with their own
contexts, training on its own filtered answers, and the system of chapter 19
rebuilt with every part added since.

By the end you can re-implement every piece of a minimal local AI system and
you know what each component of a full-size one is doing.

## Read the course

Open `docs/index.html` in a browser. It needs no server and no internet
connection.

On a phone, open the site and use the browser's "Add to Home Screen": it
installs like an app. "Keep the whole course on this device", on the contents
page, saves every chapter (about 4 MB) for reading without a network. Code
listings fold and have an outline view, and figures open to fill the screen.

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
| 13 | Staying inside the lines | Can you make a broken output format impossible instead of unlikely? |
| 14 | Knowing whether it works | Where did the agent go wrong, and did your change make it better? |
| 15 | The program around the model | What is a harness such as Claude Code or Pi made of? |
| 16 | Spending the window | What does each agent step cost, and what do you do when the context is full? |
| 17 | Text that gives orders | What happens when text the agent reads contains instructions? |
| 18 | The machine underneath | Why a GPU, and what decides your tokens per second? |
| 19 | Build the whole thing | Can you build every part yourself and run it as an agent? |
| 20 | Many experts, few at work | How can a model own far more weights than it uses for a token? |
| 21 | Room to think | How can a model answer a question that needs more work than one token allows? |
| 22 | Where the weights live | Which memory delivers the weights, and what happens when the model is too big for it? |
| 23 | As small as it goes | How much of the agent's 432 KB does the job need? |
| 24 | From text to behaviour | How is a model that continues any text turned into an assistant that behaves one way? |
| 25 | Better without retraining | With the weights fixed, what can be done at the moment of asking to get a better answer? |
| 26 | A quick draft, a careful check | Can a large model write several tokens per pass without writing anything different? |
| 27 | Many people at once | Does each of a hundred users have to wait for the ninety-nine others? |
| 28 | A second model that says yes or no | What catches the request that no rule foresaw? |
| 29 | A large model teaches a small one | Can a small model be taught by a large one instead of by the text? |
| 30 | Positions that turn | How do current models know where a token is, without a table of positions? |
| 31 | Pictures as tokens | A model reads tokens. How does a picture get in? |
| 32 | Many hands | What is gained, and what is risked, when an agent hands parts of a job to other agents? |
| 33 | Learning from its own answers | Can a model get better by training on what it wrote itself? |
| 34 | The whole thing, again | What does the system of chapter 19 look like with everything added since? |

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
python code/ch13_constrain.py          # mask the scores so the output always fits a format
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

Chapter 14 records and grades that agent. Its two scripts take `--mock` to
start the fake server themselves; without it they use your real server:

```bash
python code/ch14_trace.py --mock "What is (3 + 4) * 12?"   # every model call and tool call, as a trace
python code/ch14_eval.py --mock                            # nine tasks, graded, with a standard error
```

Chapters 15 to 19 build the harness, measure the hardware and assemble the
whole system:

```bash
python code/ch15_harness.py --mock     # a four-tool coding harness, on a small demo project
python code/ch16_context.py            # prefix cache on our model; cost of the recorded session
python code/ch17_guard.py              # prompt injection, with the gate off and on
python code/ch18_hardware.py           # loop vs block, batching, memory speed on your CPU
python code/ch19_system.py --train     # ~5 min: train a tiny tool-calling model from scratch
python code/ch19_system.py "what is 23 times 45?"
```

Chapters 20 and 21 train small models to measure two full-size ideas:

```bash
python code/ch20_moe.py                # ~8 min: three models, with and without experts
python code/ch21_think.py --train      # ~3 min: one model answers at once, one writes its steps
python code/ch21_think.py 101100111010 # ask both
```

Chapters 22 and 23 fit the model to the machine:

```bash
python code/ch22_memory.py             # cache, RAM, split models, unified memory, KV cache, mapping
python code/ch23_tiny.py --train       # ~7 min: the chapter 19 agent with three-valued weights
python code/ch23_tiny.py               # grade it on all 3,000 held-out questions
python code/tiny_agent.py "what is 23 times 45?"   # run it with the standard library only
```

Chapter 24 shapes a model's behaviour after pre-training:

```bash
python code/ch24_posttrain.py --train  # ~10 min: pre-train, fine-tune on examples, reinforce from scores
python code/ch24_posttrain.py          # what the model does after each stage
python code/ch25_infer.py              # ~5 min: the same model, improved at inference only
python code/ch26_draft.py              # ~3 min: a small model drafts, the large one checks
python code/ch27_serve.py              # ~2 min: batches, and a queue of requests
python code/ch28_classify.py           # ~2 min: a rule, a classifier and a probe as guards
python code/ch29_distill.py            # ~10 min: a small model taught by text, or by a large model
```

Chapters 30 to 34 look inside current models and take the agent further.
Chapters 32 to 34 need the models of chapter 24, and chapter 34 the classifier
of chapter 28:

```bash
python code/ch30_rotary.py --train     # ~40 min: the model of chapter 7 with rotary positions, and with none
python code/ch30_rotary.py             # their losses at windows of 32 to 256
python code/ch31_images.py --train     # ~6 min: an agent that reads its first number from a picture
python code/ch31_images.py             # how well it reads as pixels are flipped
python code/ch32_helpers.py            # ~5 min: a lead and its helpers on jobs of one to six parts
python code/ch33_selftrain.py          # ~8 min: four rounds of training on its own answers, three ways
python code/ch34_system2.py            # ~4 min: the whole system, one part switched on at a time
```

`ch15_harness.py` runs real shell commands when used with a real model. Use a
folder you can afford to lose and read chapter 17 first.

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
  ch13_constrain.py         constrained decoding: a format as a mask on the scores
  ch14_trace.py             spans and traces around the agent loop
  ch14_eval.py              tasks, graders, pass rate and standard error
  ch15_harness.py           a coding harness: read, write, edit, bash, and the loop
  ch16_context.py           prefix cache, trimming and compaction
  ch17_guard.py             prompt-injection runs and a permission gate
  ch18_hardware.py          CPU measurements: loops vs blocks, batching, memory speed
  ch19_system.py            the whole system: data, training, harness, trace, evaluation
  ch20_moe.py               mixture of experts: router, top-2 experts, balance, sparse path
  ch21_think.py             thinking tokens: the same task asked directly and step by step
  ch22_memory.py            where weights live: cache ladder, split models, KV cache size, mapping
  ch23_tiny.py              three-valued weights trained in place, packed five to a byte, graded
  tiny_agent.py             the packed model and its tool loop with no numpy (ch 23)
  ch24_posttrain.py         pre-training, supervised fine-tuning and reinforcement from a grader
  ch25_infer.py             inference with fixed weights: vote, check and retry, lean on a score, filter
  ch26_draft.py             speculative decoding: draft, check in one pass, keep; the sampling rule
  ch27_serve.py             serving: time of a batched pass, a memory-bound matrix, a queue simulation
  ch28_classify.py          guards: a keyword rule, a small classifier, a probe on activations, a cascade
  ch29_distill.py           distillation: a student trained on the teacher's probabilities
  ch30_rotary.py            rotary positions: the GPT of chapter 6 with turned queries and keys, no table
  ch31_images.py            pictures as input: 15 pixels times a matrix, added where a token would be
  ch32_helpers.py           sub-agents: a lead splits a job, each helper has a context of its own
  ch33_selftrain.py         self-training: answer, filter, train on what was kept, repeat
  ch34_system2.py           the system again: guard, trained agent, check and filter, switched on in turn
  mock_server.py            fake model server for testing without a model
data/
  shakespeare.txt           training text (public domain, 1.1 MB)
  notes.md                  short course notes, used as documents by ch 11
docs/                       the course website (plain HTML, CSS and JavaScript)
  assets/micro.js           helpers and a full GPT forward pass in JavaScript
  assets/weights.js         the trained weights, float16
  assets/data.js            training log, embedding map, text excerpts
  assets/evals.js           the recorded evaluation and its traces (ch 14)
  assets/harness.js         the recorded harness session (ch 15)
  assets/guard.js           the recorded injection runs (ch 17)
  assets/agent.js           the tool-calling model trained in chapter 19, float16
  assets/moe.js             losses and routing measured in chapter 20
  assets/think.js           the two models trained in chapter 21, float16
  assets/tiny.js            the packed model of chapter 23, byte for byte
  assets/post.js            the model after each training stage of chapter 24, float16
  assets/draft.js           the small draft model of chapter 26, float16
  assets/clf.js             the classifier of chapter 28 and its test requests
  assets/eyes.js            the agent of chapter 31 that reads pictures, float16
  assets/selftrain.js       the rounds of self-training measured in chapter 33
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
python code/ch14_eval.py --mock --save   # optional: re-record the evaluation shown in chapter 14
python code/ch15_harness.py --mock --save   # optional: re-record the session of chapters 15 and 16
python code/ch17_guard.py --save         # optional: re-record the runs of chapter 17
python code/ch19_system.py --train       # optional: retrain the agent of chapter 19
python code/ch20_moe.py                  # optional: retrain and re-measure chapter 20
python code/ch21_think.py --train        # optional: retrain the two models of chapter 21
python code/ch22_memory.py --save        # optional: re-measure the machine for chapter 22
python code/ch23_tiny.py --train         # optional: retrain the packed model of chapter 23
python code/ch24_posttrain.py --train    # optional: rerun the three training stages of chapter 24
python code/ch26_draft.py --train        # optional: retrain the draft models of chapter 26
python code/ch31_images.py --train       # optional: retrain the agent of chapter 31
python code/ch33_selftrain.py            # optional: rerun the self-training of chapter 33
python tools/export_web.py    # weights and logs -> docs/assets
python tools/build_site.py    # code snippets -> docs/*.html
node tools/test_parity.js     # browser model == Python model?
```

To train on your own text, replace `data/shakespeare.txt` with any plain-text
file of a few hundred kilobytes or more and run the commands above. A few
sentences in chapters 4, 5, 7, 9 and 13 describe what this particular model learned
and quote its measured numbers; after retraining, check them against yours.
Chapters 20 to 34 quote the losses, routing shares, speeds and scores of their
own runs; after retraining or re-measuring those, update the tables in the pages.

## How this was checked

- `ch03_gradient_descent.py` compares backpropagation with finite differences
  on a small GPT; they agree to about 1e-9.
- `ch08_sample.py` asserts that the KV-cache path gives the same outputs as
  recomputing everything.
- `ch13_constrain.py` asserts that every masked sample fits its format, and
  checks that each format fits the context window and the model's vocabulary.
- `tools/test_parity.js` confirms that the JavaScript model in the browser
  reproduces the Python model's outputs and held-out loss.
- Every page was loaded in a headless browser at phone and desktop widths with
  no script errors, and each experiment was exercised.
- `ch16_context.py` asserts that the prefix cache gives the same scores as a
  computation from scratch. `ch19_system.py --eval` grades the trained agent on
  number pairs held out of its training data.
- `ch20_moe.py` asserts that running only the chosen experts gives the same
  output as running all of them and weighting the rest by zero. Its three
  models were trained once, on the same batches. The mixture did not beat the
  small plain model at this size, and the chapter says so.
- `ch21_think.py` tests both models on 400 of the 410 rows kept out of
  training. Both models were also run on all 4,096 rows: the direct one answers
  "even" every time, the other writes the correct marks every time. One run,
  one seed each.
- The sizes of full-size models quoted in chapter 20 are their publishers'
  figures, not measurements.
- `ch22_memory.py` measured the cache ladder and the file mapping on one
  machine, once. The memory sizes and bandwidths of graphics cards and
  unified-memory machines in chapter 22 are their makers' figures, and the
  tokens-per-second ceilings are arithmetic on those figures, not measurements.
- `ch23_tiny.py` grades every model on all 3,000 held-out questions (1,000
  number pairs times three operations), not a sample. The packed file is read
  back before grading, so the score is that of the bytes on disk.
  `ch23_tiny.py --check` asks the same 3,000 questions through
  `tiny_agent.py`, which uses no numpy, and the page was checked to give the
  same answers in a browser. Every row of the chapter's tables is one training
  run with one seed.
- `ch24_posttrain.py` counts behaviours over 300 sampled answers per model,
  on number pairs kept out of all three stages. Each stage was run once. The
  figures about full-size training in chapter 24 are from the published papers
  it names, not measurements.
- `ch25_infer.py` grades every way of asking on the same 300 unseen sums,
  with the pre-trained model of chapter 24 and no change to its weights. Each
  row is one pass with one seed.
- `ch26_draft.py` compares the text written with drafts against the large
  model's own text for 100 starting points at every setting, and they were
  identical in all of them. Its full-size table is arithmetic, not measurement.
- `ch27_serve.py` measures pass times on one core. Its queue is a simulation
  driven by those times, with invented requests.
- `ch28_classify.py` tests its three guards on 288 requests whose wordings
  appear nowhere in the 288 they learned from. The figures about deployed
  classifiers in chapter 28 are their maker's published ones.
- `ch29_distill.py` trains each pair of students from the same starting
  weights on the same batches. Four trainings, one seed each.
- `ch30_rotary.py` trains its two models on the batches and settings of
  `train.py`, and scores all three on the last 32 characters of the same
  windows of unseen text. One training each; differences under 0.03 are noise.
- `ch31_images.py` is tested on 300 sums with number pairs absent from
  training, and once more with the pixels set to zero, which leaves 5 right.
- `ch32_helpers.py` runs 200 jobs per row. `ch33_selftrain.py` runs each rule
  once from the same starting model. `ch34_system2.py` sends the same 600
  requests to every version of the system. One seed each.
- The harness and injection runs of chapters 15 and 17 use a scripted model in
  `mock_server.py`; the pages say so. The GPU figures in chapter 18 are published
  specifications, not measurements.
- The chat, retrieval, agent, tracing and evaluation scripts were run end to end against
  `mock_server.py`. They follow the OpenAI-compatible API that Ollama
  documents, but **they have not been run against a real Ollama, llama.cpp or
  LM Studio server**. If one behaves differently, the fix belongs in
  `code/llm.py`.

## Look and lesson format

The pages use the visual language of Painted Homer (ivory paper, ultramarine
ink, a brush stroke under each heading, icons made of one dab and one ink
gesture, small quiet controls) and the lesson
rules that go with it: mobile first, one idea per step, a guess before the
reveal, a short self-check, and one line to keep. The guess and the line to
keep for every chapter live in one table, `M.lesson` in `docs/assets/micro.js`.
The score of your guesses is stored in your browser and nowhere else.

Fonts: Bodoni Moda and Libre Franklin, both under the SIL Open Font License
(`docs/assets/fonts/`), subset to Latin.

## Scope

This course covers the path from next-token prediction to a working local
agent: the model, its training, inference, the harness around it, the hardware
under it, and how to check and bound the result. Chapter 19 assembles all of
it into one small system. Chapters 20 and 21 then show, at toy size, two
things full-size models add: experts and thinking. Chapters 22 and 23 cover
where a model sits in a machine and how small a working one can be made.
Chapters 24 to 33 add training for behaviour, inference methods, serving,
guards, distillation, current architecture, pictures, helpers and
self-training, and chapter 34 assembles the system a second time.

Not covered yet, in rough order of how much an agent builder would miss them:

- the methods used for post-training at scale beyond the plain policy gradient
  of chapter 24, reward models trained on human rankings, and how alignment is
  evaluated
- planning, and how models are trained to produce useful thinking; chapter 21
  shows thinking tokens with the steps given in advance
- the Model Context Protocol in detail; a lead that is itself a model and
  plans the split, where chapter 32 uses plain code
- serving beyond one machine: several servers, routing between them, cost per
  token; chapter 27 stops at one server
- data collection and cleaning, scaling laws, other architecture variants
  (state-space models; chapter 30 trains rotary positions and only describes
  the other changes inside current models)
- models that write by denoising instead of one token at a time
- pruning, and distillation beyond the plain form of chapter 29
- neural processing units, servers with many graphics cards, and the cost of
  power and cooling; chapter 22 stops at one machine
- images beyond the 15-pixel digits of chapter 31, sound, and models that
  produce pictures; the benchmarking of models themselves
- learning that continues during use, beyond the rounds of chapter 33

## Further study

- Stanford CS336, *Language Modeling from Scratch* (tokenizer to systems, GPUs, inference): <https://cs336.stanford.edu>
- Stanford CS329Z, *Engineering AI Agents* (tools, memory, harness, evaluation, safety): <https://cs329z.stanford.edu/>
- Hugging Face Agents Course: <https://huggingface.co/learn/agents-course>
- Pi, a minimal open-source coding harness with four default tools: <https://pi.dev>
- Claude Code tools reference: <https://code.claude.com/docs/en/tools-reference>
- KV cache reuse in llama.cpp's server: <https://github.com/ggml-org/llama.cpp/discussions/13606>
- Qwen documentation on its chat template and tool-call format: <https://qwen.readthedocs.io/en/latest/getting_started/concepts.html>

## References

- Vaswani et al., *Attention Is All You Need*, 2017: <https://arxiv.org/abs/1706.03762>
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021: <https://arxiv.org/abs/2106.09685>
- llama.cpp (GGUF, quantization formats and their bits per weight): <https://github.com/ggml-org/llama.cpp>
- Ollama, OpenAI-compatible API: <https://docs.ollama.com/api/openai-compatibility>
- Training text: the "tiny Shakespeare" file from <https://github.com/karpathy/char-rnn>

Facts about tools and file formats were checked in November 2026. Model names
and tool versions change quickly; the arithmetic does not.

## License

MIT. See `LICENSE`.
