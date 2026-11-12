# microAI course notes

Short notes, one per chapter. They are the "documents" searched by
code/ch11_rag.py and by the retrieval experiment in chapter 11.

## Next-token prediction and loss
A language model outputs a probability for every possible next token given the text so far. Writing is a loop: get the probabilities, sample one token, append it, repeat. The loss, or cross-entropy, is the average of minus log p of the token that actually came next, measured on text the model has not seen. Uniform guessing over 65 characters costs log 65 = 4.17. A table of counts (an n-gram model) improves with more context only until the table becomes sparse; the best table here reaches 1.68 on unseen text at order 4.

## Tokens and byte-pair encoding
Models read tokens, pieces between characters and words. Byte-pair encoding starts from 256 raw bytes and repeatedly merges the most frequent adjacent pair into a new token. Common words become one token and rare words are spelled from pieces, so any text can be encoded. A typical English token is about four characters or three quarters of a word. Special tokens mark things like the end of a speaker's turn.

## Gradient descent and backpropagation
Training adjusts parameters to lower the loss. The gradient is the slope of the loss for each parameter; each step moves every parameter a little against its slope, scaled by the learning rate. A learning rate that is too large overshoots and diverges. Backpropagation applies the chain rule backward through the computation so that one backward pass yields all the slopes, at about the cost of one forward pass.

## Embeddings
Each token ID is replaced by a learned vector, a row of the embedding table. Similar tokens end up pointing in similar directions because that lowers the loss. The dot product measures how much two vectors point the same way; cosine similarity is the dot product with lengths divided out. A second table gives each position a vector. The output layer reuses the token table: the score of a token is the dot product of the final vector with that token's row.

## Attention
Attention lets each position read from earlier positions. Every token produces a query, a key and a value. Scores are dot products of a query with all keys; a softmax turns the scores into weights that sum to one; the output is the weighted average of the values. Scores are divided by the square root of the dimension because the typical size of a dot product of random vectors grows as the square root of the dimension, and unscaled scores would saturate the softmax and flatten its gradient. A causal mask stops positions from looking at later ones.

## The transformer block
A GPT stacks identical blocks around a residual stream, one vector per position. Each block adds the result of attention, which moves information between positions, and then the result of an MLP, which computes on each position separately. Adding instead of replacing gives the gradient a direct path through any depth. Layer normalisation rescales vectors before each component reads them. Parameters are approximately vocabulary times width plus layers times twelve times width squared.

## Training
The training loop picks random passages, predicts every next character, measures the loss, runs backward and takes a step with the Adam optimiser. The learning rate warms up and then decays, and gradients are clipped. The loss starts at 4.17 and falls quickly. The held-out loss, on text never used for training, reveals overfitting when it rises while the training loss still falls.

## Sampling and the KV cache
Temperature divides the scores before the softmax: low values sharpen the distribution, high values flatten it. Top-k keeps the k most likely tokens; top-p keeps the smallest set whose probabilities add up to p. The KV cache stores the attention keys and values of earlier tokens, which never change because of the causal mask, so each new token costs one row of comparisons instead of recomputing everything. Prompt processing is fast per token; generation is one token at a time.

## Quantization, memory and speed
Quantization stores each weight with fewer bits by rounding to a grid: weight is approximately scale times a small integer. One scale per block of 32 weights costs half a bit per weight, so 4 bits becomes 4.5. Quality holds down to about 4 bits per weight and then drops sharply. Model size in bytes is parameters times bits divided by eight. Tokens per second is bounded by memory bandwidth divided by model size, because every weight is read once per token.

## Running a model locally
A local model is a GGUF file containing quantized weights, the tokenizer and a chat template. A runtime such as llama.cpp performs the forward pass. A server such as Ollama, llama-server or LM Studio exposes it over HTTP on localhost with an OpenAI-style chat completions route. A chat is one long document with special tokens between turns; the model simply continues it and stops at the end-of-turn token. The server keeps no memory, so the program resends the whole conversation every turn.

## Context, retrieval and tools
The context window is the model's only working memory at run time. Retrieval-augmented generation splits documents into chunks, turns each into a vector, finds the chunks whose vectors are nearest to the question by cosine similarity, and pastes them into the prompt. Tool calling lets the model request an action by writing structured text; the program executes it and appends the result. An agent is that exchange inside a loop with a step limit.

## Fine-tuning and LoRA
Fine-tuning continues training on new examples to change a model's habits. LoRA freezes the original weights and trains two thin matrices A and B next to each weight matrix, using W plus A times B. With width d and rank r this trains 2 d r numbers instead of d squared. The adapter can be merged into the weights afterwards, so it adds no cost at inference. Prefer prompting and retrieval first; fine-tune when a behaviour cannot be obtained reliably otherwise.
