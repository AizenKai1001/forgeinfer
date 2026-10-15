# The ForgeInfer engine

`forge` is one binary with many subcommands; `forge serve` is the one most people want: an OpenAI-compatible server for a GGUF model on an NVIDIA GPU.

```
forge serve <model.gguf> [--native | --q8 | --f16] [--tokenizer FILE] [--host H] [--port P] [--ctx N] [--alias NAME]
                         [--chat chatml|smollm2|llama3|qwen3|qwen35|qwen35-think] [--api-key KEY] [--queue N]
                         [--temp T] [--top-k K] [--top-p P] [--min-p P] [--repeat-penalty R] [--device D]
                         [--mtp DIR|FILE] [--no-speculate]
                         [--kv fp16|int8|int4|k8v4] [--depth N] [--chunk N] [--threads N] [--host-ffn LAYERS|all] [--vram-reserve MIB]
                         [--prompt-cache-mb MB] [--embedding-model FILE] [--embedding-device cpu|gpu|auto] [--embedding-alias NAME]
forge serve <embedding-model.gguf> [--port P] [--alias NAME] [--device D] [--embedding-device cpu|gpu|auto]   # an embedding model on its own
forge embed <embedding-model.gguf> TEXT...                                   # the cosines between texts, `--json` for the vectors, `--cpu` / `--gpu`
```

| flag | meaning |
|---|---|
| (a `.gguf` model) | its **tokenizer and chat template are read from the file**; `--tokenizer` and `--chat` override them. The chat format is picked from the template text (Llama 3 headers, SmolLM2, ChatML, Qwen) |
| `--native` / `--q8` / `--f16` | how weights live on the GPU. For a GGUF the default is `--native`: quantized blocks are read as they are, with no conversion (the smallest memory and the fastest decode). `--f16` dequantizes to half precision (4x the memory); `--q8` re-quantizes our own full-precision checkpoints |
| `--ctx N\|auto` | context length to allocate (default 4096). `auto`: the longest context that fits the memory left next to the weights, at most what the model was trained for (`FORGE_CTX_MAX` caps it). The engine prints what it opened (`[engine] context 234496 ...`) and the model list reports it (`meta.n_ctx`). With `--kv int4` the 4B Qwen3.5 opens 234,000 positions on an 8 GB card that another program already holds 0.9 GB of (its full 262,144 is within the planner's numbers when nothing else holds video memory); the 9B opens 60,000 |
| `--kv` | KV-cache storage: `fp16` (default), `int8` (a little over half the memory, no measurable loss), `k8v4` (8-bit keys, 4-bit values: 40 percent of fp16, a small loss), `int4` (28 percent, a visible loss on some models, a dense Qwen3-4B among them), `k4v8`; the numbers are in [PERFORMANCE.md](PERFORMANCE.md#long-contexts) |
| `--depth N` | speculative decoding with the model's own prediction head (Qwen3.5): tokens drafted per step, 2 or 3 is fastest |
| `--chunk N` | prompt tokens per prefill pass (lower it to save memory) |
| `--threads N` | CPU threads for the feed-forward layers kept in system memory (default: cores minus one) |
| `--host-ffn LAYERS\|all` | force layers' feed-forward weights into system memory (the default planner does this automatically when the model does not fit the card) |
| `--vram-reserve MIB` | leave this much video memory free for other programs |
| `--prompt-cache-mb MB` | host memory for the chats the GPU is not working on (default 2048, 0 = off): going back to a long chat copies it back instead of reading it again (see below) |
| `--embedding-model FILE` | an embedding model (nomic-embed-text-v1.5, bge, MiniLM, e5, gte) served on `/v1/embeddings` next to the chat model, on the CPU unless `--embedding-device auto` (loaded by the first request: on the GPU when 700 MB of video memory are free then, else on the CPU; nothing is spent until a request comes) or `gpu` (on the GPU for sure, with 450 MB left free for it by the automatic context); the chat engine plans its memory around the whole card, so the default leaves the card alone. `forge serve nomic-embed-text.gguf` alone serves only the embeddings, on the GPU when there is one. See [EMBEDDINGS.md](EMBEDDINGS.md) |
| `--api-key KEY`, `--host`, `--port` | access control; the default host is `127.0.0.1` |
| `--mtp FILE` | the Qwen3.5 prediction head (`mtp.safetensors`, not part of the GGUF files; `tools/fetch_mtp.py REPO OUT` extracts it from the original checkpoint with range requests); picked up automatically when a file of that name sits next to the model. With `--depth 3` the 9B answers about 2 times faster ([PERFORMANCE.md](PERFORMANCE.md#speculative-decoding-on-the-rtx-4060)) |

## HTTP API

```
GET  /health, /v1/health        {"status": "ok"}
GET  /v1/models, /props         the served model
POST /v1/chat/completions       messages, tools, streaming (SSE) with reasoning_content and tool_calls deltas, usage and timings
POST /v1/completions            raw text completion
POST /v1/embeddings             input (a string or a list), dimensions, encoding_format float|base64: the vectors of an embedding model (see EMBEDDINGS.md)
POST /apply-template            the rendered prompt of a chat request
POST /tokenize, /detokenize
```

**Structured output**: `response_format` makes the answer follow a grammar by construction. `{"type": "json_object"}` gives a JSON object; `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}` (what the OpenAI SDKs, LangChain and LM Studio send for structured output) gives a document that follows the schema. The grammar is checked byte by byte (a token that spells several structural characters is judged as a whole), the candidates that would break it are removed before every draw, the generation ends when the top-level value is closed, and a reasoning block is switched off (nothing can come before the `{`). It composes with sampling, streaming, the prediction head and the prompt lookup (a draft that breaks the grammar is just a draft that differs).

The schema subset: `type` (object, array, string, number, integer, boolean, null, or a list of them), `properties` / `required` / `additionalProperties`, `items` / `minItems` / `maxItems`, `enum` and `const`, `anyOf` / `oneOf` (the first alternative whose first byte fits wins; alternatives that start alike are not told apart), `$ref` to `#`, `#/$defs/x` and `#/definitions/x` (recursive schemas work) and `true` for "anything". Properties come in the order the schema lists them, optional ones may be left out but a required one may not be skipped, and a key that is not declared is refused unless `additionalProperties` allows it (absent means no, as in OpenAI's strict mode; an object without `properties` is free-form). Not checked: `pattern`, `format`, `minLength` / `maxLength`, `minimum` / `maximum`, `uniqueItems`; `allOf` is refused. A schema that cannot be followed, or another `response_format` type, is a 400 with the reason instead of unchecked JSON. Measured on the 4B and the 9B with its head: 16 requests each over three schemas (a person, three products with exactly three items, an enum classification), greedy and sampled, streamed and not: every answer followed its schema (the ones cut by `max_tokens` aside).

Request fields understood: `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`, `stop`, `max_tokens` / `max_completion_tokens`, `stream`, `stream_options.include_usage`, `tools`, and `chat_template_kwargs.enable_thinking` (or `reasoning_effort: "none"`) for thinking models. A streamed answer sends the assistant role first, `reasoning_content` deltas for thinking text, then `content`, a final chunk with the finish reason, a usage chunk with `timings` (prompt and predicted tokens per second), and `[DONE]`. `return_progress: true` (llama.cpp's field) adds chunks with an empty delta and `prompt_progress` `{total, cache, processed, time_ms}` while a long prompt is processed, one after every 1,024 tokens and before the first token (processed / total is the share of the prompt that is in the cache, `cache` the tokens that were there from the request before, `time_ms` the time spent so far); Forge Studio and the chat page show it as a bar. Closing the connection cancels the generation. Requests are queued (`--queue`); a full queue answers 429.

## What the server protects against

- **Web pages and DNS rebinding.** A server that listens on a loopback address (the default, and what Forge Studio starts) is reachable by every web page the user opens: a script can post to `http://127.0.0.1:1234`, and a hostile host name can be made to resolve to it. It therefore answers only requests whose `Host` header names a local host and whose `Origin` header, when there is one, is local too (HTTP 403 otherwise); curl, the OpenAI SDKs and Studio send no foreign origin. `FORGE_ALLOWED_HOSTS=name,name` adds names (a reverse proxy that passes its public name on), `*` turns the check off. Servers on another address (`--host 0.0.0.0`) are not checked: use `--api-key` there.
- **Special tokens in messages.** The markup of a chat turn belongs to the template. Text inside a message that spells a special token (`<|im_end|>`, `<|im_start|>system`...) is defused before the template is rendered (a zero-width space after its first character), so that a pasted document or a tool's web page cannot end the user's turn or open a system turn: the tokenizers of Hugging Face and llama.cpp read such text as the token. `FORGE_PARSE_SPECIAL_IN_MESSAGES=1` restores that; raw `/v1/completions` prompts are never touched. This keeps the structure of the conversation the template's; it does not make a model ignore an instruction that is written as plain text in a document (`tools/injection_check.py`: the 4B answers PWNED with and without the fake turns).
- **Tokenizer parity.** `tools/tokenizer_parity.py` compares `forge tokenize` with llama.cpp's `llama-tokenize` on texts with digits, runs of white space, contractions, URLs, code in four languages, JSON, HTML, Markdown, accents, CJK, Cyrillic, Arabic, Hebrew, Thai, Devanagari, emoji with joiners and flags, math symbols, 1,000-character words and special-token look-alikes: 43 of 45 texts give the same ids in each of the Qwen3.5, Qwen2.5 and Qwen3 vocabularies. The two that differ are on purpose: ForgeInfer applies the NFC normalizer that Qwen's `tokenizer.json` declares (the tokenizer the models were trained with; llama.cpp does not for GGUF byte-level BPE models, so a decomposed letter such as e + U+0301 is one token here and two there), and `forge tokenize` reads `<|im_start|>` in a text as the special token (llama.cpp's `--no-parse-special` does not).
- **Slow and oversized requests.** The head of a request must arrive within 30 seconds and its body within 120 more (HTTP 408), bodies are limited to 8 MiB (413, also for chunked bodies, whose declared sizes cannot wrap the limit), at most 256 connections are served at once, the API key is compared in constant time, and JSON nests at most 512 levels deep.

## Other subcommands

```
forge info model.gguf                       metadata and tensor table
forge tensor model.gguf NAME [R0 NR C0 NC]  statistics and a block of one weight matrix (dequantized), as JSON
forge tokenize <tokenizer.json | model.gguf> TEXT
forge gguf-stitch BASE RECIPE OUT           a GGUF built from tensors of several quantizations of the same model (no re-quantization)
forge quantize | imatrix | hessian | ppl | mtp-agree      the quantizer and its measurement tools
forge eval-hellaswag | eval-mc | eval-ppl | eval-lambada   evaluation
forge train | tok-train | tok-encode | chat-mask | export-hf   the trainer and its data tools (docs/ARCHITECTURE.md)
forge run | chat | gpu-bench | prefill-bench | kbench     smaller tools
```

## How it works

```
GGUF file ──► forge-gguf   mmap, metadata, tensor table, bit-exact decoders for the quantization formats
            ├► forge-tok   byte-level BPE built from the file's own vocabulary and merges, NFC, pre-tokenizers (GPT-2, SmolLM, Qwen2/3, Qwen3.5, Llama 3)
            ├► forge-chat  chat templates (ChatML, SmolLM2, Qwen3, Qwen3.5, Llama 3) and the incremental parser of reasoning and tool calls
            └► forge-model configuration and weights (llama, qwen2, qwen3, qwen35), the f32 reference forward pass (the oracle for the GPU path), the sampler
forge-gpu     weights on the device, the fused decode step captured as CUDA graphs, tensor-core GEMM prefill, flash attention, KV-cache compression,
              the host-FFN planner and the AVX2 kernels for layers that live in system memory
forge-cuda    the CUDA driver API loaded at run time (nvcuda.dll / libcuda.so): no CUDA runtime library is needed to run
forge-kernels CUDA C++ kernels compiled by nvcc in build.rs into cubins (sm_75, 8.x) and PTX for newer GPUs, embedded in the binary
forge-engine  one model on the GPU behind a request queue: prompt rendering, prefix reuse and state snapshots, sampling, speculation
forge-serve   the HTTP server (std only, one thread per connection)
```

### What the engine does about memory

- **Quantized weights stay quantized.** The decode kernels read Q4_K / Q5_K / Q6_K / Q8_0 blocks directly (pre-quantized activations, dot products in integer arithmetic), so a 4-bit model uses about 4.5 bits per weight on the card.
- **KV-cache compression** (`--kv`): keys and values are stored as 8-bit or 4-bit groups with per-group scales, inside the attention kernels. The decode attention splits the context into blocks of 1,024 positions (an exact merge follows) and, with a long session, runs one block per key-value head for all the query heads that share it, reading a compressed cache in place; the numbers and the quality table are in [PERFORMANCE.md](PERFORMANCE.md#long-contexts).
- **CPU/GPU split** for models bigger than the card: the planner keeps attention, the recurrent layers and the output head on the GPU and moves the feed-forward layers that do not fit into page-locked system memory (the embedding table of a model with a separate output matrix is always there: one row of it is read per token, which costs nothing measurable and gives the card 0.3 to 0.6 GiB back; `FORGE_HOST_EMBED=0` keeps it on the device); AVX2 kernels multiply them *inside the CUDA graph* while the GPU works on its own layers. Decode of such a model is limited by system-memory bandwidth, so `docs/PERFORMANCE.md` is mostly about reading fewer bytes per token.
- **Speculative decoding** with the multi-token-prediction head that Qwen3.5 models ship: the head drafts 2 or 3 tokens, the main model verifies them in one pass over the weights (one sweep reads the weights once instead of three times, which is why it helps most when decode is bandwidth-bound).
- **Prompt lookup** (any model, on by default): tokens that repeat earlier text are drafted from the text itself and verified in one pass, up to 2x on code edits, quotes and structured output; a guard and a memory check keep it out of the way elsewhere (`docs/PERFORMANCE.md`, `--no-lookup`).
- **Chat caching**: the prompt prefix of the previous turn is reused; the recurrent state of hybrid models is snapshotted at the message boundary (right after the assistant header) so a follow-up turn only processes the new text.
- **Prompt cache** (`--prompt-cache-mb`, `FORGE_PROMPT_CACHE_MB`, default 2,048): the engine holds one conversation on the GPU, and a request that shares little with it used to throw it away, so a long document discussed in one chat had to be read again when the user came back from another chat. Now, when a request is about to discard most of a conversation of at least 1,024 tokens, the KV entries of its positions (and a hybrid model's recurrent state at its last message boundary) are copied to page-locked host memory, and a later request that continues it brings them back over PCIe (12 GB/s: 6,400 tokens of the hybrid 4B are 275 MB, 25 ms; of the dense 4B 1 GB, 75 ms). The least recently stored conversations go first. A dense model resumes from the longest prefix a stored conversation shares with the prompt; a hybrid model needs the stored tokens to be exactly a prefix of the prompt (its state summarises exactly those). Locking a gigabyte of host memory takes 0.3 s and unlocking it as long, so one buffer is kept spare (it counts against the budget) and is sized an eighth over what it holds, so the buffer a chat came back from holds the next version of the other chat. The round trip is bit-identical (`crates/forge-gpu/tests/kv_blob.rs`: logits and the greedy tokens after the detour equal those of a straight run, for a dense and a hybrid model, with the fp16 and the int8 cache) and `tools/cache_check.py` replays two long chats that take turns with the cache on and compares the answers with those of the same chats one after the other (the engine's own reuse of the previous turn); `tools/cache_soak.py` visits three or four chats in a random order.

## Recipes: mixing quantizations without re-quantizing

A *recipe* takes some tensors from another GGUF of the **same model** (another quantization) and leaves the rest of the base file alone. It is model surgery at the container level: no weights are re-quantized, so there is no new rounding error.

```json
{"overrides": [
  {"from": "Qwen3.5-27B-Q3_K_M.gguf", "tensors": ["w_down"], "if_dtype": ["Q4_K"]},
  {"from": "Qwen3.5-27B-Q3_K_M.gguf", "tensors": ["wqkv", "w_out"], "if_dtype": ["Q4_K"]}
]}
```

- `from`: the donor file. `tensors`: canonical names (`w_gate w_up w_down wq wk wv wo wqkv wz wa wb w_out tok_emb lm_head`). `layers` (optional, e.g. `"0-7,60-63"`): default all. `if_dtype` (optional): take the tensor only when the donor stores it in one of these formats.
- The example above makes the 27B file smaller and faster: the donor (a Q3_K_M file) stores the down projections in Q4_K where the Q4_K_M base uses Q6_K, so those layers get cheaper; the attention and output projections that live on the GPU shrink too, which lets three more feed-forward layers fit on an 8 GB card.
- Use it without writing a file: `FORGE_RECIPE=recipe.json forge serve base.gguf`. Or write a normal GGUF that also loads in llama.cpp and LM Studio: `forge gguf-stitch base.gguf recipe.json out.gguf`.
- Measure what you did: `forge ppl` computes the KL divergence of the new file against a reference on text of your choice ([PERFORMANCE.md](PERFORMANCE.md) shows the 27B numbers).

## Environment variables

Every switch the runtime reads from the environment (most also exist as flags above) is listed in [ENVIRONMENT.md](ENVIRONMENT.md).
