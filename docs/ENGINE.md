# The ForgeInfer engine

`forge` is one binary with many subcommands; `forge serve` is the one most people want: an OpenAI-compatible server for a GGUF model on an NVIDIA GPU.

```
forge serve <model.gguf> [--native | --q8 | --f16] [--tokenizer FILE] [--host H] [--port P] [--ctx N] [--alias NAME]
                         [--chat chatml|smollm2|llama3|qwen3|qwen35|qwen35-think] [--api-key KEY] [--queue N]
                         [--temp T] [--top-k K] [--top-p P] [--min-p P] [--repeat-penalty R] [--device D]
                         [--mtp DIR|FILE] [--no-speculate]
                         [--kv fp16|int8|int4|k8v4] [--depth N] [--chunk N] [--threads N] [--host-ffn LAYERS|all] [--vram-reserve MIB]
```

| flag | meaning |
|---|---|
| (a `.gguf` model) | its **tokenizer and chat template are read from the file**; `--tokenizer` and `--chat` override them. The chat format is picked from the template text (Llama 3 headers, SmolLM2, ChatML, Qwen) |
| `--native` / `--q8` / `--f16` | how weights live on the GPU. For a GGUF the default is `--native`: quantized blocks are read as they are, with no conversion (the smallest memory and the fastest decode). `--f16` dequantizes to half precision (4x the memory); `--q8` re-quantizes our own full-precision checkpoints |
| `--ctx N` | context length to allocate (default 4096) |
| `--kv` | KV-cache storage: `fp16` (default), `int8` (half the memory, negligible loss), `int4`, `k8v4` |
| `--depth N` | speculative decoding with the model's own prediction head (Qwen3.5): tokens drafted per step, 2 or 3 is fastest |
| `--chunk N` | prompt tokens per prefill pass (lower it to save memory) |
| `--threads N` | CPU threads for the feed-forward layers kept in system memory (default: cores minus one) |
| `--host-ffn LAYERS\|all` | force layers' feed-forward weights into system memory (the default planner does this automatically when the model does not fit the card) |
| `--vram-reserve MIB` | leave this much video memory free for other programs |
| `--api-key KEY`, `--host`, `--port` | access control; the default host is `127.0.0.1` |
| `--mtp FILE` | the Qwen3.5 prediction head (`mtp.safetensors`, not part of the GGUF files; `tools/fetch_mtp.py REPO OUT` extracts it from the original checkpoint with range requests); picked up automatically when a file of that name sits next to the model. With `--depth 3` the 9B answers about 2 times faster ([PERFORMANCE.md](PERFORMANCE.md#speculative-decoding-on-the-rtx-4060)) |

## HTTP API

```
GET  /health, /v1/health        {"status": "ok"}
GET  /v1/models, /props         the served model
POST /v1/chat/completions       messages, tools, streaming (SSE) with reasoning_content and tool_calls deltas, usage and timings
POST /v1/completions            raw text completion
POST /apply-template            the rendered prompt of a chat request
POST /tokenize, /detokenize
```

Request fields understood: `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`, `stop`, `max_tokens` / `max_completion_tokens`, `stream`, `stream_options.include_usage`, `tools`, and `chat_template_kwargs.enable_thinking` (or `reasoning_effort: "none"`) for thinking models. A streamed answer sends the assistant role first, `reasoning_content` deltas for thinking text, then `content`, a final chunk with the finish reason, a usage chunk with `timings` (prompt and predicted tokens per second), and `[DONE]`. Closing the connection cancels the generation. Requests are queued (`--queue`); a full queue answers 429.

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
            └► forge-model configuration and weights (llama, qwen3, qwen35), the f32 reference forward pass (the oracle for the GPU path), the sampler
forge-gpu     weights on the device, the fused decode step captured as CUDA graphs, tensor-core GEMM prefill, flash attention, KV-cache compression,
              the host-FFN planner and the AVX2 kernels for layers that live in system memory
forge-cuda    the CUDA driver API loaded at run time (nvcuda.dll / libcuda.so): no CUDA runtime library is needed to run
forge-kernels CUDA C++ kernels compiled by nvcc in build.rs into cubins (sm_75, 8.x) and PTX for newer GPUs, embedded in the binary
forge-engine  one model on the GPU behind a request queue: prompt rendering, prefix reuse and state snapshots, sampling, speculation
forge-serve   the HTTP server (std only, one thread per connection)
```

### What the engine does about memory

- **Quantized weights stay quantized.** The decode kernels read Q4_K / Q5_K / Q6_K / Q8_0 blocks directly (pre-quantized activations, dot products in integer arithmetic), so a 4-bit model uses about 4.5 bits per weight on the card.
- **KV-cache compression** (`--kv`): keys and values are stored as 8-bit or 4-bit groups with per-group scales, inside the attention kernels.
- **CPU/GPU split** for models bigger than the card: the planner keeps attention, the recurrent layers, the embeddings and the output head on the GPU and moves the feed-forward layers that do not fit into page-locked system memory; AVX2 kernels multiply them *inside the CUDA graph* while the GPU works on its own layers. Decode of such a model is limited by system-memory bandwidth, so `docs/PERFORMANCE.md` is mostly about reading fewer bytes per token.
- **Speculative decoding** with the multi-token-prediction head that Qwen3.5 models ship: the head drafts 2 or 3 tokens, the main model verifies them in one pass over the weights (one sweep reads the weights once instead of three times, which is why it helps most when decode is bandwidth-bound).
- **Prompt lookup** (any model, on by default): tokens that repeat earlier text are drafted from the text itself and verified in one pass, up to 2x on code edits, quotes and structured output; a guard and a memory check keep it out of the way elsewhere (`docs/PERFORMANCE.md`, `--no-lookup`).
- **Chat caching**: the prompt prefix of the previous turn is reused; the recurrent state of hybrid models is snapshotted at the message boundary (right after the assistant header) so a follow-up turn only processes the new text.

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
