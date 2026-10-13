# Models

## What the ForgeInfer engine runs

| | supported |
|---|---|
| architectures (`general.architecture`) | `llama` (Llama 3 family, SmolLM2 and other Llama-shaped models), `qwen3` (dense), `qwen35` (Qwen3.5: hybrid Gated DeltaNet + gated attention, with multi-token-prediction head) |
| tokenizers (`tokenizer.ggml.model`) | `gpt2`, i.e. byte-level BPE: the pre-tokenizer is picked from `tokenizer.ggml.pre` (`gpt2`, `smollm`, `qwen2` / `qwen3`, `qwen35`, `llama-bpe` / `llama3`) |
| quantizations | F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (and mixes of them, e.g. `Q4_K_M`) |
| chat templates | ChatML, SmolLM2, Qwen3, Qwen3.5 (with the thinking switch), Llama 3 (picked from the template text in the file) |
| GPUs | NVIDIA Turing (sm_75) and newer |

Forge Studio's **My models** shows a model's architecture and tokenizer type; **Load** is disabled with the reason when the engine cannot run a file.

## What has been tested

| model | notes |
|---|---|
| SmolLM2-135M-Instruct (f16, Q4_0, Q4_K_M, Q6_K, Q8_0) | decode and chat; tokenizer and templates checked against Hugging Face goldens |
| Qwen3 0.6B, 1.7B, 4B, 4B-Instruct-2507 | decode vs llama.cpp (see [PERFORMANCE.md](PERFORMANCE.md)), chat |
| Qwen3.5 0.8B, 4B, 9B, 27B | greedy output identical to llama.cpp on the 4B and 9B for the first 40 tokens; the 27B runs through the CPU/GPU split on an 8 GB card; speculative decoding with the prediction head |
| tokenizers from the GGUF file | the tokenizer built from a GGUF agrees with Hugging Face's `tokenizer.json` on all 475 recorded cases per model family (SmolLM2, Qwen3, Qwen3.5) and on an extra set of awkward strings (Unicode, digits, code, whitespace, special tokens) |
| Llama 3 | the tokenizer pattern and the chat format are implemented and unit-tested, **not yet checked against real Llama 3 weights** |

Other sizes of these families (for example Qwen3.5-2B) should load, but the author has not run them: reports are welcome.

## What does not run (yet)

- SentencePiece tokenizers: Llama 2, Mistral 7B v0.1, Gemma, Phi-3 (`tokenizer.ggml.model = llama`).
- Other architectures: Qwen2 / Qwen2.5 (needs attention biases), Gemma, Mistral, Phi, DeepSeek, Cohere ... and mixture-of-experts models such as Qwen3-30B-A3B.
- IQ-quantizations (`IQ2_XS`, `IQ3_M`, `IQ4_XS` ...), vision (`mmproj` files are listed but not used), embedding models.

Use Forge Studio's **llama.cpp server** engine for those; it runs everything llama.cpp runs.

## Where to get models

Hugging Face. Search in Forge Studio's **Discover** screen, or browse [huggingface.co/models?library=gguf](https://huggingface.co/models?library=gguf). Repositories by `unsloth`, `bartowski` and `lmstudio-community` are well-known sources of GGUF files. Every model has its own license.

`Q4_K_M` is the usual best choice: about 4.5 to 5 bits per weight and a small quality loss. Go to `Q5_K_M` or `Q6_K` if you have room, to `Q3_K_M` only when you must.
