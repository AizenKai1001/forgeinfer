# Roadmap

What is missing, roughly in the order that would help the most people. Nothing here is promised; it is where help is most welcome.

## Engine

1. **A CPU backend.** Today the engine needs an NVIDIA GPU. A portable CPU path (AVX2 and AVX-512 on x86, NEON on Arm) would run every supported model on any computer and make Forge Studio fully self-contained on macOS and on Linux without CUDA. The pieces exist: the f32 reference forward pass in `forge-model` is the oracle, and `forge-gpu/src/cpuk.rs` already holds hand-vectorized AVX2 kernels for Q2_K to Q8_0 that run inside the GPU graph. What is missing is the whole model on the CPU (the Gated DeltaNet and attention layers, the KV cache, the sampler loop) and NEON kernels.
2. **More architectures**: Qwen2 / Qwen2.5 (attention biases), Gemma, Mistral, Phi, and mixture-of-experts models (Qwen3-30B-A3B is a good first target: with expert offload it is fast on small cards).
3. **A SentencePiece tokenizer** (Llama 2, Mistral, Gemma). The BPE one is built from the vocabulary and merges in the GGUF file; the unigram model is a separate piece of work with its own goldens.
4. **Prefill speed** (0.6 to 0.85 times llama.cpp on the Qwen3.5 4B and 9B at 512 tokens, measured on an RTX 2060 SUPER and an RTX 4060): an INT8 tensor-core GEMM that reads the quantized blocks directly (what llama.cpp's MMQ kernels do) instead of dequantizing into fp16 tiles.
5. **IQ-quantizations** (`IQ2`, `IQ3`, `IQ4_XS`) in the decoders and kernels.
6. **Serving several clients at once**: today requests are queued; continuous batching would let a few chats share one pass over the weights.
7. **Vulkan or Metal** back ends. A long shot; the CPU backend comes first.

## Forge Studio

- Image input for vision models (the `mmproj` files are already listed), speech, document chat (retrieval).
- Presets for system prompts and sampler settings, per-model default parameters, a prompt-template editor.
- Several models loaded at once; a benchmark screen (tokens per second on your machine at several context lengths); a request log for the local server.
- Gated Hugging Face repositories (login), split-file downloads in one step, a "recommended for your hardware" list.
- A signed installer and auto-update; release binaries that bundle the engine (needs CUDA in CI).

## Quantization and measurement

- GPTQ / AWQ import, importance-matrix aware `forge quantize` that writes GGUF (the pieces are in `forge-quant`).
- A published KL-divergence and top-1-agreement table per model and recipe, produced by one command.
- Per-layer importance analysis (how much does skipping or coarsening one layer cost?) to build recipes automatically instead of by hand.

## Research line

The from-scratch trainer and the small Forge-1 models (150M to 365M parameters) are documented in [ARCHITECTURE.md](ARCHITECTURE.md) and [RUNBOOK.md](RUNBOOK.md). Open questions there: how far inheriting weights from a larger pretrained model (pruning plus a little training, as in the "Sheared LLaMA" and "Minitron" papers) beats training the same size from scratch on a small budget, and distilling a large model's answers into a small one.
