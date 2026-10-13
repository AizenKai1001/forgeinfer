# Roadmap

What is missing, roughly in the order that would help the most people. Nothing here is promised; it is where help is most welcome.

## Engine

1. **A CPU backend.** Today the engine needs an NVIDIA GPU. A portable CPU path (AVX2 and AVX-512 on x86, NEON on Arm) would run every supported model on any computer and make Forge Studio fully self-contained on macOS and on Linux without CUDA. The pieces exist: the f32 reference forward pass in `forge-model` is the oracle, and `forge-gpu/src/cpuk.rs` already holds hand-vectorized AVX2 kernels for Q2_K to Q8_0 that run inside the GPU graph. What is missing is the whole model on the CPU (the Gated DeltaNet and attention layers, the KV cache, the sampler loop) and NEON kernels.
2. **More architectures**: Gemma, Mistral, Phi, and mixture-of-experts models (Qwen3-30B-A3B is a good first target: with expert offload it is fast on small cards).
3. **A SentencePiece tokenizer** (Llama 2, Mistral, Gemma). The BPE one is built from the vocabulary and merges in the GGUF file; the unigram model is a separate piece of work with its own goldens.
4. **Prefill speed** (0.82 to 0.88 times llama.cpp b11517 on the Qwen3.5 9B and 0.9 times on the 4B at 2,000 tokens on an RTX 4060, after the fp16-accumulating GEMMs; 0.6 to 0.85 times at 512 tokens on an RTX 2060 SUPER): an INT8 tensor-core GEMM that reads the quantized blocks directly (what llama.cpp's MMQ kernels do) instead of dequantizing into fp16 tiles. What the measurements say (kernel profile with `FORGE_KPROF=2 forge prefill-bench`): the fused kernels reach 31 TFLOPs on the Q4_K gate and up shape of the 9B with fp16 accumulation, about half of what a one-second probe loop measures (61 TFLOPs at boost clock), and the RTX 4060 runs power-capped at about 2.3 GHz during a prompt pass ([PERFORMANCE.md](PERFORMANCE.md#prompt-reading-on-the-rtx-4060-what-moved-it-what-did-not)); GEMMs are 70 to 85 % of a prefill and the Gated DeltaNet scan 5 to 14 %, so only int8 (less energy per operation, and the 4-bit weights are read as they are) can go much further. llama.cpp's MMQ converts every output element to float once per 32-weight sub-block (three ALU operations against 32 multiply-adds, which makes it ALU-bound at about a fifth of the int8 peak); a variant with one activation scale per 256 values and the sub-block scales folded into two int8 weight bytes removes that epilogue at the price of two MMAs per slab. Either way the ceiling is about twice the fp16 rate and a realistic first result +25 % on prefill. **Measured ceilings** (`bench/gpu_mmq_spike.cu`, correctness checked against a reference, RTX 4060, 128 x 128 tiles, three-stage cp.async pipeline, ldmatrix fragments, no decoding and no epilogue): one int8 `m16n8k32` MMA per slab runs at **25 TMAC/s** (50 TOPS, 41 % of the card's int8 peak), twice the 12.5 TMAC/s of the fused fp16 kernel; the scale-folding variant with two weight planes and two MMAs per slab reaches only **14.2 TMAC/s** of weight MACs, 1.14 times the fp16 kernel before any decoding, so it is a dead end. A real MMQ kernel has to fit the 4-bit decode and the per-sub-block epilogue of every K-quant (Q4_K, Q5_K and Q6_K are 79 % of the GEMM time of a Q4_K_M model) into the gap between 25 and the roughly 16 TMAC/s that llama.cpp's reaches. Not started; expect +10 to +30 % of prefill for days of work.
5. **IQ-quantizations** (`IQ2`, `IQ3`, `IQ4_XS`) in the decoders and kernels.
6. **Serving several clients at once**: today requests are queued; continuous batching would let a few chats share one pass over the weights.
7. **Vulkan or Metal** back ends. A long shot; the CPU backend comes first.
8. **Structured output and API parity**: grammar (GBNF) input, `pattern` / length / range checks in the schema automaton, tool-call arguments constrained by the tool's own schema, `logprobs`, `/v1/embeddings`, `n > 1`.

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
