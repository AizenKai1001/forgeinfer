# Performance

Everything here was measured by the author on specific machines on specific dates, with the commands in `tools/`. Treat the numbers as a record of what the code did then, not as a promise; run `tools/bench_vs_llama.sh` on your own hardware.

Machines:

- **Homelab**: RTX 2060 SUPER (8 GB, 448 GB/s), Core i7-9700K, 32 GB DDR4-3200 (XMP on), Ubuntu 24.04, CUDA 13.2.
- **PC**: RTX 4060 (8 GB), Core i5-12600K, 32 GB, Windows 11, `forge.exe` from `tools/build_windows.ps1` (CUDA 12.9).

## Decode and prefill against llama.cpp

Same GGUF files (Q4_K_M from unsloth), idle GPU, llama.cpp build b11521, homelab, 2026-10-09. Decode is 128 generated tokens, prefill is a 512-token prompt (tokens per second).

| model | decode forge | decode llama.cpp | ratio | prefill forge | prefill llama.cpp | ratio |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 484 | 285 | **1.70x** | 8,859 | 9,336 | 0.95x |
| Qwen3.5-4B | 119 | 94.0 | **1.27x** | 1,933 | 2,282 | 0.85x |
| Qwen3.5-9B | 70 | 58.8 | **1.19x** | 1,081 | 1,393 | 0.78x |
| Qwen3-4B | 127 | 108.5 | 1.17x | | | |
| Qwen3-1.7B | 281 | 212.6 | 1.32x | | | |
| Qwen3-1.7B Q8_0 | 191 | 158.7 | 1.20x | | | |
| Qwen3-0.6B Q8_0 | 436 | 287 | 1.52x | | | |

- Greedy continuations of the Qwen3.5 4B and 9B are token for token identical to llama.cpp's for the first 40 tokens.
- Where decode wins: the whole decode step is one CUDA graph of fused kernels that read the quantized blocks directly, with activations pre-quantized once per layer.
- **Where it loses: prefill on short prompts.** Prefill is tensor-core GEMM; for the 4B and 9B the engine was 2 to 22 percent behind llama.cpp at 512 tokens. A later round of work brought the 4B and 9B to or above llama.cpp from 1,024 tokens on; between 256 and 512 tokens they were still 2 to 10 percent behind.

On the PC (RTX 4060, 8 GB, Windows 11, 2026-10-13): llama.cpp b11517 CUDA build (`llama-bench -ngl 99 -p 512 -n 128`) against `forge serve` (the server's own timings for a 554-token prompt and 128 generated tokens, median of four runs, Q4_K_M files):

| model | decode forge | decode llama.cpp | ratio | prefill forge | prefill llama.cpp | ratio |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 332.5 | 265.3 | **1.25x** | 13,770 | 12,273 | 1.12x |
| Qwen3.5-4B | 79.6 | 73.4 | 1.08x | 2,246 | 3,473 | **0.65x** |
| Qwen3.5-9B | 45.2 | 43.7 | 1.03x | 1,271 | 2,112 | **0.60x** |

Reading it: on the 4B and 9B, decode is memory-bound (272 GB/s) and both engines sit near the roofline, so the gain is a few percent; the engine's edge is largest on small models where launch overhead matters (the whole decode step is one CUDA graph). **Prefill on the 4B and 9B is where the engine loses on this card**: llama.cpp reads the quantized blocks with int8 tensor cores, while the engine dequantizes into fp16 tiles and uses the fp16 tensor cores; an int8 path is the first item of the prefill work in the [roadmap](ROADMAP.md).

## A 27B model on an 8 GB card

Qwen3.5-27B has 64 layers; a 4-bit file is 16 to 17 GB. The engine keeps the attention and recurrent layers, the embeddings, the head and the KV cache on the card and runs the feed-forward layers that do not fit (62 of 64) from page-locked system memory with AVX2 kernels inside the CUDA graph. Decode is then bounded by system-memory bandwidth (about 40 GB/s of weights per token set), not by the GPU.

Path of one night on the homelab (greedy, 800 tokens, KV int8, thinking off), tokens per second:

| step | tok/s |
|---|---|
| first working version: GPU layers + CPU feed-forward, one token per pass | 4.1 |
| speculative decoding with the model's own prediction head, depth 2 | 5.1 |
| depth 3, and the DDR4 profile (XMP) switched on: DRAM peak 31 to 42.7 GB/s | 6.95 |
| CPU thread pool of 7 with dynamic work chunks | 7.59 |
| SIMD kernel fixes (vector scale unpacking, F16C conversions) | 8.42 |
| a recipe: down projections in Q4_K everywhere (16.1 GB file) | 9.47 |
| plus the attention and output projections on the card in Q4_K (15.6 GB file): three more layers fit on the card | **9.65 to 9.78** |

Quality cost of the recipes, measured as KL divergence of the next-token distribution against a reference (Q4_K_M with the feed-forward from Q6_K) on 5 text domains, and top-1 agreement: the file as downloaded 0.0040; the first recipe 0.0052; the last 0.0068 with the same top-1 token as the reference 96.2 percent of the time. `forge gguf-stitch BASE RECIPE OUT` builds such files from existing GGUFs without re-quantizing (the recipe format is described in [ENGINE.md](ENGINE.md#recipes-mixing-quantizations-without-re-quantizing)).

Other numbers from the same work: short-prompt first token 2.4 s to 1.5 s (the recurrent-state snapshot now sits after the assistant header and a short tail runs as groups of four rows on the CPU kernels); long prompts read at 150 to 220 tokens per second (bounded by the PCIe link: each prefill pass moves the 9.7 GB of host-resident weights at 12.3 GB/s); sampled answers 7.5 to 8.5 tok/s; process memory 10 GiB, video memory 7.5 GB at a context of 8192 with an int8 KV cache.

## What moved the needle, and what did not

Worked:

- **Reading fewer bytes per token**: speculation depth 3 (one sweep of the weights verifies three tokens), mixed-precision recipes chosen from measured per-tensor sensitivity, KV int8.
- **Memory speed**: enabling XMP was worth +45 percent on the CPU-bound path; a manual memory clock above XMP is the only hardware lever left (+8 to 12 percent expected).
- **Instruction-level work**: hardware counters (`perf stat` uop and port counters) showed the first CPU kernels were front-end bound (3.2 of 4 uops per cycle); a software f16 to f32 conversion and scalar scale unpacking were 40 percent of the instructions.
- **Sleeping instead of spinning**: the CPU threads waiting for the GPU used to burn cores; blocking synchronization (`CU_CTX_SCHED_BLOCKING_SYNC`) gives them back to the memory-bound workers.

Did not work (kept off, documented in `docs/KERNEL_TUNING.md` and the commit messages; skipping layers is measured in [LAYER_IMPORTANCE.md](LAYER_IMPORTANCE.md): six middle feed-forwards buy 10 percent speed for a KL of 0.24): pipelined prefill GEMM (slower), per-256 activation scales (slower, quality-neutral), Q8_0 with F16C (conversion per 32 values costs more than it saves), DMA staging of prefill weights (the dequantizer already reads PCIe at line rate), Q3_K for feed-forward weights (saves RAM, hardly any time: its kernel runs below the DRAM peak).

Method notes that saved time: compare *mean and median* over 48 or more runs (12 runs have +-20 percent noise); compute-only and DRAM-sized variants of a kernel benchmark separate "the kernel is slow" from "the memory is slow"; always check the binary's timestamp after a build (a stale binary once produced a whole phantom "no improvement" series).

## Reproducing

```bash
tools/bench_vs_llama.sh        # forge vs llama.cpp on the same files (needs llama.cpp's llama-bench)
forge gpu-bench ...            # decode microbenchmarks;  forge prefill-bench ...
forge ppl / eval-*             # perplexity and KL against a reference, HellaSwag and friends
```
