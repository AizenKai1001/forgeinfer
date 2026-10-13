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
| Qwen3.5-0.8B | 332.8 | 265.3 | **1.25x** | 13,473 | 12,273 | 1.10x |
| Qwen3.5-4B | 79.6 | 73.4 | 1.08x | 2,379 | 3,473 | **0.68x** |
| Qwen3.5-9B | 45.2 | 43.7 | 1.03x | 1,360 | 2,112 | **0.64x** |

Reading it: on the 4B and 9B, decode is memory-bound (272 GB/s) and both engines sit near the roofline, so the gain is a few percent; the engine's edge is largest on small models where launch overhead matters (the whole decode step is one CUDA graph). **Prefill on the 4B and 9B is where the engine loses on this card**: llama.cpp reads the quantized blocks with int8 tensor cores, while the engine dequantizes into fp16 tiles and uses the fp16 tensor cores; an int8 path is the first item of the prefill work in the [roadmap](ROADMAP.md).

A first tuning on this card came out of that table: the engine switches between a fused dequantizing GEMM (short chunks) and a two-pass dequantize-then-GEMM (long chunks), and its 448-token threshold was measured on Turing. On the RTX 4060 the fused kernel wins at every length measured (4B, 9B, tokens per second): at 554 tokens 2,246 to 2,379 and 1,271 to 1,360; at 1,086 tokens 2,120 to 2,463 (+16 %) and 1,159 to 1,397 (+21 %); at 2,169 tokens +8 to +10 %. Greedy continuations after 1,000 to 1,400-token prompts are identical with both paths. Ada GPUs (compute capability 8.9) now use the fused path always; `FORGE_FUSED_MAX=N` overrides, `0` restores the two-pass path.

## Speculative decoding on the RTX 4060

Qwen3.5 models ship a small prediction head (multi-token prediction, `mtp.*` tensors in the original checkpoint, not in the GGUF files). With it the engine drafts 2 or 3 tokens ahead and the main model checks them all in one pass over the weights; when decode is limited by memory bandwidth, that pass costs about the same as one token, so every accepted guess is free. Measured on the PC (a 128-token continuation of English prose, 128 tokens, median of four runs, Q4_K_M files, `--depth 3`):

| model | greedy | greedy + head | sampled (T 0.7) | sampled + head |
|---|---|---|---|---|
| Qwen3.5-4B | 81.6 | **144.6** (1.77x) | 80.8 | **117.4** (1.45x) |
| Qwen3.5-9B | 45.8 | **99.4** (2.17x; depth 2: 86.7) | 45.8 | **72.0** (1.57x) |

A full streamed chat on the 9B with thinking on and the app's sampler settings (T 0.7, top-p 0.9, top-k 40, repetition penalty 1.05, 600 tokens, the server's own timings): 45.3 tokens/s without the head, **78 to 82 with it**. The sampler settings cost little (top-p, top-k and the repetition penalty one at a time: 65 to 71 against 66 plain with the head on a prose prompt).

**It is not bit-identical to plain decoding.** The drafted tokens are checked by the multi-token kernels, which round slightly differently from the single-token path (the activations are quantized to int8 in both, in different groupings). On eight short chat prompts with temperature 0, the answers matched word for word in 6 of 8 cases (4B) and 5 of 8 (9B); where they differed it was at a near-tie between two good words ("a golden-brown exterior" against "a golden-brown crust", "little political power" against "no political power") and both answers were fluent and correct. Expect the same quality, not the same text.

Two practical points:

- The head is 0.24 GB for the 4B and 0.49 GB for the 9B. Forge Studio fetches it with HTTP range requests (only the `mtp.*` tensors of the original checkpoint, no multi-gigabyte download) from the model's card in My models, and uses it automatically; `tools/fetch_mtp.py` does the same from a shell, and `--mtp FILE --depth 3` passes it to `forge serve`.
- **A window that redraws on every streamed token slows the engine down.** The first Studio build repainted for each token and measured 34.6 tokens/s with the head on, against 78 to 82 through the HTTP API alone: on Windows the graphics and the compute work of one GPU take turns, and speculative decoding makes many short steps. Studio now spaces repaints by 50 ms and measures 79.3 tokens/s in the window. (Without the head the loss had been smaller: 41.3 against 45.3.)

## Prompt lookup: speculation for every model, up to 2x on edits

The prediction head helps only Qwen3.5. **Prompt lookup** needs no head and no second model: when the last few tokens of the text (3 to 6 of them) also occur earlier, in the prompt or in the answer so far, the tokens that followed them there are proposed as drafts, and the main model verifies the whole run in one pass over the weights, exactly as it verifies the head's drafts. Code edits, quoted passages, corrections of a pasted text, lists and JSON repeat themselves, and then 4 to 8 tokens are confirmed per pass instead of 1. It is on by default (`--no-lookup` or `FORGE_PLD=0` switch it off, `--lookup-draft N` sets the longest run, default 11 (a run starts at 7 and doubles after a cycle whose drafts were all right); `--no-speculate` turns off every kind of speculation).

Measured with `tools/pld_bench.py` (a 45-line Python file to type-annotate, a JSON array to generate, a five-bullet summary with quotes, a grammar fix, and a free-prose explanation; 260 to 700 tokens each; the server's own timings), tokens per second without and with the lookup:

| model, GPU | code edit | JSON | quotes | grammar fix | prose |
|---|---|---|---|---|---|
| Qwen3.5-4B, no head, RTX 4060 (greedy) | 79.7 to **159.2** (2.00x) | 80.6 to **128.5** (1.59x) | 80.4 to 86.7 (1.08x) | 81.1 to **182.7** (2.25x) | 81.3 to 80.5 |
| Qwen3.5-4B, no head, RTX 4060 (T 0.7) | 78.9 to **155.5** (1.97x) | 79.8 to **122.8** (1.54x) | 79.7 to 89.7 (1.13x) | 80.4 to **167.8** (2.09x) | 80.2 to 80.3 |
| Qwen3-4B (dense), no head, RTX 2060 SUPER (greedy) | 112.1 to **194.3** (1.73x) | 123.8 to 135.4 (1.09x) | 116.4 to **170.7** (1.47x) | 122.2 to **258.9** (2.12x) | 125.2 to 127.4 |
| Qwen3.5-9B + head, RTX 4060 (greedy, 12 rows) | 106.5 to **144.5** (1.36x) | 108.3 to 108.5 | 93.9 to 114.3 (1.22x) | 116.4 to **162.4** (1.40x) | 76.1 to 75.9 |
| Qwen3.5-9B + head, RTX 4060 (T 0.7, 8 rows) | 106.8 to **136.8** (1.28x) | 106.7 to 106.1 | 81.4 to 78.6 | 109.6 to **148.4** (1.35x) | 75.0 to 75.4 |
| Qwen3.5-0.8B + head, RTX 4060 (T 0.7) | 480.9 to 588.0 (1.22x) | 596.5 to 649.5 | 467.5 to 488.0 | 602.4 to **814.1** (1.35x) | 327.1 to 326.2 |

Free prose is unchanged (a lookup cycle is only used when its measured payoff beats the alternative), and a model that already has a head gains less because the head predicts copied text well too (the 9B drafts 2.6 accepted tokens per pass on the code edit even without the lookup).

How it stays safe:

- **The output is the model's own sample.** Every position of a verification pass is sampled from the main model's distribution (same sampler, same random stream), and a draft only saves time when it equals that sample. With a head (9B, 0.8B) every answer was word-for-word identical with and without the lookup, greedy and sampled with the same seed: the head's own cycles already verify through the multi-row kernels, so the numbers are the same. Without a head the multi-row kernels replace the single-token ones for the whole answer, which round slightly differently, and 6 of 15 answers matched exactly (4B greedy 3 of 5, 4B sampled 1 of 5, Qwen3-4B 2 of 5); the others diverged at a near-tie between two good words ("a machine riders propelled by their feet" against "a wooden running machine"), as speculative decoding with a head does. On 40 explanation and code prompts (4B, greedy, `scratch/equiv_many.py`) 27 answers were identical, the other 13 held on average 61 % of their text before the first difference, and the speed did not change (81.9 against 82.3 tokens/s: a lookup that finds nothing costs nothing).
- **A guard measures instead of assuming.** The drafts accepted per lookup cycle are averaged per length of the matched n-gram and compared with the break-even point against the alternative (the head's own running average, or one plain step); a length that does badly is dropped and retried after a while.
- **Memory is checked first.** A hybrid model (Qwen3.5) keeps one recurrent state per verified row (50 MB each on the 4B). The engine reserves the verification buffers once, only when `cuMemGetInfo` shows at least 200 MiB left afterwards (`FORGE_VERIFY_MARGIN_MIB`), and drops to fewer rows otherwise. This matters on a Windows GPU: with 12 verification rows on the 9B in 8 GB the free memory fell to 157 MiB and every sweep took 63 ms instead of 26 (the driver pages), while 8 rows (367 MiB left) and 10 rows (260 MiB) showed no loss. The snapshots of hybrid models at message boundaries now live in page-locked host memory (50 MB, 4 ms to copy) to leave room for the rows; `FORGE_SNAP_DEVICE=1` keeps them on the GPU.

A smaller finding from the same session: the fused dequantizing prefill GEMM used the `m16n8k8` tensor-core instruction on every GPU; on sm_80 and newer it now uses `m16n8k16`, which is worth about 2 % of prefill on the RTX 4060. The kernel was already at roughly 85 % of the card's fp16 tensor-core peak (24 to 27 TFLOPs on the Q4_K shapes), so the remaining prefill gap to llama.cpp is the int8 path described in the [roadmap](ROADMAP.md).

## Loading a model

`FORGE_LOAD_TRACE=1` prints where the seconds of a load go. Repacking the GGUF blocks into the planar layouts of the kernels ran on one thread and took 7.5 of the 9.1 seconds of loading the 9B; it is cut into runs of blocks that go through all cores now (the result is byte-identical, a test compares it with the one-thread version for every format). Ready to chat on the RTX 4060 (file in the page cache, with the prediction head): 0.8B 1.3 s, 4B 4.1 to **3.4 s**, 9B 9.1 to **6.0 s**. What is left of the 9B's 6 seconds: about 1 s each for repacking and the copy to the device, 1 s of warm-up, the head.

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
