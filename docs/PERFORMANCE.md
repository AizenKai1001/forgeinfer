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
- **Where it loses: prefill on short prompts.** Prefill is tensor-core GEMM; for the 4B and 9B the engine was 2 to 22 percent behind llama.cpp at 512 tokens. A later round of work brought the 4B and 9B to or above llama.cpp from 1,024 tokens on; between 256 and 512 tokens they were still 2 to 10 percent behind. The int8 prompt GEMMs of 2026-10-13 (see below) put the engine ahead of llama.cpp on an RTX 4060 at every prompt length measured (64 to 2,226 tokens); the 2060 SUPER (Turing) has no int8 path yet.

On the PC (RTX 4060, 8 GB, Windows 11, 2026-10-13): llama.cpp b11517 CUDA build (`llama-bench -ngl 99 -p 512 -n 128`) against `forge serve` (the server's own timings for a 554-token prompt and 128 generated tokens, median of four runs, Q4_K_M files):

| model | decode forge | decode llama.cpp | ratio | prefill forge | prefill llama.cpp | ratio |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 332.8 | 265.3 | **1.25x** | 13,473 | 12,273 | 1.10x |
| Qwen3.5-4B | 79.6 | 73.4 | 1.08x | 2,379 | 3,473 | **0.68x** |
| Qwen3.5-9B | 45.2 | 43.7 | 1.03x | 1,360 | 2,112 | **0.64x** |

Reading it: on the 4B and 9B, decode is memory-bound (272 GB/s) and both engines sit near the roofline, so the gain is a few percent; the engine's edge is largest on small models where launch overhead matters (the whole decode step is one CUDA graph). **Prefill on the 4B and 9B is where the engine loses on this card**: llama.cpp reads the quantized blocks with int8 tensor cores, while the engine dequantizes into fp16 tiles and uses the fp16 tensor cores; an int8 path is the first item of the prefill work in the [roadmap](ROADMAP.md).

A first tuning on this card came out of that table: the engine switches between a fused dequantizing GEMM (short chunks) and a two-pass dequantize-then-GEMM (long chunks), and its 448-token threshold was measured on Turing. On the RTX 4060 the fused kernel wins at every length measured (4B, 9B, tokens per second): at 554 tokens 2,246 to 2,379 and 1,271 to 1,360; at 1,086 tokens 2,120 to 2,463 (+16 %) and 1,159 to 1,397 (+21 %); at 2,169 tokens +8 to +10 %. Greedy continuations after 1,000 to 1,400-token prompts are identical with both paths. Ada GPUs (compute capability 8.9) now use the fused path always; `FORGE_FUSED_MAX=N` overrides, `0` restores the two-pass path.

## Against LM Studio's engine and llama.cpp's own speculation

`tools/bench_engines.py` runs the five chat workloads of `tools/pld_bench.py` (a code edit, a JSON array, a summary with quotes, a grammar fix, free prose; 260 to 700 tokens each) against `forge serve`, llama.cpp's `llama-server` (build b11517) and the `llama-server` that ships inside LM Studio 0.4.16 (its CUDA 12 runtime 2.29.1, llama.cpp b10355, started from its own folder with its own libraries; nothing of LM Studio's settings is touched), on the same GGUF file, greedy, thinking off, and reads the generation speed from each server's own timings. RTX 4060 (8 GB), Windows 11, 2026-10-13, tokens per second. Run to run the same engine moves by up to 4 percent with the card's clocks, so read ratios within about 5 percent as ties.

Qwen3.5-9B Q4_K_M:

| engine | code edit | JSON | quotes | grammar fix | prose |
|---|---|---|---|---|---|
| LM Studio engine (llama.cpp b10355) | 43.1 | 43.0 | 43.0 | 42.7 | 43.1 |
| llama.cpp b11517 | 43.7 | 43.2 | 42.6 | 42.8 | 42.8 |
| llama.cpp b11517 + `--spec-type ngram-simple` (4-token lookup, 16 drafted) | 76.0 | 45.1 | 55.0 | 74.5 | 43.2 |
| ForgeInfer, no speculation | 46.5 | 46.9 | 46.8 | 47.1 | 47.0 |
| ForgeInfer, prompt lookup (the default for a model without a head) | 94.8 | 68.2 | 83.2 | 110.7 | 47.0 |
| ForgeInfer, lookup + the model's prediction head (what Forge Studio does) | 150.7 | 111.0 | 111.7 | 155.4 | 78.6 |

Qwen3.5-4B Q4_K_M (the llama.cpp and ForgeInfer rows are one run; the LM Studio row is from a second run in which llama.cpp b11517 read 72.3 and ForgeInfer without speculation 83, with the head 255, 192, 130, 280 and 134):

| engine | code edit | JSON | quotes | grammar fix | prose |
|---|---|---|---|---|---|
| LM Studio engine (llama.cpp b10355) | 76.1 | 76.0 | 75.9 | 76.1 | 75.9 |
| llama.cpp b11517 | 71.0 | 71.9 | 70.6 | 72.4 | 72.5 |
| llama.cpp + `ngram-simple`, defaults (12-token lookup, 48 drafted) | 99.9 | 68.2 | 70.2 | 75.6 | 71.8 |
| llama.cpp + `ngram-mod`, defaults | 88.0 | 72.7 | 73.2 | 72.8 | 72.8 |
| llama.cpp + `ngram-map-k`, defaults | 95.1 | 70.4 | 71.8 | 70.4 | 72.7 |
| llama.cpp + `ngram-simple`, 4-token lookup, 16 drafted | 113.0 | 75.1 | 64.2 | 112.8 | 71.6 |
| ForgeInfer, no speculation | 79.1 | 80.0 | 79.8 | 80.5 | 80.6 |
| ForgeInfer, prompt lookup | 159.1 | 127.7 | 86.4 | 181.1 | 80.0 |
| ForgeInfer, lookup + the model's prediction head | 247.9 | 186.4 | 126.6 | 270.7 | 130.8 |

Qwen3.5-0.8B Q4_K_M, no speculation, morning session: 324 to 348 tokens/s against 222 for LM Studio's engine and 210 for llama.cpp b11517 (1.5x). In the afternoon hour measured below the three others stood at 254 to 264 and the engine at 358 (1.4x): small models are launch-bound and move with the load of the computer.

**Ollama 0.40.3** (the portable Windows build, which starts its own llama.cpp runner with a CUDA 13 backend) was measured with the same GGUF files and the same prompts, through `/api/generate` with raw prompts (so the prompt tokens equal the other engines'), context 4,096, `/api/ps` confirming that the whole model sits on the GPU. A first run in the morning gave 41.5, 68.8 and 248 tokens/s (9B, 4B, 0.8B), a second run in the afternoon 44.8, 74.7 and 257: the same engine on the same files, 8 percent apart, with the state of the card and the load of the computer. Within one session the engines are comparable, across sessions only to about 8 percent, so the table below is one afternoon hour in which Ollama, llama.cpp b11517, LM Studio's engine and ForgeInfer were run one after the other (ForgeInfer with the prompt work described further down, no speculation). Decode is the mean of the five workloads above, tokens per second; prompt reading in tokens per second from each server's own timings; video memory above the idle desktop after loading (context 4,096).

| model | engine | decode | prompt, 592 tokens | prompt, 2,226 tokens | video memory |
|---|---|---|---|---|---|
| 9B | ForgeInfer | **46.7** | **2,613** | **2,581** | 5,552 MiB |
| | llama.cpp b11517 | 44.3 | 1,863 | 1,863 | 5,261 |
| | LM Studio's engine | 44.6 | 1,725 | 1,810 | 5,245 |
| | Ollama 0.40.3 | 44.8 | 1,671 | 1,734 | 5,239 |
| 4B | ForgeInfer | **82.7** | **4,453** | **4,286** | 3,392 |
| | llama.cpp b11517 | 73.7 | 2,890 | 3,355 | 2,973 |
| | LM Studio's engine | 74.2 | 2,629 | 2,981 | 2,963 |
| | Ollama 0.40.3 | 74.7 | 2,401 | 2,906 | 2,957 |
| 0.8B | ForgeInfer | **358** | **21,028** | **19,372** | 842 |
| | llama.cpp b11517 | 254 | 9,480 | 12,673 | 709 |
| | LM Studio's engine | 264 | 8,499 | 10,928 | 717 |
| | Ollama 0.40.3 | 257 | 7,134 | 9,802 | 713 |

Ollama tracks the llama.cpp builds (the same kernels). The engine's plain decoding is 4 to 6 percent faster on the 9B, 10 to 12 percent on the 4B and 1.36 to 1.41x on the 0.8B; with lookup and head the gaps are those of the tables above. The prompt columns were measured again later the same afternoon, after the int8 prompt GEMMs (item 5 of [the prompt-reading section](#prompt-reading-on-the-rtx-4060-what-moved-it-what-did-not)), with the four engines run one after the other within half an hour (`tools/bench_engines.py --prefill N` for the three servers, Ollama through its `/api/generate` with raw prompts, medians of five); the decode column is the earlier hour's. Reading prompts the engine is now **39 to 40 percent faster than llama.cpp b11517 on the 9B** (2,613 against 1,863 tokens/s at 592 tokens, 2,581 against 1,863 at 2,226), **28 to 54 percent faster on the 4B** (4,453 against 2,890; 4,286 against 3,355) and 1.5 to 2.2x faster on the 0.8B, and further ahead of LM Studio's engine and Ollama; before that work it was 5 to 18 percent slower than the others on the 9B. The sustained clock of this card moves with its temperature: a run an hour earlier gave llama.cpp 1,859 and 2,079 on the 9B and the engine 2,509 and 2,461 (before Q6_K joined the int8 path), a lead of 35 and 18 percent, so read the lead as 18 to 40 percent on the 9B, 22 to 54 on the 4B. Video memory of the engine is 4 to 12 MiB higher than with the fp16 path (the int8 copy of the activations of a 1,024-token chunk: 5,540 to 5,552 MiB on the 9B, 3,380 to 3,392 on the 4B, 838 to 842 on the 0.8B, nvidia-smi after loading and reading one 600-token prompt, FORGE_MMQ=0 against 1). Ollama loads the model with its first request: 4.0 to 6.6 s (9B), 4.4 to 4.8 s (4B) and 2.2 to 2.3 s (0.8B) with the file in the page cache; a reload after unloading takes 3.3, 2.7 and 1.9 s; ForgeInfer is ready 4.4, 2.9 and 2.4 s after the process starts, the llama.cpp servers 3.5, 2.0 and 1.5 s.

What it says:

- Without speculation the engine is about 5 to 12 percent faster than LM Studio's on the 9B and 4B (decode is memory-bound and every engine sits near the card's bandwidth) and 1.4 to 1.5x faster on the 0.8B (the decode step is one CUDA graph; launch overhead matters on small models).
- llama.cpp has n-gram speculation (`--spec-type ngram-*`), off by default. With its default settings it paid on the code edit only (1.2 to 1.4x); with a 4-token lookup and 16 drafted tokens (one setting tried, not a tuned optimum) it reached 1.6 to 1.7x on the code edit and the grammar fix, and lost speed on the 4B quotes (64.2 against 70.6: nothing measures whether drafts pay). The engine's lookup is on by default, reached 2.0 to 2.3x over the engine's own plain decoding on the same two workloads (2.2 to 2.6x over LM Studio's engine on the 9B), and lost no measurable speed on prose.
- llama.cpp's own head mode (`--spec-type draft-mtp`) refuses these GGUF files: "context type MTP requested but model doesn't contain MTP layers". The standard files do not contain the prediction head; Forge Studio fetches it from the original checkpoint (0.2 to 0.5 GB) and the engine uses it from a separate file.
- **Same model, same probabilities, text that drifts at near-ties.** No two engines give the same text for long: against llama.cpp's greedy answers the engine's differ somewhere in all five workloads (first difference at character 9 to 682 of 628 to 1,548, each time a near-tie between two good words: "Yesterday, me and my friends" against "Yesterday me and my friends", "designer" against "design"), and llama.cpp's own n-gram speculation changes its text in 1 of 5 workloads against its plain decoding, as the engine's speculation does against its own (see below). `tools/teacher_force.py` measures how close the probabilities are: it feeds the other engine's own answers through the engine and compares the engine's top-1 token with the token the other engine produced at every position of its answer. Against llama.cpp b11517 the engine picks the same token at 99.2 percent of 1,519 positions on the 4B, 99.0 percent of 1,431 on the 9B and 99.4 percent of 1,350 on the 0.8B; against LM Studio's engine 99.2 percent of 1,449 on the 9B. Where it differs, the other engine's token is a near-tie: the median gap to the engine's top choice is 0.04 to 0.11 nats (largest 0.89). Control: LM Studio's b10355 and llama.cpp b11517 disagree with each other in 4 of the 5 greedy answers on the same 9B file (first difference at character 154 to 419 of 670 to 1,514), so a drift like this is what two builds of the same engine do.

Prompt reading is in the table above (a prompt of N tokens, 8 tokens generated, median of five runs; `llama-bench -p 512` reads faster on every engine, 3,497 and 2,166 tokens/s for llama.cpp b11517 on the 4B and 9B in the morning, because a server also samples and handles requests). In the morning session, before the prompt work of 2026-10-13, the engine read the 9B at 1,456 and 1,466 tokens/s (592 and 2,226 tokens) and the 4B at 2,541 and 2,598, against 1,718 / 1,924 and 2,567 / 2,969 for LM Studio's engine and 1,847 / 2,073 and 2,878 / 3,341 for llama.cpp; the fp16-accumulating GEMMs and the two-pass Q5_K / Q6_K path took the 9B to about 1,700 and the 4B to about 3,000, and the int8 GEMMs to the values of the table. First token of the 9B (median of nine, a one-token completion): 30 ms for 12 prompt tokens, 242 ms for 300, 778 ms for 1,100.

## Prompt reading on the RTX 4060: what moved it, what did not

A prompt pass of the 9B was about 85 percent tensor-core GEMM (the fused dequantizing kernels), 7 percent the Gated DeltaNet scan and a few percent normalisation, SwiGLU and attention. The work of 2026-10-13 took the 9B from 1,466 to about 1,700 tokens/s at 2,226 tokens in the server (+15 percent), the 4B from 2,598 to about 3,000 (+15 percent) and the 0.8B from 12,554 to about 15,400 (+23 percent). Items 1 to 4 are the first round (its numbers are the ones in the tables above); item 5, an int8 path for the quantized matrices, came after the comparison tables were measured and is the biggest single step. In order of size:

1. **fp16-accumulating tensor-core MMAs** (`mma.m16n8k16` with fp16 accumulators). On GeForce Ampere and Ada the instruction runs at twice the rate of the fp32-accumulate form (a probe loop: 61.4 against 30.3 TFLOPs on the RTX 4060). The fused kernels add the 16 products of one k16 step in fp16 and flush the partial sum into fp32 accumulators after every step (compute capability 8.6 and 8.9; `FORGE_F16ACC=0` switches it off). **+6 percent of prompt speed**, not 2x: ablations in a stand-alone copy of the kernel showed that the weight loads, the decoding and the barriers add to the tensor time instead of overlapping with it, and that the kernel is not limited by wave quantization, by the coalescing of the weight loads or by the depth of their lookahead. Numerics against fp32 accumulation on the 4B (three 2,046-token texts): KL of the top 64 at most 1e-5 nats, top-1 token the same at 99.90 to 100 percent of positions, change of the negative log-likelihood +0.00006, +0.00001, -0.00001.
2. **Q6_K keeps its scales in registers**: the 16 scale bytes and the fp16 `d` of a super-block are the same for its 16 tiles; they were two tiny loads per tile, each touching 16 cache lines per warp. Loading them once per super-block is bit-identical and worth 1 to 3 percent of the whole pass.
3. **Dequantize, then multiply with the plain Ampere GEMM at fp16 accumulation** for Q5_K and Q6_K matrices of chunks above 384 and 256 tokens (Q4_K stays on its fused kernel, the best of the three). The cp.async-pipelined kernel of `gemm_s80.cu` reaches 41 TFLOPs in isolation with fp16 accumulation (31 with fp32); in the model, per layer of the 9B at 1,024 tokens, the Q6_K down projection takes 3.57 ms (dequantization 0.58 ms + GEMM 2.99 ms) against 5.49 ms fused and the Q5_K projections 2.33 against 3.07, while the Q4_K gate and up projections would take 3.47 against 3.34. The matrix is dequantized in slabs of rows into buffers of the prefill that are dead at that moment (the gate, up and SwiGLU buffers), so the path needs no scratch of its own and video memory is unchanged. **+7 percent at 1,024-token chunks** on the 9B (1,686 to 1,803 tokens/s in `prefill-bench`), +6 percent on the 4B. Against fp32 accumulation the whole prompt path (fused and two-pass) differs by a KL below 5e-6, the same top-1 token at 99.85 to 100 percent of positions on the 4B and 9B and a change of the log-likelihood of at most 7e-5; against llama.cpp the teacher-forced top-1 agreement stayed at 99.3 percent (4B) and 99.1 percent (9B).
4. **The Gated DeltaNet scan runs 32 columns per block on a GPU with fewer than 32 SMs.** One block of 1,024 threads fills an SM of the RTX 4060 (24 SMs), so the 32 heads ran in two waves: the scan took 1.77 ms per layer on the 9B, 1.25 ms now (0.84 to 0.57 ms on the 0.8B, where the scan was 22 percent of the pass): +1 percent on the 9B, +3 percent on the 4B, +7 percent on the 0.8B.

5. **Integer tensor-core GEMMs for Q4_K, Q5_K and Q6_K weights** (`gemm_mmq.cu`, the idea of llama.cpp's `mul_mat_q` kernels, written from scratch for the engine's planar layouts; sm_80 and newer, chunks of at least 33 tokens, `FORGE_MMQ=0` switches it off). The prompt's activations are quantized to int8 with one scale per 32 values (the rule and the arrays of the decode kernels: d = amax / 127, scale rounded to fp16, an exact integer sum per group), the weights stay 4 to 6 bits in memory, and the products run on the int8 tensor cores: `mma.m16n8k32` int8 peaks at 1,024 MAC per clock and SM on the RTX 4060 (a probe loop, `bench/gpu_imma_rate.cu`: 62 to 68 TMAC/s depending on the clock, 125 to 136 TOPS), twice the fp16-with-fp16-accumulate rate and four times the fp16-with-fp32-accumulate one. Three things make it fast. (a) A Q4_K weight is d sc q - dmin m, so the dot product of an output with one sub-block of 32 activations is a_d (d sc S - dmin m T) with S = sum q a_q (the integer MMA) and T = sum a_q (the quantizer's sum): the second term is a product of two small matrices (a_d T and -dmin m, both fp16) that ONE fp16 MMA (k = 8 of 16) per output tile and super-block adds into the same fp32 accumulators. (b) The int32 sums are not converted with I2F (16 per clock per SM): the MMA's accumulator input is the constant 0x4B400000 (the float 1.5 x 2^23), so its output is the bit pattern of 1.5 x 2^23 + S, and d sc S is ONE fma(bits, d sc, -1.5 x 2^23 d sc); a second fma applies the token's scale and accumulates (two FFMA per output and sub-block). (c) Everything arrives by cp.async, nothing passes through registers: the activation tile, the weights as they sit in memory (one ldmatrix fetches the packed fragments of two sub-blocks, a LOP3 masks the nibbles, the high nibbles are used unshifted as 16 q with the factor 1/16 in the scale, Q5_K ORs its 5th bit in), and the per-token scales and minimum operands, which the quantizer writes in the shared-memory layout. Q6_K has scales per 16 weights and no minimum: two k = 16 MMAs per sub-block with their own scales, a three-FFMA epilogue, weights made signed by an add and an xor per register. **The 9B reads prompts 49 to 62 percent faster** (`forge prefill-bench`, A/B interleaved within the same minutes by `tools/ab_speed.py`, FORGE_MMQ=0 against 1): 1,804 to 2,700 tokens/s at 1,024 tokens, +49 percent at 64 tokens, +55 to +62 at 128 to 256, +49 to +50 from 512 to 2,048; the 4B +36 to +48 percent (2,931 to 4,243 at 1,024 tokens), the 0.8B +20 to +37 (16,156 to 19,300). Chunks of 33 to 57 tokens, which the multi-row path used to take, read 37 to 84 percent faster on the GEMM path now (the multi-row path keeps up to 32). In the 9B pass the Q4_K gate or up GEMM went from 3.34 to 2.01 ms (25.6 TMAC/s, llama.cpp's kernel needs 2.3 ms for the same shape) and the Q6_K down projection from 3.77 ms (dequantization + fp16 GEMM) to 3.01. Numerics against the fp16 path (0.8B and 9B, three 4,092-token texts each, `tools/ab_numerics.py`): KL of the top 64 between 4.0e-4 and 6.2e-4 nats, the same top-1 token at 98.7 to 99.5 percent of the positions, change of the negative log-likelihood -0.0006 to +0.0011 (perplexity at most +0.11 percent on the 0.8B, below 0.06 percent on the 9B): the noise of an 8-bit activation, which llama.cpp's kernels and the engine's own decode kernels carry as well. A stand-alone copy of the kernel with ablations (no epilogue, no loads, no barrier) shows where the rest of the tensor-core peak goes: the loop with only ldmatrix and MMAs reaches 44 TMAC/s, the stage barrier costs 15 percent (all warps drain their MMA pipeline at it), the two FFMA per output 10 percent, the cp.async traffic 12 percent.

**The card is power-capped, which changes what a TFLOPs figure means.** During a sustained prompt pass `nvidia-smi` shows the SM clock at 2.28 to 2.39 GHz (maximum 3.1 GHz) with the clock-throttle reason "software power cap" (limit 115 W). Probe numbers from a one-second loop (30 and 61 TFLOPs) are boost-clock numbers; the sustained rate of the same instructions is lower by roughly 15 to 25 percent, and the plain GEMM that measured 41 TFLOPs in a short loop ran at 34.5 inside the model. Work that moves more data (a dequantized copy of a matrix written to and read from memory) probably costs more than its bandwidth share in such a regime, which may be why the fused Q4_K kernel, which writes no copy, still wins on the biggest matrices.

What was tried and did not help (A/B on the model or in a stand-alone harness): keeping the Q4_K header in registers, a BK = 32 k step, deeper lookahead of the weight loads and coalesced weight loads (no change in time), interleaving two accumulation groups with a second lookahead (register spills of 72 to 120 bytes, slower), the `m16n8k16` form of the old long-context attention kernel (-1 percent on its own: that kernel loaded its tiles through registers; with cp.async tiles and fp16-accumulate scores the same instruction form is the 2x described under Long contexts), a larger prefill chunk (2,048 tokens: +2 percent on the 4B, -4 percent on the 9B). Still open: an int8 path for Q8_0 and the legacy formats, a chunked form of the Gated DeltaNet recurrence on tensor cores (the scan is about 6 percent of a 16,000-token pass after the change described under Long contexts) and a faster Q6_K integer kernel (15 against 22 TMAC/s for Q4_K).

## Video memory

Video memory held by the server above what was in use before it started (nvidia-smi, context 4,096, MiB), right after loading. With speculation the engine reserves the rows of its verification pass at the first request, so the last two columns are after one chat:

| model | LM Studio engine | llama.cpp b11517 | ForgeInfer before 2026-10-13 10:30 | ForgeInfer now | now + lookup | now + head |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 5,238 | 5,266 | 6,178 | 5,521 to 5,551 | 6,154 | 6,477 |
| Qwen3.5-4B | 2,954 | 2,973 | 3,423 | 3,377 | | 4,224 |
| Qwen3.5-0.8B | 691 | 708 | 849 | 819 to 839 | | 1,052 |

Two things changed the first column of ForgeInfer into the second:

- **The embedding table of a model with its own output matrix lives in page-locked host memory** (Qwen3.5-9B, Llama, Qwen3-8B and larger; a tied output matrix reads the whole table every token and keeps it on the device). One row of 2 to 4 KB is read per token, so it costs nothing measurable: Qwen3.5-9B decodes at 46.6 against 46.5 tokens/s, the first token takes the same 30 / 242 / 778 ms, and the answers are identical (5 of 5 workloads, plain and with the head, on the RTX 4060; on an RTX 2060 SUPER under Linux a short and a 1,153-token prompt give identical answers and the card holds 544 MiB less). llama.cpp keeps the table in host memory too (its log for the 9B: `CPU_Mapped model buffer size = 545.62 MiB`, `CUDA0 model buffer size = 4861.28 MiB`, KV cache 128 MiB, recurrent state 50 MiB, compute buffer 108 MiB). The memory it frees lets the speculative pass reserve 12 verification rows instead of 8 on the 9B (the memory check wants 200 MiB to stay free: with the table on the device 12 rows were refused and, with a little less memory free, 8 as well), which made the head-assisted answers up to 10 percent faster (grammar fix 155 to 170 tokens/s). `FORGE_HOST_EMBED=0` keeps the table on the device; a failed allocation of page-locked memory falls back to the device by itself.
- **The fp16 dequantization scratch of the prompt path** (50 to 100 MiB) is taken when a large matrix needs it, not at load: the fused kernels read the quantized blocks directly and on Ada never use it. A 4 MiB buffer serves the small matrices that have no fused kernel (the 32-row Q8_0 projections of a Gated DeltaNet layer).

What is left over llama.cpp (0.3 to 0.4 GiB) is mostly the buffers of a 1,024-token prompt chunk (280 to 350 MiB; chunks of 512 tokens save 128 MiB and cost 1 to 5 percent of prompt speed, chunks of 256 save 190 MiB and cost 6 to 10 percent, so they stay at 1,024) and, with speculation, one recurrent state per verified row (about 50 MB each).

Ready to answer after the process starts, model file in the page cache: 9B 4.4 s for the engine (6.9 s with the head), 4.5 s for LM Studio's engine and 3.0 s for llama.cpp b11517; 4B 2.9 s, 2.1 s and 2.0 s.

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

## Long contexts

Qwen3.5 is a hybrid: only one layer in four attends over the whole context, the others keep a fixed-size recurrent state, so a long prompt costs little. Qwen3.5-4B Q4_K_M on the RTX 4060 (`forge prefill-bench MODEL --native --tokens N`, prompt of N tokens, then 128 tokens of greedy decoding at that depth; before the prompt work of 2026-10-13 the three prefill rates were 2,598, 2,329 and 1,969). Prefill rates of the table are with the changes below (the pass before them was 31 percent attention at 16,000 tokens and read 3,015, 2,667 and 2,193 tokens/s at these contexts; decode rates move by 5 to 10 percent with the state of the card from one hour to the next, the same binary gave 71.7 and 75.9 at 4,096):

| context | prefill, tokens/s | decode, tokens/s |
|---|---|---|
| 4,096 | 4,402 | 71.7 |
| 12,288 | 3,864 | 68.9 |
| 24,576 | 3,266 | 63.4 |

How far it goes on an 8 GB card (4B, `forge serve`, a fact planted in the middle of a synthetic prompt, which the answer must repeat; `tools/needle_check.py`): 16,000, 32,000, 64,000 tokens with the fp16 KV cache, and **129,000 tokens with `--kv int8`** (the 4B's KV cache is 4.2 GB at 128K positions in fp16 and 2.1 GB in int8; with fp16 the card runs out of video memory and the run does not finish): the prompt is read in 83 seconds (1,555 tokens/s) and the fact is found; decoding at that depth runs at 17 tokens/s.

**The long-prompt work of 2026-10-13 (afternoon and evening).** A per-kernel profile of a 16,000-token prompt (`FORGE_KPROF=2 forge prefill-bench MODEL --native --tokens 16000`, the 4B) showed the attention kernel at 31 percent of the pass, the integer GEMMs at 48 and the Gated DeltaNet kernels at 14. Four changes, each measured against the previous state in the same hour (best of six passes):

1. **Ampere attention kernels** (`flash_a80.cu`, sm_80 and newer; `FORGE_FLASH_A80=0` keeps the old ones). The old kernels were written for Turing: K and V tiles went through registers and the `m16n8k8` MMA form runs at half the tensor-core rate on Ampere. The new ones use `m16n8k16`, a double-buffered cp.async pipeline for the K and V tiles, no mask on tiles that lie entirely below the diagonal and the heaviest blocks first; all accumulators are fp32 and the results are **bit-identical to the old kernels** at head width 128 and 256. A prompt of 16,000 tokens of the dense Qwen3-1.7B (head width 128) is read 12 percent faster (6,000 to 6,680 tokens/s).
2. **Warp pairs for Qwen3.5's 256-wide gated attention.** The accumulators of 64 rows x 256 output dimensions do not fit one warp, so the old kernels gave each half of the dimensions its own block and each block computed the whole scores S = Q K^T: 1.5 times the tensor work. In the new kernel a block of 8 warps is four pairs; a pair shares 16 query rows, each warp computes S for 16 of the 32 keys of a tile, the halves meet in shared memory behind a 64-thread named barrier, and each warp runs the softmax over all 32 keys and P V for its own 128 dimensions. The tiles (32 keys x 256 dimensions, two buffers, 64 KB) are not padded but XOR-swizzled (the 16-byte chunk c of row r sits at c ^ (r & 7)), which keeps ldmatrix and cp.async conflict free and lets 32-key tiles fit. The kernel for a chunk of 1,024 tokens at position 14,336 takes 11.1 instead of 21.3 ms (1.9x), at 76 percent of the card's fp32-accumulate tensor peak; the output is bit-identical to the old kernel (the 32-key online-softmax tile and the summation order are the same), `compute-sanitizer` (memcheck, racecheck, synccheck, initcheck) is clean.
3. **Gated DeltaNet scan with several state columns per thread** (`gdn_scan_nc_*`, bit-identical to the old kernel; `FORGE_SCAN_NC=0` keeps it). The scan was not limited by arithmetic (3 FMAs per state element and token) but by the shared-memory pipe: a warp issued 12 LDS.128 per token to serve 4 state columns. With 2 or 4 columns per thread the same loads serve 8 or 16 columns, in blocks of 128 threads, and every column is summed in the order of the old kernel, so outputs, state and convolution window are identical bit for bit. The scan takes 740 instead of 1,230 microseconds per layer for 1,024 tokens of the 4B; a 16,000-token prompt is read 4 percent faster.
4. **The prep kernel for 8 tokens per block** (`gdn_prep_tile_kernel`, also bit-identical): 535 to 427 microseconds, which is 84 percent of the card's memory bandwidth for the three fp32 tensors it writes.

All four together on the 4B (`forge prefill-bench`, 16,000 tokens, one window): 3,245 tokens/s with the old kernels (`FORGE_FLASH_A80=0 FORGE_SCAN_NC=0`), 3,871 with the new ones (+19 percent). Same hour, same files, `forge prefill-bench` against `llama-bench -p N -n 0` of llama.cpp b11517 (4B unless noted), tokens per second:

| prompt | llama.cpp | ForgeInfer | |
|---|---|---|---|
| 4,096 | 3,476 | about 4,450 | +28 percent |
| 16,384 | 3,078 | 3,967 | +29 percent |
| 16,384, Qwen3.5-9B | 2,030 | 2,467 | +22 percent |
| 32,768 | 2,859 | 3,258 | +14 percent |
| 65,536 | 2,353 | 2,466 | +5 percent |

The lead shrinks with the length because the attention takes a growing share, and llama.cpp's attention kernel accumulates the output in fp16 (about twice the tensor rate for P V, at some cost in accuracy); ForgeInfer's accumulates in fp32 everywhere and is bit-identical to its own earlier kernels.

**One more, small:** the feed-forward layer's SwiGLU and the int8 quantization of the down projection's input run as one kernel (`swiglu_quant_rows_kernel`; `FORGE_SWIGLU_Q=0` keeps the two): the fp16 activations are never written or read back, the values are rounded through fp16 exactly as the separate kernel stores them, so the logits are bit-identical (tests on a hybrid and a dense model). The two kernels took 374 microseconds per layer for 1,024 tokens of the 4B, the fused one 231: 1.6 percent of a prompt pass (inside the noise of a whole-prompt benchmark, which moves by 5 percent with the card's clock).

What did not help: an fp16-accumulate form of the scores S (flushed into fp32 after every k16 step: 3 to 4 percent more prompt speed for a KL of 4e-4 on the 4B and 1.2e-3 on the dense 28-layer Qwen3-1.7B against the old kernels, so it was removed); fusing the gate and up projections of the feed-forward layer with SwiGLU into one integer GEMM (bit-identical, but the pass got no faster: the integer GEMMs run compute-bound, so the saved 75 MB of elementwise traffic per layer hides behind the MMAs); a conditional rescale of the attention accumulators; a warp per token for the gated RMSNorm after the scan (as fast: it reads and writes what the bandwidth allows); spreading, staggering or trickling the cp.async issue of the integer GEMM, and 16 warps of 32 x 32 instead of 8 of 64 x 32 (all within noise of 25 TMAC/s: the kernel is latency-bound with two warps per scheduler).

## Chat follow-ups at long context, and going back to a chat

A follow-up in a chat that already holds a long document is the common long-context case: 30 to 40 new tokens after thousands. Measured on the RTX 2060 SUPER (34 SMs, 448 GB/s) with `tools/followup_probe.py MODEL [paragraphs]` (a document of N paragraphs and a question, then short follow-up questions, each answered with up to 24 tokens) and `tools/cache_check.py`, three things made it slow, none of them the prompt GEMMs:

**1. The decode attention ran one block per head.** The dense decode attention kept every score of a head in shared memory (4 bytes per position of the context the session was started with) and ran one block of 128 threads per head: 32 blocks of 4 warps on a card with 34 SMs, too little to hide the latency of the loads of keys and values. After 6,400 positions it read the KV cache at 49 GB/s. The split-KV kernels (blocks of 1,024 positions, an exact merge; the output is the same up to the rounding of the sum, `crates/forge-gpu/tests/attn_split.rs`) existed for contexts beyond 11,000 positions only, where the one-block kernel no longer launches. They are the default for every session started with a context of more than 1,024 positions now, dense and hybrid models alike (`FORGE_ATTN_SPLIT=0` keeps the old kernel where it can launch); on this card they read the cache at 276 GB/s. A whole conversation was faster by more than the attention alone: the chunks of a prompt that go through the multi-row path (below) use the same kernels.

| model, context | decode per token, one block per head | split-KV | the 40 new tokens of a follow-up, one block per head | split-KV |
|---|---|---|---|---|
| Qwen3-4B Q6_K, 200 positions | 10.0 ms | 10.0 ms | 61 ms | 61 ms |
| Qwen3-4B Q6_K, 1,700 | 14.6 ms | 12.6 ms | 135 ms | 99 ms |
| Qwen3-4B Q6_K, 6,400 | 29.2 ms | 13.4 ms | 900 ms | 230 ms |
| Qwen3.5-4B Q4_K_M, 1,700 | 9.3 ms | 8.4 ms | about 100 ms | about 95 ms |
| Qwen3.5-4B Q4_K_M, 6,400 | 12.7 ms | 8.8 ms | 187 ms | 115 ms |

The compressed KV caches gain the same, because the attention kernels read the layer's fp16 scratch either way: with `--kv int8` or `--kv int4` the 24 tokens of an answer after 6,400 positions take 419 ms instead of 790 on the dense 4B (the greedy answers agree except where two words are a near tie and the order of summation decides, as it does between the two attention kernels with the fp16 cache).

**2. A short chunk took the multi-row path whatever the context.** Chunks of up to 56 tokens run through the multi-row kernels (the weights are streamed once for groups of 16 rows), whose attention reads the whole cache again for every row, while the GEMM path's attention reads each tile of keys once per 64 rows. After 6,400 positions the multi-row path costs 5.6 ms per new token, the GEMM path 102 ms for any chunk of 16 to 56 tokens:

| new tokens after 6,400 positions | 11 | 16 | 21 | 26 | 31 | 41 | 51 |
|---|---|---|---|---|---|---|---|
| multi-row path | 63 ms | 87 ms | 122 ms | 146 ms | 170 ms | 229 ms | 286 ms |
| GEMM path | 145 ms (token by token) | 102 ms | 101 ms | 101 ms | 102 ms | 103 ms | 103 ms |

The crossover moves from 56 tokens on an empty cache to 35, 26 and 18 after 1,600, 3,200 and 6,400 positions (37, 30 and 22 once the attention of the multi-row path shares its loads between four rows, see below), so a dense model's limit now follows the context (`FORGE_MROW_CTX=0` keeps the fixed 56; the rule scales with the model's bytes of KV cache per position). The hybrid Qwen3.5-4B stays on the multi-row path at every context: its attention is a quarter of its layers, and the path won 70 to 125 ms against 120 to 140 ms at 1,700, 3,200 and 6,400 positions. A follow-up of 40 new tokens after 6,400 positions reads in 104 ms instead of 900 (8.7 times), and the whole request (40 new tokens, 24 generated) takes 0.43 s instead of 1.4 to 1.6.

The same cost hits the verification of drafted tokens (prompt lookup): an answer that copies text out of a 6,400-token document ran at 109 tokens/s against 72 without the lookup, and at 400 against 87 on a 436-token document, because the 12 rows of a verification pass each read the cache again. The attention of the multi-row path now reads the keys and values once for four rows (`attn_split_rows4_kernel`, whose partial results are bit-identical to the one-row kernel's; `FORGE_ATTN_ROWS=0` turns it off): 4.6 instead of 5.6 ms per token of a chunk after 6,400 positions, 134 instead of 109 tokens/s for the copying answer. That kernel is bound by latency, not by memory (three blocks of four warps per SM), so the next step is one on the tensor cores (a block of 16 rows against 128-key tiles), which was not done.

**3. Going back to an earlier chat read it again.** The engine holds one conversation on the GPU; a request that shared little with it threw it away. With a second chat in between, a return to a 6,400-token chat cost its whole prompt (3.1 to 5.3 s here). Now the conversation is kept in page-locked host memory (`--prompt-cache-mb`, default 2,048; see [ENGINE.md](ENGINE.md#what-the-engine-does-about-memory)) and copied back. `tools/cache_check.py` replays two such chats that take turns (A, B, A, B, A, B; greedy answers, identical with the cache off and on; the figures are one request each, the copy of the chat that goes out included):

| model | return to a 6,400-token chat, cache off | cache on |
|---|---|---|
| Qwen3-4B Q6_K (dense, 1 GB of KV cache) | 3.46 s | 0.58 s |
| Qwen3.5-4B Q4_K_M (hybrid, 250 MB) | 3.14 s | 0.38 s |
| Qwen3.5-9B Q4_K_M with its prediction head | 5.25 s | 0.52 s |
| Qwen3.5-27B (feed-forward layers on the CPU, int8 KV cache, head), chats of 2,500 tokens | 19.3 to 20.4 s | 3.8 to 4.4 s |

On the 27B the answers of five of six turns were identical with the cache off and on; the sixth differed by one word at a near tie, exactly as it does when the same chat simply goes on without another chat in between (the engine's own reuse of the previous turn, same six turns in the order AAABBB with the cache off: the same word differs): the entries of the earlier answer were written by the decode kernels, a read from scratch writes them with the prompt kernels, and the two round differently. Two runs of the engine with the cache off agree on all six answers. The rest of a return is reading the new 40 tokens (0.1 s) and answering (0.3 s on the dense 4B); the copies are 0.08 s each way for the dense 4B (12 GB/s over PCIe) and 0.02 s for the hybrid ones. Locking 1 GB of host memory takes 0.3 s and unlocking it as long, which made the first version of the cache cost 0.46 s per swap on the dense model; it keeps one spare buffer now (an eighth bigger than what it holds, so the next version of a growing chat fits), and a swap costs the two copies (0.16 s on the dense 4B).

## What the integer GEMMs cost in quality at long context

The integer prompt GEMMs quantize the activations to int8 (one scale per 32 values, as llama.cpp's mul_mat_q does). Against the plain fp16 path (`forge ppl --dump-ref` with `FORGE_MMQ=0 FORGE_F16ACC=0 FORGE_FLASH_A80=0 FORGE_SCAN_NC=0`) the Qwen3.5-4B on a 16,000-token text of English prose and tables shows a KL of 0.0152 (top-1 token equal at 98.05 percent, +0.0096 nats per token, perplexity 12.38 to 12.50); the same text in windows of 512 tokens shows 0.0010. With `FORGE_MMQ=0` and everything else at its default the KL is 0.00009, so the new attention, scan and fusion kernels add nothing. The error is not there from the start: the first 2,000 tokens are at 0.0008, then it sits on a plateau of 0.006 to 0.023 that moves with the text and does not keep growing (65,536 tokens: 0.0089 on average). It is carried by very few positions: the worst 0.1 percent of them hold half of the KL and the worst 1 percent hold 92 percent, confident predictions where the text offers several continuations (the next word after a date in a changelog, a list of links) and the int8 path commits to another one that also occurs in the text; the median position is at 0.0003. The same effect shows when the single-token decode kernels, a separate implementation, are compared with the fp16 path on a second GPU (KL 0.0161 at 16,000 tokens, 0.0009 at 512), so it comes from the int8 activations and not from the integer GEMM kernels; the 0.8B shows none of it (0.0008 at both lengths), and the 9B shows a KL of 0.0098 at 16,000 tokens (0.0019 at 512; the decode kernels give 0.0108) with no change in perplexity (+0.0005 nats per token for the emulated int8 rule, -0.0004 for the decode kernels). Retrieval is not hurt: 16 access codes spread over 16,000 tokens, and 16 of 790 near-identical lines, are found 48 of 48 times by the fp16 path and by an emulation of the int8 rule (`tools/retrieval_check.py`). Keeping a matrix out of the integer path shows where it comes from (`FORGE_MMQ_SKIP=role,...` with the roles `dn_qkv dn_z dn_out attn_q attn_kv attn_o ffn_gate_up ffn_down`, groups `dn attn ffn all`; those matrices then take the fp16 GEMM):

| skipped | KL | extra NLL | prompt speed (8,000 tokens) |
|---|---|---|---|
| nothing | 0.0152 | +0.0096 | 4,206 tokens/s |
| `dn_out`, the DeltaNet output projection | 0.0113 | +0.0023 | 4,096 (-2.6 percent) |
| `dn`, all three DeltaNet matrices | 0.0060 | +0.0029 | 3,766 (-10.1 percent) |
| `attn` | 0.0143 | +0.0083 | |
| `ffn` | 0.0133 | +0.0071 | |

So the output projection of the DeltaNet layers alone carries three quarters of the perplexity loss, for 2.6 percent of the speed. Nothing is changed by default: the loss is 1 percent of perplexity at 16,000 tokens, measured on one model and one text. `FORGE_FAKEQ="roles:group:mode;..."` (quantize and dequantize in place before the fp16 GEMM, run with `FORGE_MMQ=0`) emulates int8 activations on any GPU and compares schemes the integer kernels do not have, on the 4B at 16,000 tokens (KL against the plain path): the engine's rule 0.0207, groups of 16 values 0.0098, groups of 64 0.0254, **an asymmetric range with a zero point 0.0049**, the largest value of each group kept exact 0.0044, two int8 passes 0.00005. A zero point would cut the KL by four and the perplexity loss by half; it is not implemented in the integer kernels.

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
python tools/bench_engines.py Qwen3.5-9B-Q4_K_M llama forge-plain forge-lookup forge-head   # five chat workloads, tokens/s, video memory, time to ready
python tools/bench_engines.py --prefill 2100 Qwen3.5-9B-Q4_K_M llama forge-plain            # prompt reading speed from the servers' timings
python tools/teacher_force.py Qwen3.5-9B-Q4_K_M bench_Qwen3.5-9B-Q4_K_M.json llama          # how often forge's top-1 equals the token llama.cpp produced
python tools/followup_probe.py MODEL.gguf 100                                                    # a chat follow-up after a 6,400-token document: time to read and to answer
python tools/cache_check.py MODEL.gguf 100                                                      # two long chats that take turns, prompt cache off and on, answers compared
tools/bench_vs_llama.sh        # forge vs llama.cpp on the same files (needs llama.cpp's llama-bench)
forge gpu-bench ...            # decode microbenchmarks;  forge prefill-bench ...
forge ppl / eval-*             # perplexity and KL against a reference, HellaSwag and friends
```
