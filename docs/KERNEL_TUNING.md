# Trainer kernel tuning for the homelab GPU (RTX 2060 SUPER, 34 SMs) - track H2

Branch `gpu-tune`. Goal: more training tokens/s at the production configuration (150M Forge-1, micro 2, 65,536 tokens per step) on the 34-SM card, without slowing other GPUs.
Everything below is switched by the SM count at run time: the new attention path is the default only on a 34-SM device; every other device keeps the previous kernels and the previous split-K constants
(`FORGE_ATTN_DS=1` forces the new path on, `FORGE_ATTN_DS=0` off).

## What was changed

| change | where | result (2060 SUPER) |
|---|---|---|
| **dS spill for the dense attention backward.** dK dV already holds dS^T (fp16) in registers for its own dK product; it now also writes the 64 x 64 tiles to a spill buffer (`flash_bwd_dkdv_ds_64`). dQ is then one matmul per tile (`flash_bwd_dq_ds_64`, dQ += dS K) instead of three (S, dP and dS K are not recomputed). dq, dk, dv are bit-identical to the old kernels (test `dense_backward_with_the_ds_spill_is_bit_identical`). Cost: 38 MB of spill buffer per sequence (micro 2: 76 MB), dK dV +2..5 % | flash_train.cu, forge-train | dQ kernel 586 -> 258 us per launch (-56 %), dK dV 982 -> 843 us (with the next row), attention backward -31 % at 4096 tokens (kbench) |
| **longest-block-first launch order over the whole grid** for dK dV (more than one sequence), dQ and the dense forward (flag through the unused `window` argument). The grid's own order (key tile fastest, then head, then sequence) started the longest blocks of the second sequence last: list-scheduling model for micro 2 gives a makespan of 183 against 156 block-iterations for dK dV and 110 against 94 for dQ. Bit-identical. A single sequence keeps the old order for dK dV (measured +4 % slower there) | flash_train.cu | dK dV -14 % (4096 tokens), dQ another -7 %, forward kernel -7 % (KPROF) |
| **split-K model refitted for 34 SMs** (`split_k_turing`: overhead 6 k-tiles and 0.7 for a lone block instead of the T4's 24 and 0.55). The T4 constants split 576 x 576 and 960 x 576 too little here: kbench 576 x 576 x 4096: 0.168 -> 0.145 ms, 960 x 576 x 2048: 0.157 -> 0.127 ms. Other SM counts unchanged (test pins both) | forge-train | weight-gradient GEMMs -1.7 % |
| `forge kbench --only attnds`: interleaved A/B of the old and the spill attention backward | forge-cli | tool |

## Per-kernel time, KPROF=2, 15-layer proxy at micro 2 (8 steps; the profiler serialises launches)

| kernel | before | after |
|---|---|---|
| flash dK dV (96 x 2 blocks), per launch | 982 us | 843 us (includes the dS stores) |
| flash dQ, per launch | 586 us | 258 us |
| flash forward, per launch | 417 us | 385 us (-7.4 %) |
| gemm_tn_add, all shapes, total | 3458 ms | 3399 ms |
| all kernels, total | 17135 ms | 16161 ms (-5.7 %) |

Unchanged (within 1-2 %): the forward GEMMs, the data-gradient GEMMs, the head, rmsnorm / mix / qkv kernels.

## Throughput, interleaved A/B (base = main of 2026-10-11, new = this branch), 12 steps per run, the first 3 ignored, alternating, same data

The homelab GPU cannot run micro 2 of the 30-layer model while the chat engine holds 2.7 GB (micro 1 already needs 5.0 GB; micro 2 fails with an out-of-memory error). So:

* **micro 2, 15 layers (value embeddings 0,1,13,14), 65,536 tokens per step** (same GEMM and attention shapes as production, 97M parameters):
  base 33,335 / 32,842 / 32,771 / 32,749 against new 35,125 / 34,878 / 34,825 / 34,819 tokens/s: +5.4 %, +6.2 %, +6.3 %, +6.3 % (mean 32,924 -> 34,912, **+6.0 %**).
* **micro 1, full 30 layers, 65,536 tokens per step**: base 16,643 / 16,610 / 16,588 against new 17,527 / 17,528 / 17,514: +5.3 %, +5.5 %, +5.6 % (**+5.5 %**).
  (Intermediate: the dS spill alone gave +3.1 % at micro 1 and +3.75 % at the micro-2 proxy; the launch order added about +1.5 %.)

Loss trajectory (15-layer micro-2 proxy, 200 steps, same seed and data): base against a second base run differs by 2.3e-4 relative per step (mean; the trainer is not deterministic: atomics), base against new by 2.9e-4,
the second base against new by 1.3e-4: within the run-to-run noise. Final losses 4.9633 / 4.9624 (base, base) and 4.9623 (new).

## Tests (homelab, `cargo test --release -p forge-train -p forge-gpu --no-fail-fast`)

All pass (forge-gpu: gemm 3, gemm_s80 4, gpu_forward 4, mtp, native, prefill_variants, qformats, qwen35 3, verify2 3; forge-train: attn_bwd, decode 6, f16acc 4, gemm_e2 3, generation, ignore 2, kopt 12, muon 2,
muon_opts 3, tiny 17, window 3). New: `dense_backward_with_the_ds_spill_is_bit_identical` (three shapes incl. 2048 tokens x 9/3 heads and two sequences), the forward launch-order flag in
`new_attention_kernels_are_bit_identical_in_the_dense_case`, and the 34-SM factors in `split_k_model_picks_the_measured_best_factors`. The golden gradient tests (tiny.rs) run with the new path on this card.

## What did not help or was not done

* **Per-shape split-K / tile selection** (idea 1) is worth little at production: with the T4 model the wrong choices cost only 0.3-0.5 % of the step (576 x 576 and 960 x 576 at 2048/4096 tokens); that is the refit above.
  The wave quantisation of the N = 576 forward GEMMs (160 blocks of 128 x 128 on 68 slots) is already handled where it pays (128 x 144 tiles for the fp32 data gradients); the residual-epilogue variant measured 0.85-1.02x
  earlier, because the 128 x 144 kernel has no staged epilogue. A 128 x 144 kernel with the staged epilogue could give about 0.5-1 % (not done).
* **Forward launch order**: -7 % of the forward kernel in KPROF, but no change visible in the whole-step A/B (+-0.4 %); kept (bit-identical, gated with the spill).
* **dK dV** (idea 2): 230 registers, 2 blocks per SM, 58 % of the tensor peak, instruction mix 256 HMMA to about 1300 others per iteration per warp; no cheap register or occupancy change found (K/V fragments
  in registers are the reason; moving them to shared memory frees 32 registers, not the 60+ needed for three blocks). Splitting the longest blocks over the three query heads would cut the model makespan
  from 157 to 149 iterations (about 0.5 % of the step) but needs atomics on dK dV: not done.
* **dK dV block order for one sequence** (micro 1): the whole-grid order was 4-9 % slower than the grid's own order for the kernel alone, so one sequence keeps the old order. dQ gains in both cases.
* **Small kernels** (idea 3): rmsnorm_mix_bwd moves 8.5 fp32 tensors in 107 us per 2048 tokens (about 400 GB/s: at the memory roofline); the other row kernels are similar. Fusing them saves a pass only by changing
  what is stored. The launch gaps are not a problem either (GPU busy, the main thread sits in launches at 100 % CPU but queues far ahead).
* **Muon**: the whole optimiser costs about 75 ms of a 3.9 s step (measured from steps with one and two micro-batches); batching the equal-shaped matrices would save at most half of that.
* The T4 (40 SMs) was not measured: the new path is off there by default. `forge kbench --only attnds` prints the interleaved comparison; if it wins on a T4, set `FORGE_ATTN_DS=1`.

# Q8 weights for the checkpoint decoder (`forge serve CKPT --q8`) - branch `q8-decode`, 2026-10-12

One decoded token streams every weight once, so the decode speed of the chat engine is the memory bandwidth over the bytes of the weights: 596 MB per token for the 365M model in fp16 (the 24 layers' matrices and the tied head, 297.8M weights; the value-embedding tables and the embedding row are gathers) on a card whose plain streaming read reaches 424 GB/s. Q8_0-style weights (int8 and one fp16 scale per 32 weights, 1.0625 bytes per weight) read 316 MB (53 percent).

## What was built

* `kernels/decode_q8f.cu` (self-contained, included by `all.cu`), `forge-train/src/decode/q8.rs` (`Decoder::enable_q8(tr, release)`), glue in `decode.rs`, `forge-engine` (`WeightMode::Q8` on a `forgeckpt1` checkpoint), `forge q8-eval`, `forge gen-check --q8 --no-ref`.
* Quantization on the device from the fp32 master weights (`q8f_quantize`: d = fp16(amax / 127), q = round(x / d), the Q8_0 rule with the scale rounded to fp16 first): 316 MB of matrices of the 365M model in 8 to 10 ms. Matrices: q|k|v (one [q_dim + 2 kv_dim][n_embd] matrix, as the three weights lie in the flat parameters), the output projection, up (or gate|up of a SwiGLU model), down, the tied embedding (the head; an untied model quantizes both). Kept in fp32 / fp16: the norms, the layer scalars, the value-embedding tables (134 MB for the 365M model: 8 tables of 32768 x 256 fp16).
* Layout: 16 rows form a tile, along K a tile is cut into super-chunks of 64 weights per row (1088 bytes: two 512-byte blocks of 16 rows x 32 k, then their 64 bytes of fp16 scales). The 512 bytes of a block are stored in the order in which the lanes of a warp need them as the A operand of `mma.m16n8k16`: one coalesced `LDG.128` per lane, no shared-memory staging, no `ldmatrix`. Bytes are q + 128, so that PRMT with 0x64 as the high byte and one HSUB2 give q exactly in fp16.
* Matrix-vector kernel (`q8f_gemv_*`: the same five fusions as `f1d_gemv_*`: layer-input mixing + norm prologue, norm prologue, fp16 vector prologue; fp16, fp32 + residual, ReLU^2 and fp32 logits epilogues): the WEIGHTS are the 16 x 16 A matrix and the fp16 activation is column 0 of the 16 x 8 B matrix (the other columns are zero), so one mma covers 256 weights (the fp16 kernels cover 128 with the weights as B). Per 32-weight block: two mma, then `tot += d * acc` in fp32 (the fp16 scale is applied per block and row, so the products are exact in fp32 and only the weights are rounded). `W` warps per CTA (4), `S` of them share a tile and split K (reduced through shared memory; `S` from the number of tiles: enough for about 128 tile slices, at most 4, `FORGE_Q8_WARPS`, `FORGE_Q8_SPLIT`), batches of 3 super-chunks with the next batch loading while one is consumed, the first batch requested before the prologue, the prologue (all threads: norm of the fp32 input, fp16 permuted into the order of the B fragments) overlapping its latency.
* Prompt pieces (more than one token): every layer matrix is dequantized into one fp16 scratch buffer (8.4 MB for the 365M model; `q8f_dequant`, one warp per 512-byte block) and the trainer's tensor-core GEMM runs on it; the head of a piece goes through the Q8 matrix-vector kernel, the embedding rows through `q8f_embed_rows`.
* `release` (`forge serve --q8`): the trainer's fp32 master weights and fp16 copy are freed after the Q8 matrices and the small parameters (norms, scalars, value-embedding tables) were copied to compact buffers: `Trainer::weights_released` makes every other use of that trainer an error. Without `--q8` the decoder is the old one, bit for bit (tests `decode` 6 and `decode_q8::the_default_decoder_is_unchanged`).

## Numbers (homelab RTX 2060 SUPER, 424 GB/s plain streaming read; the chat engine of the user holds 2.7 GB next to it)

| 365M chat model (`tpu2_chat_2300.bin`) | fp16 | Q8 | |
|---|---|---|---|
| decode, greedy, `forge serve` + `tools/serve_bench.py` | 484 tok/s | 766 tok/s | +58 % |
| decode, top-k 40 / top-p 0.95 and min-p / repeat penalty (CPU sampling) | 419 / 441 | 644 / 702 | +54 % / +59 % |
| decode loop of `gen-check --no-ref` (3 runs each: 462.5-463.2 / 741.2-742.5) | 462.8 | 741.7 | +60 % |
| weights on the device (`serve` line) | 730 MB | 451 MB (316 MB of Q8 matrices + 134 MB value-embedding tables) | -38 % |
| GPU memory of the server process (nvidia-smi, 2048-token context) | 2574 MiB | 920 MiB | -64 % |
| prefill of a 1,219-token prompt | 46 ms (26.4K tok/s) | 52 ms (23.3K tok/s) | +6 ms |
| load (file read dominated; 12 s from the disk, 0.6 s from the page cache) | | + 8 ms of quantization | |

| 150M chat model (`tpu1_patch11_drills.bin`) | fp16 | Q8 | |
|---|---|---|---|
| decode, greedy (`serve_bench.py`) | 696 tok/s | 1101 tok/s | +58 % |
| decode, top-k / top-p and min-p / repeat penalty | 588 / 640 | 859 / 948 | +46 % / +48 % |
| decode loop of `gen-check --no-ref` (3 runs: 664.6-666.2 / 995.5-999.8) | 665.6 | 997.5 | +50 % |
| weights / GPU memory of the process | 300 MB / 1250 MiB | 183 MB / 568 MiB | -39 % / -55 % |
| prefill of 1,219 tokens | 32 ms (37.7K tok/s) | 35 ms (34.5K tok/s) | +3 ms |

Kernel by kernel (stand-alone `tools/q8dev.cu`, cold weights, back-to-back launches, two runs: the clock of this card drifts by up to 10 percent): 365M shapes q|k|v 1.67 MB 7.3-8.7 us, output projection 1.11 MB 5.4-6.3 us, up 4.46 MB 14.5-15.6 us, down 4.46 MB 14.0-15.2 us, head 35.65 MB 88-91 us (390-405 GB/s, 92-96 percent of the streaming read); 150M shapes 0.59 MB 5.0-6.0 us, 0.35 MB 3.6-3.8 us, 0.94 MB 5.1 and 5.5 us, head 20 MB 50-52 us. The fp16 kernels in the real step (KPROF, 99 tokens): output projection and down 17.5 us on average, up 27.2 us, q|k|v 16.5 us, head 166 us; the Q8 ones 12.0, 17.2, 10.2 and 91 us. A token of the 365M model: 2.16 ms in fp16, 1.35 ms in Q8.

Quality, `forge q8-eval` (both decoders fed the same tokens one at a time, 16 windows of 512 tokens per file, held-out text: `val_domains` web / wiki / python and the chat validation shard of the SFT mix, whose masked tokens are not scored):

| | Q8 - fp16, nats per token (paired, standard error 0.0002-0.0003) | logits: largest diff / largest logit, mean over steps (worst) | mean diff / mean abs logit | top-1 agreement | KL(fp16 / Q8) |
|---|---|---|---|---|---|
| 365M web, wiki, python, chat val | +0.0001, -0.0000, -0.0000, +0.0000 (fp16 3.01, 2.69, 1.58, 1.79) | 6.1e-3 to 7.2e-3 (1.7e-2 to 2.3e-2) | 5.2e-3 to 6.7e-3 | 99.0 to 99.5 % | 0.9e-4 to 1.3e-4 |
| 150M web, wiki, chat val | +0.0002, +0.0002, +0.0002 (fp16 3.22, 2.92, 2.02) | 8.8e-3 to 9.0e-3 (3.3e-2 to 4.6e-2) | 6.7e-3 to 7.0e-3 | 99.1 to 99.3 % | 1.9e-4 |

Greedy text, 20 chat prompts x 64 tokens, each decoder on its own text: 365M 19 of 20 identical (the one difference, prompt 16, starts at token 48, after the answer ended), arg-max agreement along the fp16 text 99.92 % (1279 of 1280); 150M 15 of 20 identical, first differences at tokens 8, 43, 31, 9 and 0 (prompts 1, 10, 13, 15, 17; four of them inside the answer, near-ties of a model whose top two logits are close), agreement along the fp16 text 99.61 % (1275 of 1280). Unit tests (`forge-train/tests/decode_q8.rs`, random Forge-1-shaped, head-dimension-128 / SwiGLU / skips / untied, and sliding-window models): the largest relative logits error stays below 3 percent, the cosine above 0.9998.

## What it costs and what limits it

* Prefill: the dequantization of the 252M layer weights runs once per piece of 512 rows (the decoder's chunk), 2.2 ms each (96 launches of 23 us; it reads 268 MB and writes 504 MB, 1.9 ms at the streaming rate; the first version, a per-element gather, needed 6.9 ms per piece). A 15-token chat prompt costs 7.7 ms instead of 5.9 ms; a 1,219-token prompt 52 ms instead of 46 ms (three pieces). A batched W8A16 GEMM that reads the Q8 tiles directly would remove it (not done).
* The 150M model gains less than the 365M one (+50 percent against +60 percent in the decode loop): its matrices are small (0.35 to 0.94 MB), so the fixed cost per kernel dominates.
* Where the rest of the token goes (365M, Q8, 1.35 ms): the four matrix-vector kernels of a layer 41 to 46 us in the harness (DRAM time of the 11.7 MB of a layer at 424 GB/s: 28 us), the fused attention 7.1 us per layer, the head 91 us, the rest embedding, final norm, graph. Findings: (1) a chain of dependent kernel nodes in a CUDA graph costs 0.8 us per empty node and 1.4 us for a kernel with one dependent load, a barrier and a store; (2) switching off all the dequantization and mma work of a Q8 kernel (or its prologue load) changes its time by 0.1 to 0.5 us, so the compute is hidden and each kernel is 1 to 2 us from "DRAM time plus the graph floor"; (3) the same kernel with weights that stay in the L2 cache (4 MB) is not faster than with cold weights (5.1 against 5.4 us for the output projection, slower for the others), so prefetching the next matrix into the L2 would not pay; (4) batch sizes of 2, 3, 4, 6 and 8 super-chunks and 2, 4 or 8 warps per CTA with 1 to 8 k-slices differ by 5 percent at most (3 and 4 warps / 2 slices were the best, run-to-run noise is 3 percent). What would give more is fewer launches: a persistent kernel for a layer (cooperative launch, the weights of the next stage requested before the grid barrier) or merging the stages; estimated 25 percent on the step, not built.
* CPU sampling (top-k, top-p, repeat penalty over the full 32768 logits) costs 0.12 to 0.28 ms per token and now limits the chat page's settings (644 to 702 tok/s against 766 for greedy): sampling on the device (the `forge-gpu` engine has the kernels) would give back 10 to 15 percent.

## Tools

`tools/q8dev.cu` (stand-alone nvcc check and benchmark of the kernels against a double-precision host reference: `nvcc -O3 -arch=sm_75 --use_fast_math -std=c++17 tools/q8dev.cu -I crates/forge-kernels/kernels`, `--sweep`, `--hot`, `--nb`), `tools/q8_serve_measure.sh FORGE_BIN MODEL [--q8]` (load time, nvidia-smi memory of the server process, `serve_bench.py`), `forge q8-eval CKPT --tokenizer TOK --data A.u16 B.u16 [--windows 16 --seq 512 --tokens 64]`, `forge gen-check CKPT --tokenizer TOK --q8 --no-ref`. Tests: `forge-train/tests/decode_q8.rs` (5), `forge-engine/tests/q8.rs` (1; the engine's greedy text equals a Q8 decoder driven by hand, prefix reuse included).
