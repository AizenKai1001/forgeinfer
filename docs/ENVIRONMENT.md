# Environment variables

Every variable below was found by grepping `crates/`, `tools/`, `cloud/` and `tpu/` for `env::var`, `std::env`, `option_env!`, `os.environ`, `os.getenv`, `${NAME:-default}` and `FORGE_`. "Where" is the file that reads it. A variable that only switches an A/B experiment is called a knob. The tuning knobs were read from the code and its comments; what is not clear from them says `unverified`.

Truthiness differs per variable: many Rust switches test only whether the variable is set (`is_ok()` / `is_err()`), so `FORGE_X=0` still turns them on; those marked "set = on" behave that way. Others compare the value and are marked with their accepted values.

## 1. Rust crates (run time)

| name | where | meaning | default |
|---|---|---|---|
| `FORGE_KPROF` | `forge-cuda/src/lib.rs` (level), `forge-cli` `train_cmd.rs` and `gencheck_cmd.rs` (report) | per-kernel GPU timing; `1` = one row per kernel, `2` = one row per kernel and launch shape; a value that does not parse counts as 1; `0` or empty = off. `train` prints the report when the value is not `0`; `gen-check` prints it when the variable is set at all | off |
| `FORGE_GPU_ORDINALS` | `forge-cli/src/train_cmd.rs` | comma list mapping `--gpus` workers to CUDA device numbers (`0,0` tests the data-parallel path on a one-GPU machine) | worker rank |
| `FORGE_STRICT_FLAGS` | `forge-cli/src/train_cmd.rs` | `1`: `forge train` stops with exit code 2 when a `--flag` of the command line was not used (a misspelling, or a flag that this configuration does not use); otherwise it only prints `WARNING: not used: ...` just before the first step. A numeric flag that is not a number always stops the run. | off |
| `FORGE_NO_KV` | `forge-engine/src/lib.rs` | set = on: load a `forgeckpt1` checkpoint without the KV-cache `Decoder` | off |
| `FORGE_SAMPLE_STATS` | `forge-engine/src/lib.rs` | set = on: sampler statistics switch; what is printed is `unverified` | off |
| `FORGE_BIG_REGS` | `forge-gpu/src/lib.rs` | register budget above which a quantized row uses the streaming kernel instead of the pipelined one | 170 |
| `FORGE_ALLOWED_HOSTS` | `forge-serve/src/lib.rs` | `name,name`: host names a server that listens on a loopback address answers to besides localhost, 127.0.0.0/8 and ::1 (a reverse proxy that passes its public name on needs it; the Host and Origin headers of every request are checked against the list so that DNS rebinding and a script of a foreign web page cannot use the server); `*` turns the check off. Servers on another address (`--host 0.0.0.0`) are not checked | none |
| `FORGE_PARSE_SPECIAL_IN_MESSAGES` | `forge-engine/src/lib.rs` | `1`: special-token strings inside the messages and tool definitions of a chat request (`<|im_end|>`, `<|im_start|>`...) are read as the tokens they name, as Hugging Face and llama.cpp do. By default they are defused (a zero-width space after the first character) so that a pasted document or a fetched web page cannot end a turn or open a system turn; raw `/v1/completions` prompts are not touched | off |
| `FORGE_PREFILL_PIECE` | `forge-engine/src/lib.rs` | tokens of a prompt the engine hands to the model at a time: a cancellation is noticed and a progress event (`return_progress`) is sent between two pieces | 1024 |
| `FORGE_PROMPT_CACHE_MB` | `forge-engine/src/lib.rs`, `forge-engine/src/pcache.rs` | megabytes of page-locked host memory for the prompt cache: the conversation on the GPU is copied there when a request is about to discard most of it, and comes back (12 GB/s over PCIe) when a later request continues it, so going back to a long document costs a copy instead of reading it again. The least recently stored conversations go first; one buffer the size of a conversation is kept spare (and counts against the budget) because locking 1 GB of host memory takes 0.3 s. Conversations of fewer than 1,024 tokens are never kept. `0` = off | 2048 |
| `FORGE_PREFILL_CHUNK` | `forge-gpu/src/lib.rs` | prefill chunk size in tokens; values below 16 are ignored, the value is capped at the compiled maximum and rounded down to a multiple of 16 | 1024 |
| `FORGE_OLD_Q8` | `forge-gpu/src/lib.rs` | set = on: select the older Q8 activation path (`old_q8`); exact effect `unverified` | off |
| `FORGE_DRAFT_ROWS` | `forge-gpu/src/lib.rs` | rows of the vocabulary used by the speculative-decoding draft head (positive integer, capped at the vocabulary size); `tools/draft_sweep.sh` sweeps it | 65536 |
| `FORGE_NO_FUSED` | `forge-gpu/src/lib.rs` | set = on: fused dequantizing GEMMs for K-quants are never used (limit 0) | off |
| `FORGE_FUSED_MAX` | `forge-gpu/src/lib.rs` | K-quant matrices use the fused dequantizing GEMM when the chunk has at most this many tokens | 448 |
| `FORGE_MROW_MAX` | `forge-gpu/src/mrow.rs` | chunks of at most this many tokens of a quantized model (reduction lengths a multiple of 256) run through the multi-row kernels (`decode_m.cu`: the weights are streamed once for groups of 4 / 8 / 12 / 16 rows) instead of the GEMM path; `0` = never. The engine also sends pieces of 2 to 15 tokens to `prefill_batched` when this is on. Measured crossover with the GEMM path on the 2060 SUPER: 62 tokens (Qwen3-4B Q4_K_M), 64+ (Q6_K), 80 (1.7B), 56 (Qwen3.5-4B) | 56 |
| `FORGE_MROW_CTX` | `forge-gpu/src/mrow.rs` | `0`: a chunk of `FORGE_MROW_MAX` tokens or fewer takes the multi-row path at every context. By default a dense model's limit falls as the context grows (the attention of that path reads the whole KV cache for every four rows, the GEMM path's attention once per 64 rows): Qwen3-4B Q6_K on a 2060 SUPER, 40 new tokens after 6,400 positions: 228 ms multi-row, 103 ms GEMM; the limit is 56 tokens on an empty cache and 39, 31 and 23 after 1,600, 3,200 and 6,400 positions (the measured crossovers: 37, 30 and 22), never below 16. Hybrid models and models with their FFN in host memory keep `FORGE_MROW_MAX` | on |
| `FORGE_MROW_GROUP` | `forge-gpu/src/mrow.rs` | rows per group of the multi-row path: 4, 8, 12 or 16 (other values are ignored); the last group of a chunk takes the smallest kernel variant that holds it | 16 |
| `FORGE_MROW_TUNE` | `forge-gpu/src/mrow.rs` | `0` = no autotuning of the multi-row launches (the warps per block are then chosen by the number of rows); otherwise the candidates (4, 8, 12 warps per block) are timed once per launch shape (format, rows, K) at its first use, at warm-up in the engine. The K slices of a launch are a function of the shape alone, so neither this nor the grouping of a chunk changes a bit of the result | on |
| `FORGE_MROW_TUNE_LOG` | `forge-gpu/src/mrow.rs` | set = on: print the configuration the autotuner chose for every launch shape (stderr) | off |
| `FORGE_MROW_TUNE_NJ` | `forge-gpu/src/mrow.rs` | set = on: the autotuner also times up to 3 more K slices than the deterministic rule gives and may pick them, so the summation order and the last bits of the result depend on the timings (experiments on the noise floor of the path only) | off |
| `FORGE_MROW_EVAL` | `forge-gpu/src/lib.rs` | `1`: `eval_chunk` / `prefill_logits` (`forge ppl`) run through the multi-row kernels, the logits of every row of a group from the multi-row LM head (measures the numerics of the path) | off |
| `FORGE_EVAL_DECODE` | `forge-gpu/src/lib.rs` | `1`: `eval_chunk` runs every row through the single-token decode graphs, token by token (a diagnostic: the third implementation, with the same int8 activations as the multi-row path) | off |
| `FORGE_TPB` | `forge-gpu/src/lib.rs` | threads per block of the decode stage kernels (knob: 128 or 256) | 128 |
| `FORGE_SCAN_CPB` | `forge-gpu/src/lib.rs` | columns per block of the Gated DeltaNet scan (Qwen3.5 layers) | 128 if n_v >= 32, 64 if >= 16, else 32 |
| `FORGE_SCAN_NC` | `forge-gpu/src/lib.rs` | `0`: the Gated DeltaNet prompt kernels of one state column per thread (`gdn_scan_kernel_*`, `gdn_prep_kernel`); otherwise blocks of 128 threads with 1, 2 or 4 columns per thread (`gdn_scan_nc_*`, bit-identical results) and the prep kernel for 8 tokens per block. Setting `FORGE_SCAN_CPB` also selects the old scan kernels | on |
| `FORGE_PLAIN_S80` | `forge-gpu/src/lib.rs` | `0`: the dequantize-then-multiply path of the prompt GEMMs uses the Turing-style kernels of `gemm.cu` instead of the Ampere ones of `gemm_s80.cu` (compute capability 8.0 and newer) | on |
| `FORGE_MMQ` | `forge-gpu/src/lib.rs` | `0`: the integer tensor-core prompt GEMMs (`gemm_mmq.cu`, Q4_K / Q5_K / Q6_K, compute capability 8.0 and newer) are off and the fp16 paths are used | on |
| `FORGE_MMQ_MIN_M` | `forge-gpu/src/lib.rs` | shortest chunk (tokens) that goes through the integer GEMMs | 16 |
| `FORGE_MMQ_SKIP` | `forge-gpu/src/lib.rs` | quality study: the prompt matrices of these roles take the fp16 GEMM instead of the integer one. Roles `dn_qkv dn_z dn_out` (Gated DeltaNet), `attn_q attn_kv attn_o`, `ffn_gate_up ffn_down`; groups `dn attn ffn all`; comma separated. On a 16,000-token text of the 4B, `dn_out` alone removes three quarters of the perplexity loss of the integer GEMMs for 2.6 percent of the prompt speed | none |
| `FORGE_MMQ_SKIP_FROM` | `forge-gpu/src/lib.rs` | with `FORGE_MMQ_SKIP`: the roles leave the integer path only for prompt chunks that start at this position or later (the integer GEMMs lose almost nothing in the first 2,000 tokens, a hybrid model's recurrent state accumulates their noise after that) | 0 |
| `FORGE_FAKEQ` | `forge-gpu/src/lib.rs` | quality study: `roles:group:mode;roles:group:mode` quantize and dequantize the fp16 activations of these roles (as `FORGE_MMQ_SKIP`) in place before the fp16 GEMM, so a batched prompt pass on any GPU sees int8-like activations with schemes the integer kernels do not have. Group 16 to 128 values (default 32); mode 0 the engine's rule (scale amax / 127 stored as fp16), 1 fp32 scale, 2 asymmetric min / max with 255 steps, 3 the largest value of a group exact, 4 two int8 passes. Run with `FORGE_MMQ=0 FORGE_SWIGLU_Q=0` | none |
| `FORGE_SWIGLU_Q` | `forge-gpu/src/lib.rs` | `0`: the SwiGLU of the FFN and the int8 quantization of the down projection's input are two kernels (`swiglu_kernel`, `quant_rows_h_kernel`); otherwise one (`swiglu_quant_rows_kernel`: the fp16 activations are never stored, bit-identical results, 374 -> 231 microseconds per 1,024-token chunk and layer on the 4B). Only with the integer GEMMs | on |
| `FORGE_FLASH_A80` | `forge-gpu/src/lib.rs` | `0`: the prompt attention uses the sm_75 kernels (`flash.cu`, `flash256.cu`) instead of the Ampere ones (`flash_a80.cu`, compute capability 8.0 and newer) | on |
| `FORGE_F16ACC` (engine) | `forge-gpu/src/lib.rs` | `1` / `0`: the fused GEMMs of the prompt pass accumulate in fp16 (flushed into fp32 after every k16 step) / in fp32; compute capability 8 and newer only | on for 8.6 and 8.9 (GeForce Ampere / Ada), off elsewhere |
| `FORGE_ATTN_SPLIT` | `forge-gpu/src/lib.rs` | `1` / `0`: the decode attention runs in splits of 1,024 positions with an exact merge / as one block per head. The split kernels are required beyond about 11,000 positions (9,900 for a hybrid model), where the one-block kernel no longer fits its scores in shared memory (`0` is ignored there), and they are the faster ones from about a thousand positions: one block per head leaves a card of 30 SMs with 32 blocks of 4 warps (Qwen3-4B Q6_K on a 2060 SUPER at 6,400 positions: 29 ms per token with the one-block kernel, 14 ms with the split one; the 40 new tokens of a chat follow-up 910 ms against 225 ms; at 200 positions both take 10 ms; the hybrid Qwen3.5-4B, which attends in a quarter of its layers: 12.7 against 8.8 ms per token at 6,400 positions, 9.3 against 8.4 at 1,700) | on when the context the session was started with is longer than 1,024 positions |
| `FORGE_KV_FUSED` | `forge-gpu/src/lib.rs` | `0`: with a compressed KV cache (`--kv int8 / int4`) the decode attention dequantizes the whole context into the fp16 scratch of the layer first, as before. By default (the split attention kernels are in use, a context of more than 1,024 positions) the attention kernels read the compressed store directly (`attn_split_q_kernel`, `attn_split_g_q_kernel`: the group scale outside the sum, the rows of the step itself from the scratch), which skips a pass that wrote and read the context as fp16 for every layer of every token. Qwen3.5-4B, 64,000 positions: 39.4 tokens/s with the pass, 44.1 without it (int8), the fp16 cache 60.5; the results differ from the pass by the fp16 rounding of the dequantized values only (`crates/forge-gpu/tests/kv_fused.rs`, `kv_fused_model.rs`) | on |
| `FORGE_KV` | `forge-gpu/src/lib.rs` | KV-cache storage (what `--kv` sets): `fp16` (the plain cache), `int8` (`k8v8`), `int4` (`k4v4`), `k8v4`, `k4v8`: keys and values as 8- or 4-bit integers with one fp16 scale per group of `FORGE_KV_GROUP` values of a (position, head) row (kvq.cu). Dense Qwen3-4B Q6_K against the fp16 cache (KL of the top-64 distribution, same top-1 token): int8 0.0002 to 0.0005 (99.4 to 99.7 %), k8v4 0.0004 to 0.0012 (98.5 to 99.4 %), int4 0.012 to 0.021 (93.7 to 96.8 %); the hybrid Qwen3.5-4B and 9B: int4 0.001 to 0.009 (97.4 to 99.0 %) | fp16 |
| `FORGE_KV_GROUP` | `forge-gpu/src/lib.rs` | values per fp16 scale of a compressed KV row: a power of two from 8 to 256 (not above the head width). Smaller groups help the 4-bit keys of a dense Qwen3-4B only partly (int4: KL 0.012 to 0.021 with 32, 0.0070 to 0.0156 with 16, 0.0045 to 0.0080 with 8, against 0.0004 to 0.0012 for `k8v4` with 32) and cost 0.5 bit per value more with 16 than with 32, 1.5 with 8 | 32 |
| `FORGE_ATTN_GQA` | `forge-gpu/src/lib.rs` | `0`: the split decode attention runs one block per query head (as before) instead of one block per key-value head for the 2 to 8 query heads that share it (decode.cu `attn_gqa_impl`: the keys and values of a position are loaded and decoded once for all of them, the cross-lane sums of the heads come out of one butterfly: 6 shuffles instead of 20 for four heads). The partial results are bit-identical to the one-head kernels' (`crates/forge-gpu/tests/attn_gqa.rs`). Qwen3.5-4B decode, tokens/s, one block per head -> per key-value head: 32,768 positions fp16 79.9 -> 90.1, int8 63.7 -> 81.9; 65,536 fp16 60.4 -> 72.5, int8 44.0 -> 62.5 (RTX 2060 SUPER). Kernels exist for head width 64 (4 or 8 heads per group; 3 and 5 to 7 are padded), 128 (2, 4 or 8) and 256 (4: the hybrid Qwen3.5 models); another layout keeps the one-head kernels | on |
| `FORGE_ATTN_GQA_FROM` | `forge-gpu/src/lib.rs` | the GQA attention is used for sessions of at least this many splits of 1,024 positions (the session's context, not the current position). It has one block per key-value head and split, a quarter of the one-head kernels' blocks for four heads per group, and with fewer than about 32 blocks it loses: the Qwen3.5-0.8B (2 key-value heads) at 8,192 positions, 16 blocks: -7 percent (fp16), -3 (int8); at 16,384, 34 blocks: +1 to 2 percent; the dense Qwen3-4B (8 key-value heads) at 2,048 / 4,096 positions: +4 / +5 percent (fp16), -1 / -0.5 (int8) | 32 / key-value heads, rounded up (8 for 4 key-value heads) |
| `FORGE_CTX_MAX` | `forge-engine/src/lib.rs` | caps the context that `--ctx auto` opens (positions); the model's own trained context caps it anyway | none |
| `FORGE_ATTN_ROWS` | `forge-gpu/src/mrow.rs` | `0`: the rows of a multi-row step of a dense model (head width 64 or 128, split attention: chunks of a few prompt tokens and the verification of drafted tokens) take the one-row split attention kernel. By default four rows share the loads of keys and values (`attn_split_rows4_kernel`; its partial results are bit-identical to the one-row kernel's, `crates/forge-gpu/tests/attn_split.rs`). Qwen3-4B Q6_K on a 2060 SUPER after 6,400 positions: 4.6 against 5.6 ms per token of a chunk, and an answer that copies text out of the document with prompt lookup 134 against 109 tokens/s (plain decode: 72) | on |
| `FORGE_NO_FLASH` | `forge-gpu/src/lib.rs` | set = on: do not use the flash attention path in prefill | off |
| `FORGE_NO_EXTRA_CUBIN` | `forge-kernels/src/lib.rs` | `1` selects the portable images instead of a native cubin built with `FORGE_CUDA_ARCHS` (the second check in `lib.rs` treats any set value as on) | off |
| `FORGE_KV_GEMV` | `forge-train/src/decode.rs` | matrix-vector kernel variant of the KV decoder: `1` one weight tile at a time, `3` batches of eight tiles per warp, `0` warp-specialised (bitwise equal results) | 1 |
| `FORGE_KV_ATTN` | `forge-train/src/decode.rs` | single-token attention: `auto`, `fused`, `gqa`, `flash`, `split`; anything else = auto | auto |
| `FORGE_KV_SPLIT_FROM` | `forge-train/src/decode.rs` | with `auto`: switch to the split attention from this many cached positions | 256 |
| `FORGE_KV_GENERIC` | `forge-train/src/decode.rs` | set = on: single tokens go through the row path instead of the fused kernels | off |
| `FORGE_KV_NOGRAPH` | `forge-train/src/decode.rs` | set = on: do not capture the fused step as a CUDA graph | off |
| `FORGE_KV_GV_WARPS` | `forge-train/src/decode.rs` | warps per block of the matrix-vector kernels, clamped to 1..8 | 4 |
| `FORGE_Q8_WARPS` | `forge-train/src/decode/q8.rs` | warps per block of the Q8 matrix-vector kernels of the checkpoint decoder (`forge serve CKPT --q8`, `forge gen-check --q8`), clamped 1..8 | 4 |
| `FORGE_Q8_SPLIT` | `forge-train/src/decode/q8.rs` | warps that share one 16-row tile of a Q8 matrix (the reduction over K is split between them); 0 = by the number of tiles (enough for about 128 tile slices, at most 4) | 0 |
| `FORGE_SPLITK` | `forge-train/src/lib.rs` | split the weight-gradient GEMM reduction into N slices on Turing (1 = never) | chosen per shape |
| `FORGE_NO_FUSE` | `forge-train/src/lib.rs` | value other than empty or `0`: ReLU-squared and its backward as separate kernels | off |
| `FORGE_CE_OLD` | `forge-train/src/lib.rs` | non-`0`: the three-pass cross-entropy kernel | off |
| `FORGE_ATTN_OLD` | `forge-train/src/lib.rs` | non-`0`: the first-generation attention kernels for the dense layers | off |
| `FORGE_NO_FUSE_MIX` | `forge-train/src/lib.rs` | non-`0`: layer-input mixing and attention norm as separate kernels | off |
| `FORGE_F16ACC` | `forge-train/src/lib.rs` | `P` = 2, 4, 8 or 16: forward NT GEMMs accumulate in fp16 and add into fp32 every P k8 steps (Turing; not bit-identical); `0` = fp32 accumulation | 0 |
| `FORGE_GEMM_E2` | `forge-train/src/lib.rs` | `0` = original GEMM epilogue instead of the staged one (bit-identical results) | on |
| `FORGE_GEMM_W144` | `forge-train/src/lib.rs` | `0` = no 128x144 tiles for the data-gradient GEMMs; `force` = always; otherwise used when it saves a wave | auto |
| `FORGE_NO_S80` | `forge-train/src/lib.rs` | set = on: use the Turing GEMMs even on sm_80 and newer | off |
| `FORGE_S80` | `forge-train/src/lib.rs` | set = on: use the Ampere GEMM source with emulated instructions on any device (a correctness test) | off |
| `FORGE_MUON_STREAMS` | `forge-train/src/lib.rs` | CUDA streams for the Muon updates, clamped 1..8 (1 = everything on the main stream) | 4 |
| `FORGE_NAIVE_ATTN` | `forge-train/src/lib.rs` | set = on: do not use the flash backward attention (it is used only when `head_dim` is 64 and the length is a multiple of 64); sliding-window layers then return an error | off |

Flag, not a variable: `forge serve CKPT --q8` on a `forgeckpt1` checkpoint (weights-only files work) quantizes the decoder's matrices to Q8 on load (the engine prints `[engine] Q8 weights: ...`; `FORGE_NO_KV` or a structure the decoder does not reproduce makes `--q8` an error instead of a silent fp16 fallback). The fp32 master and fp16 copy of the weights are freed, so nothing else in that process can run the model (`Trainer::weights_released`); the knobs above act on the Q8 kernels. Without `--q8` nothing changes. `--native` has no meaning for a checkpoint. See `docs/KERNEL_TUNING.md`, section "Q8 weights for the checkpoint decoder".

## 2. Build time

| name | where | meaning | default |
|---|---|---|---|
| `NVCC` | `forge-kernels/build.rs` | path of the `nvcc` binary | looked up in the order below |
| `CUDA_PATH` | `forge-kernels/build.rs` | CUDA toolkit root; uses `$CUDA_PATH/bin/nvcc` (`nvcc.exe` on Windows) | unset |
| `HOME` | `forge-kernels/build.rs` | also tries `$HOME/cuda-13.2.0/bin/nvcc`, then `/usr/local/cuda/bin/nvcc`, then the PATH (per the comment in `build.rs`). Without nvcc the build still succeeds with empty kernel images | unset |
| `NVCC_CCBIN` | `forge-kernels/build.rs` | host compiler for nvcc on Windows; when unset and `cl.exe` is not on the PATH the build script looks for one itself | unset |
| `ProgramFiles(x86)` | `forge-kernels/build.rs` | Windows only, used to find Visual Studio's `cl.exe` | `C:/Program Files (x86)` |
| `FORGE_CUDA_ARCHS` | `forge-kernels/build.rs`, `lib.rs` (`option_env!`), `tools/build_windows.ps1` | comma list of extra native architectures, e.g. `89` for Ada; the values 75 and 80 and anything below 50 are filtered out of the list (the portable sm_75 and sm_80 images are built separately) | unset |
| `FORGE_NO_SM80` | `forge-kernels/build.rs` | `1` = skip the sm_80 cubin | off |
| `OUT_DIR`, `CARGO_MANIFEST_DIR` | `forge-kernels/build.rs` | set by cargo | |

## 3. Rust tests and benchmarks

| name | where | meaning | default |
|---|---|---|---|
| `FORGE_MODELS` | `forge-gguf`, `forge-model`, `forge-tok`, `forge-gpu`, `forge-engine` tests | directory of the reference model files; tests skip when files are absent | `$HOME/forgeinfer/oracle/models` |
| `HOME` | the same tests, `forge-tok/src/fast.rs` | base of the default paths | |
| `FORGE_BENCH_TEXT`, `FORGE_BENCH_TOKENIZER` | `forge-tok/src/fast.rs` (ignored bench test) | text file and tokenizer for `cargo test --release -p forge-tok -- --ignored --nocapture bench` | tokenizer: `$HOME/forgeinfer/data/tokenizer_f1.json` |
| `FORGE_SPEC_SIZE`, `FORGE_SPEC_DIAG` | `forge-engine/tests/speculative.rs`; `tools/spec_battery.sh`, `tools/window4.sh` | model size and diagnostics switch of the speculative-decoding tests; exact values `unverified` | |
| `FORGE_VARIANTS_MODEL` | `forge-gpu/tests/prefill_variants.rs` | model for the prefill variants test | |
| `FORGE_VERIFY_MODEL`, `FORGE_VERIFY_HEAD`, `FORGE_TIMING_PHASE` | `forge-gpu/tests/verify2.rs`; `tools/ncu_t2.sh`, `tools/spec_profile*.sh` | model, MTP head and timing phase of the verification test | |
| `FORGE_F16ACC_DATA` | `forge-train/tests/f16acc.rs` | data for the fp16-accumulation test | |

## 4. Python: JAX trainer and Kaggle kernels

| name | where | meaning | default |
|---|---|---|---|
| `FORGE_NO_PIN` | `tpu/forge_jax.py` | `1`: do not install the tested JAX on Kaggle TPU VMs (same as `--no-pin`) | off |
| `FORGE_JAX_PIN` | `tpu/forge_jax.py` | JAX version to install into `/tmp` on Kaggle TPU VMs | `0.11.2` |
| `FORGE_JAX_PINNED` | `tpu/forge_jax.py` | set by the script itself in the child process so it does not pin again | |
| `JAX_PLATFORMS` | `tpu/forge_jax.py`, `tpu/eval_*.py`, tests | `cpu` skips the pin; the evaluation scripts and tests default to `cpu` (`setdefault`) | |
| `XLA_FLAGS` | `tpu/test_*.py` | set to `--xla_force_host_platform_device_count=N` for multi-device CPU tests | |
| `FORGE_TEST_PRODUCER_FAIL`, `FORGE_TEST_HANG_AT_STEP` | `tpu/forge_jax.py` | test hooks: fail the data producer after N batches; hang at a step to test the watchdog | off |
| `FORGE_KAGGLE_IN`, `FORGE_KAGGLE_OUT` | `cloud/kaggle_tpu_train.py`, `cloud/kaggle_tpu_sft.py` | input and output directories of the Kaggle kernel | `/kaggle/input`, `/kaggle/working` |
| `PYTHONPATH` | the same kernels and `forge_jax.py` | prefixed with the pinned JAX directory for the child process | |
| `CUDA_VISIBLE_DEVICES` | `cloud/kaggle_kernel.py` | set per arm to pin one GPU | |

## 5. Python: tools and Modal

| name | where | meaning | default |
|---|---|---|---|
| `FORGE_KPROF` | `cloud/modal_forge.py`, `tools/kv_t4_job.sh` | the smoke job sets `1` when profiling; passed to the Rust variable above | |
| `FORGE_BIN` | `cloud/modal_forge.py` (set for the ladder job), `tools/launch_f1.sh` and many other `tools/*.sh` | path of the `forge` binary | `launch_f1.sh`: `$HOME/forgeinfer/target-dev3/release/forge` |
| `TOK` | `cloud/modal_forge.py` (set to `/data/eval_pack/tokenizer_f1.json`), `tools/f1_sft_*.sh` | tokenizer path | `$HOME/forgeinfer/data/tokenizer_f1.json` in the sft scripts |
| `FORGE_EVAL_BIN` | `tools/f1_eval.sh`, `tools/f1_track.sh` | `forge` binary for the evaluation ladder (`--forge` wins) | search list in `f1_eval.sh --help` |
| `FORGE_BINARY` | `tools/bench_gemm.sh`, `tools/bench_f16acc.sh`, `tools/kv_serve_e2e.sh` | not an environment variable in the code read: it is the name of a positional argument in the usage text (`unverified` whether any script reads it from the environment) | |
| `FORGE_SRC_BIN` | `tools/switch_to_f1.sh` | source binary to copy | `$HOME/forgeinfer/target-dev3/release/forge` |
| `FORGE_SFT_BIN` | `tools/sft_data/sft_assemble.py`, `human_assemble.py` | `forge` binary used for `tok-encode` and `chat-mask` | `/tmp/forge_sft` |
| `FORGE_TOKENIZER` | `tools/sft_data/sft_lib.py` | tokenizer for the SFT data tools | `/home/user/forgeinfer/data/tokenizer_f1.json` |
| `FORGE_CREATOR` | `tools/sft_data/sft_handwritten.py` | default of `--creator` | `an independent developer` |
| `F1_TOKENIZER` | `tools/test_contamination.py` | tokenizer for the real-tokenizer tests | search list |
| `FORGE` | `tools/tok_pieces.py` | `forge` binary | `./target/release/forge` |
| `SFT_BUILD` | `tools/sft_data/sft_assemble.py`, `sft_audit.py`, `sft_verify.py` | build directory | `/tmp/sft_mix2_build` |
| `HUMAN_BUILD`, `HUMAN_CARDS`, `HUMAN_INCLUDE_OASST2` | `tools/sft_data/human_cfg.py`, `probe_human.py`, `human_assemble.py` | build directory, cards directory, include the OASST2 source when `1` | `<work>/build`, `/mnt/data/forgeinfer-data/sft_work_human/cards`, `0` |
| `BENCH_NGRAM` | `tools/sft_data/sft_bench_index.py` | n-gram length of the benchmark index | 10 |
| `ARITH_MODE` | `tools/sft_data/sft_arith.py` | `facts` selects the facts mode | unset |
| `BENCH_QUICK` | `tools/serve_bench.py` | `1` = quick run | off |
| `CONTAM_TEST_REAL_TOKENIZER` | `tools/test_contamination.py` | set by the test itself from the tokenizer search | |
| `HF_HOME`, `HF_HUB_DISABLE_TELEMETRY`, `TOKENIZERS_PARALLELISM`, `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | `tools/sft_data/*.py`, `tools/contamination_check.py` | library settings applied with `setdefault`; `HF_HOME` goes under the work directory | |
| `PYTHONUTF8`, `PYTHONIOENCODING`, `MSYS_NO_PATHCONV` | `cloud/modal_*.sh`, `tools/tpu_arrival.sh`, `cloud/modal_forge.py` header | set on Windows (Git Bash) before calling Modal | |
| `MICRO`, `MAXH`, `APPROVED`, `STRUCT`, `WARMUP` | `cloud/modal_sft_patch.sh`, `modal_chat_365m.sh`, `modal_sft*.sh` | micro-batch (default 8; 4 in `modal_chat_365m.sh`), hour limit (0.5; 1.6), cost guard in USD (1.5; 3.5), structure flags of the base model, warmup steps | see the RUNBOOK |
| `DRY`, `NO_PRE`, `FULL_UPLOAD`, `ARRIVAL_STRUCT` | `tools/tpu_arrival.sh` | `DRY=1` only prints the commands; `NO_PRE=1` skips the pre-decay checkpoint; `FULL_UPLOAD=1` uploads the whole checkpoint instead of the weights; `ARRIVAL_STRUCT` is the structure printed in the fine-tune hints | off |
| `RUN`, `F1_ARGS`, `DATA`, `LAYERS`, `DIM`, `HEADS`, `KV`, `FFN`, `MICRO`, `TOKENS_PER_STEP`, `STEPS`, `WARMUP`, `DECAY_FRAC`, `MIN_LR_FRAC`, `EVAL_BATCHES`, `SAVE_EVERY`, `LR`, `MUON_LR` | `tools/launch_f1.sh` | run name and training settings of the homelab run | `f1_v1`, empty, `/mnt/data/forgeinfer-data/tokens_f1`, 30, 576, 9, 3, 1536, 2, 65536, 45000, 100, 0.1, 0.05, 8, 100, 1.5e-3, 1.5e-3 |
| `RUN`, `SAVE_EVERY`, `LAUNCH` | `tools/pause_training.sh` | run to pause, checkpoint interval, optional launch command | `f125_muon_v1`, 100, empty |
| `RUN_DIR`, `CKPT`, `BIN` | `tools/start_chat.sh` | training run for the `/status` page, default checkpoint (when no argument is given), `forge` binary of the chat engine | `$HOME/forgeinfer/runs/f1_v1`, `/mnt/data/forgeinfer-data/ckpt_f1_v1/sft_demo_s10000_step300.bin`, `$HOME/forgeinfer/agents/reviewer/bin/forge-fixed` |
| `FORGE_PIPE`, `FORGE_NO_PIPE`, `FORGE_SKIP_DEQUANT` | `tools/window9.sh`, `window10.sh` | named in benchmark scripts as switches of the prefill weight pipeline and the dequantization pass; no crate reads them (grep of `crates/` finds nothing), so they have no effect in this tree | |

The many other `${NAME:-default}` variables in `tools/*.sh` (benchmark windows, quantization experiments, `f1_track.sh`, `auto_ladder.sh`, `branch_at_40500.sh`, `f1_ab.sh` and so on) are local parameters of those scripts; their defaults are at the top of each script and are not repeated here.

The SSH key path and the homelab address (`user@homelab`) used by `tools/chat_add_model.py`, `page_probe.py`, `probe_local.py` and `sft_data/build_shard.sh` are constants in the scripts, not environment variables. The scripts that call Kaggle go through a wrapper (`<dev>/kaggle/kg.sh`) that keeps the API token; no token is read by this repository's own code.
