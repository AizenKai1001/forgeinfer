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
| `FORGE_PREFILL_CHUNK` | `forge-gpu/src/lib.rs` | prefill chunk size in tokens; values below 16 are ignored, the value is capped at the compiled maximum and rounded down to a multiple of 16 | 1024 |
| `FORGE_OLD_Q8` | `forge-gpu/src/lib.rs` | set = on: select the older Q8 activation path (`old_q8`); exact effect `unverified` | off |
| `FORGE_DRAFT_ROWS` | `forge-gpu/src/lib.rs` | rows of the vocabulary used by the speculative-decoding draft head (positive integer, capped at the vocabulary size); `tools/draft_sweep.sh` sweeps it | 65536 |
| `FORGE_NO_FUSED` | `forge-gpu/src/lib.rs` | set = on: fused dequantizing GEMMs for K-quants are never used (limit 0) | off |
| `FORGE_FUSED_MAX` | `forge-gpu/src/lib.rs` | K-quant matrices use the fused dequantizing GEMM when the chunk has at most this many tokens | 448 |
| `FORGE_MROW_MAX` | `forge-gpu/src/mrow.rs` | chunks of at most this many tokens of a quantized model (reduction lengths a multiple of 256) run through the multi-row kernels (`decode_m.cu`: the weights are streamed once for groups of 4 / 8 / 12 / 16 rows) instead of the GEMM path; `0` = never. The engine also sends pieces of 2 to 15 tokens to `prefill_batched` when this is on. Measured crossover with the GEMM path on the 2060 SUPER: 62 tokens (Qwen3-4B Q4_K_M), 64+ (Q6_K), 80 (1.7B), 56 (Qwen3.5-4B) | 56 |
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
| `FORGE_FLASH_A80` | `forge-gpu/src/lib.rs` | `0`: the prompt attention uses the sm_75 kernels (`flash.cu`, `flash256.cu`) instead of the Ampere ones (`flash_a80.cu`, compute capability 8.0 and newer) | on |
| `FORGE_F16ACC` (engine) | `forge-gpu/src/lib.rs` | `1` / `0`: the fused GEMMs of the prompt pass accumulate in fp16 (flushed into fp32 after every k16 step) / in fp32; compute capability 8 and newer only | on for 8.6 and 8.9 (GeForce Ampere / Ada), off elsewhere |
| `FORGE_ATTN_SPLIT` | `forge-gpu/src/lib.rs` | `1` / `0`: the decode attention of a dense model runs in splits of 1,024 positions with an exact merge (needed beyond about 11,000 positions, where the one-block kernel no longer fits its scores in shared memory) / as one block per head | on when the context is longer than 11,000 positions |
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
