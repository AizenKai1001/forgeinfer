# Architecture: from text to model

This page follows the data from raw text to a served model. Every statement comes from the code (file names are given); where the code did not settle a point it says `unverified`.

```
corpus text (documents separated by a NUL byte)
   | forge tok-train                      -> tokenizer.json
   | forge tok-encode                     -> .u16 token shards
   | forge chat-mask   (chat data only)   -> bit 15 set on non-target tokens
   v
forge train  (Rust + CUDA)   or   python tpu/forge_jax.py train  (JAX)
   |  both write the same checkpoint format: forgeckpt1
   v
ckpt_latest.bin  -> forge chat / eval-* / serve / export-hf, tools/ckpt_params_only.py (weights only)
```

## 1. Tokenizer

- `forge tok-train <corpus files...> --vocab N --out tokenizer.json [--threads T]` (`crates/forge-cli/src/data_cmds.rs`, `forge-tok/src/train.rs`). Defaults: vocab 32768, 4 threads. Corpus files hold documents separated by a NUL byte (`tools/data_fetch.py` writes them).
- The tokenizer is byte-level BPE with a GPT-2 style pre-tokenizer written as a character scanner and NFC normalisation (`forge-tok`).
- The first 16 ids are fixed special tokens (`SPECIAL_TOKENS` in `data_cmds.rs`): `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>`, `</think>`, `<|pad|>`, then `<|reserved_0|>` ... `<|reserved_5|>` (reserved so that later additions do not change the vocabulary size).
- `forge tok-encode <tokenizer.json> <corpus.txt> --out shard.u16 [--threads T] [--eos ID]`: a pipeline of one reader (32 MB blocks cut at a document end), T encoder threads and one writer. The end-of-text id defaults to the id of `<|endoftext|>` (else 0). A vocabulary above 65536 is refused because the shards are uint16.

## 2. Token shards

- A shard is a flat file of little-endian `u16` tokens, extension `.u16`. `forge train --data DIR` reads every `*.u16` in the directory, sorted by file name (`Data::open` in `train_cmd.rs`).
- Bit 15 marks a token that is not a training target ("not a target"): it is still an input, but the position that predicts it has no loss and no gradient. The flag is honoured only when the vocabulary is at most 32768 (`cfg.vocab <= 32768`); then the id is `v & 0x7fff` and the mark moves to bit 31 (`forge_train::IGNORE_TARGET = 0x8000_0000`) inside the trainer.
- `forge chat-mask <tokenizer.json> <in.u16> --out <out.u16>` sets bit 15 on every token of a ChatML stream that the assistant does not write (system and user turns, role headers, the newline after each `<|im_end|>`, document separators). The assistant text and its closing `<|im_end|>` stay trainable.
- Validation tail: the LAST shard (by sorted name, which is why validation files are called `zzz_val.u16`) keeps `min(--val-tokens, tokens/4)` tokens (default `--val-tokens 20000000`) at its end as the validation region; the trainer does not train on that tail. Validation windows are taken inside the tail at positions `val_start + (k * 7919 * 4093) mod span`.
- Training windows are `seq + 1` tokens long at positions drawn from a hash; a window that would straddle the end of a shard is moved to the start of that shard (the JAX loader does the same, `Data._raw`).
- Cool-down data (`--cool-data DIR`): files `cool_NNN.u16` (names ending in `_val` are never trained on; an optional `heldout/cool_val.u16` or `cool_val.u16` gives a `val_cool` loss). From the first step of the learning-rate decay (or `--cool-start`) a share of the windows, rising linearly from 0 to `--cool-share` (default 0.6) over `--cool-blend-steps`, comes from this pool.

## 3. The model and its structure flags

The base model is a Llama-style decoder (`forge-train/src/lib.rs`, `TrainConfig`): RMSNorm (eps 1e-5), RoPE (theta 10000 by default, `--theta`), grouped-query attention, a tied embedding and output head, and a SwiGLU MLP. Defaults of `forge train`: `--seq 2048`, `--micro 2`, `--heads 4`, `--kv 2`, `--head-dim 64`, `--vocab 32768`. The structure flags (all off = the plain Llama-style decoder):

| flag | meaning |
|---|---|
| `--layers`, `--dim`, `--heads`, `--kv`, `--head-dim`, `--ffn`, `--vocab`, `--seq` | sizes: layers, model width, query heads, key/value heads, head dimension, MLP width, vocabulary, window length (`n_layer, n_embd, n_head, n_kv, head_dim, n_ff, vocab, seq_len` in the checkpoint) |
| `--qk-norm` | RMSNorm on queries and keys (Qwen3 style) |
| `--relu2` | ReLU-squared MLP instead of SwiGLU; there is then no gate matrix and `--ffn` is the hidden width |
| `--untied` | a separate, zero-initialised output head instead of the tied embedding |
| `--lambdas` | per-layer residual scalars: the input of layer l is `a*x + b*x0` where `x0` is the embedding output (the `lam` array, shape `[n_layer, 3]`, initialised to a = 1, others 0) |
| `--skips` | U-net style skips: layer l in the second half adds `c * output of layer n_layer-1-l` |
| `--value-emb 0,1,10,11` | value embeddings: in the listed layers the value vectors become `v*velam[0] + velam[1]*table[token]`, with a vocab x (n_kv*head_dim) table per listed layer |
| `--softcap 30` | logits = `30 * tanh(z / 30)` |
| `--lr-emb-mult`, `--lr-head-mult`, `--lr-table-mult`, `--lr-scalar-mult` | AdamW learning-rate multipliers for the embedding, the separate head, the lookup tables (value embeddings, n-gram memory) and the scalars (`lam`, `velam`); default 1 |
| `--ngram 2,3 --ngram-bits 17 --ngram-dim 128` | hashed n-gram memory (Rust trainer only) |
| `--attn-gate K`, `--peri-norm`, `--emb-norm`, `--window N --window-layers auto\|0,1,..`, `--doc-mask` | gated attention, Peri-LN norms, sliding-window attention, document masking (Rust trainer only) |

Which parameters go to which optimizer is in section 5. `export-hf` writes only the plain structure; for a checkpoint with any structure option it returns an error (`forge_train::export_hf`).

## 4. Checkpoint format `forgeckpt1`

Written by `Trainer::save_checkpoint_with` (`forge-train/src/lib.rs`) and by `save_checkpoint` in `tpu/forge_jax.py`:

1. One JSON header line ending in a newline: `{"magic":"forgeckpt1","n_params":N,"step":S,"loss_scale":L,"cfg":{...}}`. The Rust `cfg` has `n_layer, n_embd, n_head, n_kv, head_dim, n_ff, vocab, seq_len, qk_norm, rope_theta, eps, untied, relu2, softcap, lambdas, skips, ve_layers, ngram, ngram_bits, ngram_dim, attn_gate, peri_norm, emb_norm, window, window_layers`. The JAX header writes the same keys up to `ngram_dim` and may add `plan_steps` (the Rust loader reads the header by key and ignores it).
2. `N` float32 little-endian parameters (the master weights), in the flat order given by `layout()` in `tpu/forge_jax.py` (documented there as "the order of the flat parameter array of the Rust trainer").
3. `N` float32 values of the Adam first moment, then `N` float32 values of the second moment (`master`, `adam_m`, `adam_v` are written in that order). Muon does not use the second-moment array; NorMuon keeps its running averages there.

So a full checkpoint is `12 * N` bytes plus the header, and the weights alone are `4 * N`. Readers that only need weights (`forge train --init-from` = `load_params`, `read_checkpoint_params`, `export_hf`, the chat engine, the evaluation ladders) read just the prefix, header plus parameters; `--resume` needs the moments. `tools/ckpt_params_only.py IN OUT [--check]` cuts a checkpoint down to that prefix. Files are written to a temporary name and renamed when complete (`path.with_extension("tmp")`; the Python cutter uses `.part`).

A training run directory (`--out DIR`) holds `ckpt_latest.bin`, `log.jsonl` (appended; one line per step and per evaluation), and, with `--ema`, `ckpt_ema.bin`. `train.log` is the stdout/stderr that `tools/launch_f1.sh` redirects into the run directory. The JAX trainer with `--time-limit-s` also writes `ckpt_predecay.bin` at the last step before the decay starts (`tpu/forge_jax.py`). The snapshots named `ckpt_<step>.bin` are copies of the rolling `ckpt_latest.bin` made by `tools/f1_snapshots.sh` on the homelab (the trainer keeps only one rolling file).

## 5. The trainer (Rust and CUDA)

`crates/forge-train` and `forge train` (`crates/forge-cli/src/train_cmd.rs`).

- Precision (header of `forge-train/src/lib.rs`): fp32 master weights, gradients and optimizer state; an fp16 working copy of the weights; fp32 residual stream; fp16 GEMM operands and saved activations; fp32 accumulation; dynamic loss scaling. The backward pass is written by hand; the CUDA kernels are in `forge-kernels`.
- Micro-batches: one `forward_backward` call processes `--micro` sequences of `--seq` tokens and accumulates gradients; `step` clips, checks for non-finite values and applies the update. Micro-steps per optimizer step = `ceil(--tokens-per-step / (micro * seq))`, rounded up to a multiple of `--gpus` (`--gpus N` is data parallel over N devices, gradients summed through host memory); the effective tokens per step are printed from `accum * micro * seq`. `--tps-schedule` and `--seq-schedule` change tokens per step and window length over the run.
- Optimizers: with `--optim muon`, Muon (Moonshot-style scaling `0.2 * sqrt(max dim)`, momentum `--muon-mu` 0.95, weight decay `--muon-wd` 0.05, learning rate `--muon-lr`, default = `--lr`) updates the hidden 2-D matrices; AdamW (`--lr`, `--wd` 0.1, `--beta2` 0.95, gradient clip `--clip` 1.0) updates the embedding, the norm gains, the separate head, the lookup tables and the scalars, each group with its multiplier. Without `--optim muon` everything uses AdamW. The Muon orthogonalisation is five Newton-Schulz iterations (`--muon-iters`), or Polar Express coefficients with `--muon-polar`; `--normuon`, `--cautious-wd`, `--ema` are further options (see the header of `train_cmd.rs`).
- Schedule (`lr_at`): linear warmup over `--warmup` steps (default 100), constant at the peak, then decay over the last `--decay-frac` of `--steps` (default 0.2) down to `--min-lr-frac` of the peak (default 0.1). The decay is linear by default; `--decay-shape sqrt|cosine` are the alternatives. This is the warmup-stable-decay (WSD) schedule.
- Run control: `--eval-every` (100), `--eval-batches` (8), `--save-every` (500) steps; `--resume` continues from `ckpt_latest.bin`; `--init-from CKPT` loads the weights only (fine-tuning: fresh optimizer, step 0), and `--resume` wins over it; `--time-limit-s N` stops after writing a checkpoint at the first step past N seconds.
- If the same flag is given twice with different values, the trainer panics instead of silently using the first (`flag()` in `train_cmd.rs`).

## 6. Engine and chat server

- `forge-engine` runs one model on the GPU on a dedicated thread behind a bounded request queue: render the prompt with the chat format of the model, tokenize, keep the longest prefix shared with the KV cache (prefix reuse), prefill, sample, and stream events (content, reasoning, tool calls) through a UTF-8 safe detokenizer and a stop-string matcher.
- A path is treated as a `forgeckpt1` checkpoint when the first bytes of the file are `{"magic":"forgeckpt1"`; the engine then loads it through the trainer's `Decoder` (KV-cache decode kernels in `forge-train/src/decode.rs`) unless `FORGE_NO_KV` is set. A checkpoint trained with sliding-window attention (`window`, `window_layers`) is served by the same decoder: in the window layers the prefill and single-token attention kernels use the training mask (query i sees the keys with i - window < j <= i), over a full-length KV cache (`tests/decode.rs`, `forge-engine/tests/window.rs`: logits bitwise equal to the trainer's forward). Other model formats (GGUF, Hugging Face safetensors, quantized directories) go through `forge-model` and `forge-gpu`.
- `forge serve <model> [--native|--q8] [--tokenizer FILE] [--host H] [--port P] [--ctx N] [--alias NAME] [--chat chatml|smollm2|qwen3|qwen35|qwen35-think] [--api-key KEY] [--queue N] [--temp T] [--top-k K] [--top-p P] [--min-p P] [--repeat-penalty R] [--device D] [--mtp DIR|FILE] [--no-speculate]`. On a GGUF or Hugging Face model `--q8` quantizes the matrices to Q8_0 for the `forge-gpu` engine. On a `forgeckpt1` checkpoint `--q8` switches the `Decoder` to Q8 weights (`Decoder::enable_q8`, `forge-train/src/decode/q8.rs`, kernels `decode_q8f.cu`): the big matrices (q|k|v, output projection, up or gate|up, down, the tied embedding and head) are quantized on the device from the fp32 weights into int8 plus one fp16 scale per 32 weights (1.0625 bytes per weight), the norms, scalars and value-embedding tables stay fp32 / fp16, the single-token step runs fused Q8 matrix-vector kernels (fp16 activations, fp32 accumulation), a prompt piece dequantizes each matrix into a scratch buffer for the tensor-core GEMMs, and the trainer's fp32 master weights and fp16 copy are freed (the 365M model keeps about 0.45 GB of weights on the device instead of 2.2 GB). Without `--q8` the checkpoint path is unchanged (bitwise the trainer's forward). `forge q8-eval CKPT` compares the two on the same tokens (perplexity, logits, greedy text), `forge gen-check CKPT --q8 --no-ref` times the Q8 decoder; numbers and the kernel in `docs/KERNEL_TUNING.md`. `forge-serve` exposes `GET /health`, `/v1/models`, `POST /v1/chat/completions` (streaming SSE), `/v1/completions`, `/apply-template`, `/tokenize`, `/detokenize` (list from the header of `forge-serve/src/lib.rs`).
- `forge chat <checkpoint> --tokenizer FILE (--prompt TEXT | --chat TEXT [--system TEXT] | --interactive)` talks to a checkpoint directly.
- The chat page (`tools/chat_proxy.py`, standard library only) serves one HTML page on port 8191 and forwards `/v1/...` to the engine on 127.0.0.1:8190, adding the API key from a key file so the page needs none. See `docs/RUNBOOK.md`.

## 7. The JAX trainer and how it matches the Rust one

`tpu/forge_jax.py train ...` takes "the flags of `forge train`" (its own docstring) and reproduces the model, the optimizers (Muon plus AdamW with learning-rate groups), the learning-rate schedule, the data windows and the `forgeckpt1` format, so a run can start on one trainer and continue on the other. The JAX model is checked against PyTorch goldens from `tools/gen_train_goldens.py` (`tpu/test_forge_jax.py`).

What is the same: the flat parameter layout (`layout()`), Muon step (`muon_update`: same Newton-Schulz coefficients, five iterations, `0.2*sqrt(max dim)` scale), AdamW groups (`group_of`: emb, head, table, scalar, norm), the warmup/constant/linear-decay schedule (`lr_at`), the validation windows (`val_batch`, same positions as the Rust trainer), the bit-15 target mask (`Data._split`), the checkpoint header and body.

Differences, on purpose (docstring): bf16 operands instead of fp16 (no loss scaling), XLA or splash attention instead of the CUDA flash kernels, `jax.random` instead of the Rust generator for initial weights and data windows. Flags with no JAX counterpart in `parse_args`: `--ngram`, `--attn-gate`, `--peri-norm`, `--emb-norm`, `--window`, `--doc-mask`, `--ema`, `--normuon`, `--muon-polar`, `--decay-shape`, `--tps-schedule`, `--seq-schedule`, `--gpus`. JAX-only flags include `--attn auto|xla|splash`, `--no-auto-steps` (with `--time-limit-s` the number of steps is planned so the decay ends inside the limit: every 100 steps until the decay starts, from the larger of the median step time plus the median evaluation and checkpoint share and the MEAN wall time per step since the compilation, which contains the stalls of the host; `--plan-reserve-s` keeps seconds free, default 600 but at most 10 percent of the limit; a run that the limit still stops inside the decay prints a WARNING with the learning rate it ended at), `--watchdog-s`, `--no-pin`, `--cool-glob`, `--cool-blend-frac`.

On Kaggle TPU VMs the script can re-install a tested JAX into `/tmp` and re-run itself in a child process (`_pin_jax_for_tpu`; off with `--no-pin`, `FORGE_NO_PIN=1` or `JAX_PLATFORMS=cpu`). The cross-checks `tpu/eval_ckpt.py` and `tpu/eval_ppl_jax.py` score a checkpoint of either trainer with the JAX model on the windows the Rust commands use.
