# Runbook

Commands to build the data, train, fine-tune, evaluate and serve ForgeInfer models on the three places it runs. The usage text is copied from the headers of the scripts (file names are given, read the header for the details). Nothing here was executed while writing this page. Paths such as `/mnt/data/forgeinfer-data`, `~/forgeinfer`, `<dev>/kaggle/kg.sh` and the Modal client path are the author's machine layout; adapt them.

Rules that the scripts themselves enforce or state:

- Modal spends money: every Modal entry point has a hard timeout and a cost guard (`--approved-usd` must cover the worst case) and appends the estimated cost to `cloud/ledger.jsonl`. Nothing is billed until a command is started.
- The structure flags of a fine-tune must equal those of its base checkpoint, otherwise the trainer stops at once with "checkpoint does not match this model" (header of `cloud/modal_chat_365m.sh`).
- The homelab hosts other services: look at `uptime` and `nvidia-smi` before a heavy GPU run (header of `tools/hl.sh`).

## 0. Get the code onto the homelab and build

```
tools/push.sh "message"                        # commit the work tree on the editor machine and push it to the homelab bare repo; a hook checks it out into ~/forgeinfer/src
tools/hl.sh "cargo build --release"            # run a command in the work tree on the homelab (nice 15); -n = no nice, no cd
```

`cargo test` does NOT rebuild `target/release/forge`: after a merge that changes the trainer or the engine, run the build above before measuring anything. Check that the binary has the new code (for a new environment variable or flag: `tools/hl.sh -n 'strings ~/forgeinfer/src/target/release/forge | grep -c FORGE_NEW_FLAG'` must print 1 or more, and the file time must be after the push). Found 2026-10-11: an A/B of the gpu-tune merge ran the 5:11 PM binary against itself and showed no difference.

## 0a. TPU compile lab (no TPU needed)

```
tools/aot.sh LAB --only base,muon-shard          # compile the real training step for a v5e-8 on the homelab CPU (nice 15), ~45 s per variant; results in ~/forgeinfer/aot_lab/LAB/out
python tpu/hlo_cost.py FILE.hlo.txt --top 30     # where the compiler thinks the step time goes (fetch the file with tools/hl.sh -n 'cat ~/forgeinfer/aot_lab/LAB/out/base.hlo.txt' > FILE.hlo.txt)
```

What it says: a variant compiles for the TPU or not (a flag libtpu rejects, a sharding or Pallas kernel the TPU compiler refuses), its peak memory per chip against the 15.75 GB of a v5e chip, and the TPU compiler's own estimate of the step (the sum of the `estimated_cycles` of the entry computation at 1.5 GHz). The estimate of the base 365M step is 0.365 s against 0.379 s measured on the Kaggle v5e-8; for an XLA-attention step it was 0.77 s against about 0.60 s expected from the measured attention kernels, so it ranks changes of the XLA-compiled part (muon-shard, layouts, fusions, remat policies) but is no promise in seconds, and it does not look inside Pallas kernels (window attention, splash block sizes show no change). The measured step stays the last word (`bench_step.py` on a TPU). Results of 2026-10-11 for the 14 variants of `bench_step.py`: research note "ForgeInfer - TPU Compile Lab".

## 0b. Data: text to token shards

```
~/forgeinfer/venv/bin/python tools/data_fetch.py <dataset> <shard> [--out DIR] [--raw DIR] [--keep]     # corpus text, documents separated by NUL (datasets: fwedu, cosmo, fmath, owmath, wiki, gut, ...)
forge tok-train <corpus files...> --vocab 32768 --out tokenizer.json [--threads T]
forge tok-encode tokenizer.json corpus.txt --out shard.u16 [--threads T] [--eos ID]
tools/f1_data.sh            tools/f1_data.sh val        # the Forge-1 mixture (homelab); `val` builds tokens_f1/zzz_val.u16
tools/prep_shards.sh 1 2 3                             # FineWeb-Edu shards, one at a time
```

The last file of a shard directory (by name) holds the validation tail, so name it `zzz_val.u16`. See `docs/ARCHITECTURE.md` section 2.

## 0c. Run the Python tests (tpu/, cloud/, tools/tests) on the homelab

```
tools/agent_sync.sh NAME                                                              # copy the work tree (a helper's worktree too) to ~/forgeinfer/agents/NAME/src
tools/agent_hl.sh NAME "cd tpu && ~/forgeinfer/tpuc/bin/python run_all_tests.py --jobs 4"
```

`tpu/run_all_tests.py` runs every test of the Python side in parallel worker processes (79 jobs: one per check of `tpu/test_forge_jax.py`, one per check of the other `tpu/test_*.py` files that list their checks in the `__main__` block, `cloud/test_autotune.py`, the unittest files of `tools/tests`), keeps the output of each job in its own log file (the directory is printed) and ends with one summary; on the homelab with `--jobs 4` the whole set takes 7.5 minutes instead of the 19 minutes of running the files one after the other (2026-10-12, all passed both ways); exit code 0 only when every job passed, and a process that dies at exit (SIGABRT) counts as a failure even when its checks printed ok. `--only TEXT` and `--skip TEXT` select jobs by a text in the name, `--list` shows them, the default `--jobs` is half the cores, at most 4. `tpu/run_checks.py` is the single-process version for `test_forge_jax.py` alone (every check is reported, none stops the others; `--only check_fuse_qkv,...`). All of it runs on the CPU (`JAX_PLATFORMS=cpu`, fake devices) and needs no TPU or GPU, but it is CPU-heavy for several minutes: run it on the homelab (nice 15), not on the editor machine while someone is gaming.

## 0d. Short prompts: the multi-row path (a chat turn, the tail of a hybrid model's prompt)

A chunk of at most `FORGE_MROW_MAX` (56) tokens of a quantized model (Q8_0 / Q4_0 / Q5_0 / Q4_K / Q5_K / Q6_K, reduction lengths a multiple of 256) streams every weight matrix once for groups of up to 16 rows (`crates/forge-kernels/kernels/decode_m.cu`, `crates/forge-gpu/src/mrow.rs`) instead of going through the dequantizing GEMM, whose decode work is the same for 8 or 64 rows (a fixed ~70 ms on a 4B model). Longer chunks, fp16 models and calibration passes take the old path unchanged.

```
tools/agent_hl.sh NAME "./target/release/forge prefill-bench MODEL.gguf --native --tokens 5 --reps 8"                  # time of a 5-token prompt (best run), multi-row path
tools/agent_hl.sh NAME "FORGE_MROW_MAX=0 ./target/release/forge prefill-bench MODEL.gguf --native --tokens 5"           # the GEMM path (A/B)
tools/agent_hl.sh NAME "FORGE_KPROF=2 ./target/release/forge prefill-bench MODEL.gguf --native --tokens 16 --reps 4"    # per-kernel, per-shape times (the autotuner's launches are in the table too)
tools/agent_hl.sh NAME "FORGE_MROW_EVAL=1 ./target/release/forge ppl MODEL.gguf --native --tokenizer T --text eval.txt --ref REF.bin"   # perplexity / KL / top-1 of the path against a reference dump of the GEMM path (--dump-ref)
tools/agent_hl.sh NAME "FORGE_EVAL_DECODE=1 ./target/release/forge ppl ..."                                               # the same through the single-token decode graphs (the third implementation)
tools/agent_hl.sh NAME "cargo test --release -p forge-gpu --test mrow -- --nocapture"                                    # kernel tests (every format, T = 4 / 8 / 12 / 16, launch shapes, K slices, fused launches), prompts of Qwen3 / Qwen3.5 against the decode and GEMM paths, verify_rows
tools/agent_hl.sh NAME "tools/mrow_kbench.sh && ./mrow_kbench q4_k 9728 2560"                                            # kernel micro-benchmark: per-matrix time against the one-token kernel (mvt1)
tools/agent_hl.sh NAME "python3 tools/mrow_chat_probe.py LABEL ./target/release/forge MODEL.gguf tokenizer.json qwen35 [--mtp mtp.safetensors]"   # chat path: prompt time of fresh prompts and first-token time of the turns of a conversation, through a server of this build on port 18177 (FORGE_MROW_MAX=0 for the old path)
```

Numerics: like the decode graphs the path quantizes every activation to int8 (blocks of 32) and multiplies with dp4a; the GEMM path works on fp16 operands. Against the GEMM path on 24k tokens of text (`forge ppl`, `--dump-ref` / `--ref`) the multi-row path gives a perplexity difference of -0.0004 .. +0.0006 nats, KL(top 64) 0.0004 .. 0.0015 and top-1 agreement of 98.1 .. 99.1 % (Qwen3-4B Q4_K_M, Qwen3-4B-Instruct Q6_K, Qwen3-1.7B, Qwen3.5-4B). That is the noise floor of int8 activations and not a defect of the kernels: the single-token decode graphs (`FORGE_EVAL_DECODE=1`) are 98.5 % against the GEMM path as well, the multi-row path against the decode graphs 98.6 %, and two runs of the multi-row path that differ only in the order of the K slices (`FORGE_MROW_TUNE_NJ=1`) 98.6 % (every flipped rounding of an int8 value moves the next layer's roundings, so any change of the float summation order shows up as such a difference). The multi-row path itself is deterministic: the K slices are a function of the shape alone, and a row gives the same bits whatever its group, the launch shape or the run (tests/mrow.rs). Compare implementations by perplexity and KL, not by top-1 agreement above 98.6 %.

The pieces of the engine: `GpuSession::prefill_batched` sends a chunk down this path when `mrow_ok(t)`; the engine sends pieces of 2 to 15 tokens to `prefill_batched` instead of `prefill` (`prefill_batched_min`) and warms the kernels up at load. `verify_rows(tokens)` runs 2 to 16 tokens of a speculative draft in one sweep and `accept_rows(m)` goes back to the position after row m - 1 (hybrid models: the recurrent states after every row are saved on the side).

## (a) Train on the homelab (RTX 2060 SUPER)

Start or resume a long pre-training run, detached (`tools/launch_f1.sh`):

```
RUN=f1_v1 F1_ARGS="--relu2 --ffn 2304 --lambdas --lr-scalar-mult 10" tools/launch_f1.sh
```

Re-running resumes from `ckpt_latest.bin`. Checkpoints go every `SAVE_EVERY` steps (default 100) to `~/forgeinfer/runs/$RUN` (`ckpt_latest.bin`, `log.jsonl`, `train.log`). Flags that must survive every restart go on one line in `$OUT/extra_args.txt`. The defaults of the script (30 layers, dim 576, 9 heads, 3 kv, ffn 1536, seq 2048, 65536 tokens per step, 45000 steps, Muon at 1.5e-3, decay over the last 10 percent to 5 percent) are listed in `docs/ENVIRONMENT.md`.

Related scripts: `tools/pause_for_chat.sh CKPT` stops the training right after a checkpoint and starts the chat engine (the 8 GB card cannot hold both plus another program); `tools/pause_training.sh`, `tools/gpu_window.sh` and `tools/window*.sh` pause the trainer for GPU measurements. `tools/f1_snapshots.sh` copies the rolling checkpoint every N steps (default 2500).

A continuation fine-tune of a finished chat checkpoint on the idle GPU (`tools/homelab_patch.sh`; it waits until no other `forge train` of this user is running and uses the structure of the first TPU base):

```
nohup bash ~/forgeinfer/src/tools/homelab_patch.sh patch9 /mnt/data/forgeinfer-data/models/tpu1_patch8_skills.bin /mnt/data/forgeinfer-data/tokens_patch9 1200 1e-4 &
#   usage: homelab_patch.sh NAME INIT_CKPT DATA_DIR STEPS LR [WARMUP=40] [SEED=5]
```

## (b) Train on Modal (A100)

Client and shell setup on Windows Git Bash (from `cloud/modal_forge.py`): `export MSYS_NO_PATHCONV=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8`. Data and results live in two Modal volumes, `forge-data` (tokenizer and token shards) and `forge-runs` (logs, checkpoints).

```
modal run cloud/modal_forge.py::smoke --gpu T4 [--micro 4] [--steps 30]      # image build, kernels, speed on random tokens (cents)
modal run cloud/modal_forge.py::sweep --gpu A100-40GB --arms cloud/arms_round1.json     # proxy A/B arms in parallel   (header form; the code takes an arms name such as round2a: unverified which is current)
modal run --detach cloud/modal_forge.py::train --name NAME --gpu A100-40GB --args="<forge train flags>" --data "a.u16,b.u16,..." --max-hours H --approved-usd USD
modal run cloud/modal_forge.py::ladder    --name NAME --label NAME --approved-usd 1      # evaluation ladder on a T4 (about 10 minutes by its docstring)
modal run cloud/modal_forge.py::checklist --name NAME --label NAME --approved-usd 0.6
modal run cloud/modal_forge.py::probe     --name NAME --prompts-file FILE.json --label L --approved-usd 0.6
modal volume put forge-runs LOCAL.bin /BASE_NAME/ckpt_latest.bin       # a base checkpoint must be in a forge-runs folder first
modal volume get forge-runs sft_BASE_MIX_STEPS_LR/ckpt_latest.bin LOCAL.bin
```

Wrappers (read each header for the arguments):

```
bash cloud/modal_sft.sh BASE_NAME MIX [STEPS] [LR]          # MIX = human | mix2; defaults 2250 / 4000 steps, LR 3e-4; about 1 hour and 2.5 USD for mix2
bash cloud/modal_grid.sh BASE_NAME [MIX] [STEPS] [LR ...]   # the same base at several learning rates in parallel, one A100 container each
bash cloud/modal_sft_safety.sh RUN_NAME SAFETY_SHARD [STEPS] [LR] [BASE]
bash cloud/modal_sft_patch.sh RUN_NAME BASE STEPS LR "DATA,LIST,..." [WARMUP]
bash cloud/modal_chat_365m.sh [RUN_NAME] [STEPS] [LR] [BASE]    # defaults: sft_tpu2_chat_2300_1.5e-4  2300  1.5e-4  tpu2
```

`modal_sft_patch.sh`: a short continuation fine-tune of an already chat-tuned checkpoint on an A100-40GB, about 0.76 s per step of 65,536 tokens (700 steps are about 9 minutes and about 0.45 USD by the header). `BASE` is a `forge-runs` folder holding `ckpt_latest.bin`; `DATA` is a comma-separated list of shard paths in `forge-data`, the LAST one must be the validation shard (`zzz_val`). Environment: `MICRO` (default 8; the 365M model needs 4 or less), `MAXH` (hour limit, 0.5), `APPROVED` (cost guard in USD, 1.5; the run is refused if `MAXH` at the GPU price is above it), `STRUCT` (the structure flags of `BASE`, default the first TPU base: 150M, value embeddings in layers 0,1,28,29). Example from the header:

```
bash cloud/modal_sft_patch.sh sft_patch1 sft_tpu1_human_2250_2e-4 700 1.5e-4 "tokens_sft_human_replay/a_replay.u16,tokens_sft_human_replay/sft_000.u16,tokens_sft_human_replay/sft_001.u16,tokens_skills/sft_skills.u16,tokens_safety2/sft_safety2.u16,tokens_sft_human_replay/zzz_val.u16"
```

`modal_chat_365m.sh`: the single chat-tuning run of the 365M base (24 layers, dim 1024, 16 heads, 4 kv, ffn 4096, value embeddings in layers 0,1,2,3,20,21,22,23); measured 1.35 s per step on 2026-10-11 (the header's first estimate was 1.8 s), about 52 minutes and 2.1 USD; defaults `MICRO=4`, `MAXH=1.6`, `APPROVED=3.9` (the guard prices the worst case at 2.42 USD/h for the A100-40GB, 3.87 USD for 1.6 h, so 3.5 refuses the run). `BASE` is the folder where `tools/tpu_arrival.sh` put the base as `/tpu2`; a weights-only file is enough for `--init-from`. Start it only after the base was judged usable.

## (c) Train on a Kaggle TPU (free)

The JAX trainer `tpu/forge_jax.py` runs inside a Kaggle kernel started from the Kaggle web page with the accelerator set to TPU.

1. `cloud/kaggle_tpu_train.py` is the kernel script. Inputs (datasets of the account `YOUR_KAGGLE_USER`): `forge-jax` (`forge_jax.py`), `forge-tokens` (token shards), `forge-tpujob` (`settings.json`), optionally `forge-ckpt-tpu` (`ckpt_latest.bin` and `log.jsonl` of the previous session of a chain). `settings.json` example from the header: `{"args": "--layers 30 --dim 576 ... --steps 45000 --time-limit-s 28000", "data": "forge-tokens", "micro": [4, 2, 1], "require_accelerator": true}`. The kernel tries the micro-batch sizes in order and uses the first that fits (every entry must divide the batch: with 65536 tokens per step, 8 chips and 2048 tokens, micro 4 / 2 / 1 give 1 / 2 / 4 micro-steps of the same 32 sequences; a micro of 8 would silently train with 131072 tokens per step, see docs/TPU_PREMORTEM.md F4). Optional settings: `session_s` (default 32400), `end_margin_s` (1500), `silence_s` (1500), `print_every` (10), `jax_pin` ("0.11.2", "" = preinstalled JAX), `autotune` (off when absent): `{"candidates": [{"name": "muon-shard", "flags": "--muon-shard", "env": {}}], "budget_s": 1200, "per_variant_s": 420, "steps": 6, "min_gain_pct": 2.0}`: before the training the session benchmarks the candidates against `base` with the run's own model arguments (`tpu/bench_step.py`, one worker process per candidate; `bench_step.py` must sit next to a byte-identical copy of the trainer, in the folder of `forge_jax.py` or in the dataset `forge-jax-code`), and the fastest candidate that computes the same function (same first-step loss, same tokens per step) and is at least `min_gain_pct` percent faster adds its flags and environment to the trainer's; a failure or a missing file leaves the settings as they are. It replaces a separate benchmark session (a second wait in Kaggle's queue); costs about 10 to 20 minutes of the 9 hours; candidates must never change the model (window, head size, batch). Outputs in `/kaggle/working`: `run/` (`log.jsonl`, `ckpt_latest.bin`), `autotune/` (`bench.json`, `decision.json`), `train.log`, `env.txt`, `exit.txt`.
2. Ready-made settings are in `cloud/tpu_settings/` (`v1_recipe_150M.json`, `v2_ve8_150M.json`, `m365_ve8.json`, `sft365_grid.json`; its `README.md` explains them). Copy one to `<dev>\kaggle\stage\forge-tpujob\settings.json` and publish the dataset with `cd <dev>/kaggle/stage/forge-tpujob && bash <dev>/kaggle/kg.sh datasets version -p . -m "text"` (a `-p X:/...` path fails on Kaggle's upload cache).
3. A training of more than one session: `python cloud/kaggle_chain.py --name forge-train --target-step 45000 [--max-hops 8] [--poll-s 300]` pushes the kernel, waits, makes the checkpoint a dataset and pushes again (each hop at most 12 hours). Staging of datasets and kernel folders: `python cloud/kaggle_stage.py bin|job|kernel ...` (header).
4. Chat fine-tunes on the TPU: `cloud/kaggle_tpu_sft.py` runs several fine-tunes of a finished base one after the other in one session (a learning-rate grid costs one session start). Settings (`sft365_grid.json`): `{"data": "forge-sft365", "micro": [4, 2, 1], "runs": [{"name": "lr15", "init": "forge-base365/ckpt_latest.bin", "max_s": 5400, "args": "... --steps 2300 --lr 1.5e-4 ... --no-auto-steps --save-every 100000"}, ...]}`. The "data" dataset holds the `.u16` chat shards, the LAST file by name holds the validation tail. After each successful run the pre-decay checkpoint is removed and `ckpt_latest.bin` is cut to weights only; `summary.json` lists the runs and `exit.txt` is 0 when all finished. Runs that no longer fit in the session are skipped.
5. When the session has ended, run `tools/tpu_arrival.sh [KERNEL] [TAG]` from the repo root on the PC (defaults `YOUR_KAGGLE_USER/forge-tpu` and `tpu1`; `DRY=1` only prints the commands). It (1) downloads the kernel output first, before any push to that kernel, (2) prints the log summary and a JAX check of the checkpoint on the PC's CPU, (3) cuts the checkpoint to weights (`tools/ckpt_params_only.py ... --check`), uploads it to the Modal volume `forge-runs` as `/TAG/ckpt_latest.bin` and runs the evaluation ladders on a Modal T4, (3b) does the same for the pre-decay checkpoint (`NO_PRE=1` skips it), (4) prints the fine-tune commands (`ARRIVAL_STRUCT` sets the structure it prints). `FULL_UPLOAD=1` uploads the whole file (needed only to resume the pre-training). It needs the Kaggle CLI wrapper `<dev>/kaggle/kg.sh`, the Modal client and a free credit or a spend limit on Modal.

The JAX trainer itself can also be started by hand: `python tpu/forge_jax.py train --data DIR --out DIR --layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --seq 2048 --tokens-per-step 65536 --steps 45000 --relu2 --lambdas ...` (flags of `forge train`; `--time-limit-s` makes it plan its steps so the decay ends inside the limit).

## (d) Fine-tune for chat and evaluate

Data (`tools/sft_data/`; most scripts are builders of one pool, read the header of each):

```
python tools/sft_data/safety_shard.py --in A.jsonl [--in B.jsonl ...] --out corpus.txt [--repeat N] [--seed S]   # ChatML documents ending in NUL
bash tools/sft_data/build_shard.sh CORPUS.txt OUT.u16            # runs tok-encode then chat-mask on the homelab CPU, copies the shard back
python tools/sft_data/sft_assemble.py plan | encode | publish    # the SFT mix v2 (plan, encode with tok-encode + chat-mask, publish)
```

The fine-tune is `forge train --init-from BASE.bin` on shards whose non-target tokens carry bit 15 (see section 2 of `docs/ARCHITECTURE.md`), run through (a) the homelab patch script, (b) the Modal wrappers or (c) `kaggle_tpu_sft.py`.

Evaluation:

```
bash tools/f1_eval.sh CKPT [--quick] [--dry-run] [--out DIR] [--data DIR] [--forge BIN] [--tokenizer FILE] [--tasks LIST] [--limit N] [--batch N] [--tight] [--name LABEL]   # the evaluation ladder; --help lists everything
forge eval-ppl      CKPT --data FILE.u16 [FILE.u16 ...] [--windows 64] [--seq 512] [--batch 4]
forge eval-hellaswag CKPT --tokenizer FILE --data hellaswag_val.jsonl [--limit N]
forge eval-mc       CKPT --tokenizer FILE --data FILE.jsonl [--limit N] [--seq 256] [--batch 8]
forge eval-lambada  CKPT --tokenizer FILE --data lambada_openai.jsonl [--limit N] [--seq 256]
forge chat          CKPT --tokenizer FILE --chat "What is 2 + 2?" [--system "..."] [--tokens 80] [--temp 0.8]
bash tools/sft_data/sft_checklist.sh CKPT [OUT_FILE]            # the 20-prompt checklist; needs a GPU; FORGE_BIN and TOK set
python tools/sft_data/score_checklist.py checklist_a.txt checklist_b.txt [--json OUT.json] [--verbose]    # automatic score out of 40 ("usable" from 28, "good for its size" from 34)
python tools/sft_data/score_probe.py probe_a.txt probe_b.txt [--verbose]     # harmful refusal rate and harmless help rate
python tools/sft_data/score_skills.py PROBE.json OUT1.txt [OUT2.txt ...] [--verbose]
```

All of the `forge eval-*` and `chat` commands need a GPU. The same ladder on Modal is `modal run cloud/modal_forge.py::ladder ...` above.

Probes of a model through the chat page or locally (run on the PC; they need the SSH key of the homelab):

```
python tools/probe_local.py CKPT_ON_HOMELAB PROMPTS.json OUT.txt [ANSWER_LIMIT_TOKENS=160]
    # greedy probe through `forge chat` on the homelab GPU instead of a Modal T4; needs the GPU to be free enough for a 1.3 GB inference run next to the chat engine, about 3 s per prompt
python tools/page_probe.py PROMPTS.json OUT.txt [--model "Title of the model"] [--n 1] [--restore]
    # runs the prompts through the chat page's own settings (temperature 0.7, top-k 40, top-p 0.95, repeat penalty 1.1); output in the probe format for score_probe.py
python tools/page_eval.py "Model title on the page" TAG [--default "Title to switch back to"]
    # the whole page-side evaluation of one chat model (everyday requests, harm A/B, arithmetic, skills, 20-prompt checklist), about 12 minutes on the homelab GPU; files scratch/pe_<TAG>_*.txt
```

Prompt files are JSON lists (`{"kind", "prompt", optional "system"}`, optional `"turns"` for multi-turn); the probe files named in the headers (`scratch/safety_probe.json`, `safety_probe_b.json`, `arith_probe.json`, `skills_probe.json`, `everyday60.json`, `page_checklist.py`) are under `scratch/` and are not part of this tree (`unverified` whether they are committed anywhere).

## (e) Serve the chat page

On the homelab (`tools/start_chat.sh`):

```
bash ~/forgeinfer/start_chat.sh [CKPT]       # starts forge serve on 127.0.0.1:8190 with an API key and tools/chat_proxy.py on port 8191; stop: bash ~/forgeinfer/start_chat.sh stop
```

The key lives in `~/forgeinfer/chat.key` (created once, mode 600; only the proxy reads it). Page: `http://<homelab-ip>:8191/`, training status: `/status`. The engine needs about 1.1 GB of GPU memory next to a training run. It starts `forge serve CKPT --tokenizer TOK --host 127.0.0.1 --port 8190 --ctx 1024 --alias forge-1 --chat chatml --api-key KEY --temp 0.7 --top-k 40 --top-p 0.95 --repeat-penalty 1.1`.

Q8 weights (since 2026-10-12 the live page runs on them): `"q8": true` at the top of `~/forgeinfer/chat_models.json` makes `start_chat.sh` and the proxy's model switch start the engine with `--q8` (int8 weights with an fp16 scale per 32 values: greedy decode +58 percent, 365M model 484 to 766 tokens/s, GPU memory 2.6 to 1.0 GB, perplexity +0.0002 nats; an entry of the models file may say `"q8": false` for one model). When the Q8 engine does not come up (a width that is no multiple of 64, an n-gram memory, an unknown file) both fall back to the same engine on fp16 weights. The switch on 2026-10-12 left backups next to the live files (`start_chat.sh.before_q8`, `chat_proxy.py.before_q8`, `chat_models.json.before_q8`, `bin/forge-chat.prev`) and `~/forgeinfer/rollback_q8.sh [CKPT]`, which puts them back and restarts the page. To go back by hand: remove `"q8": true` from the models file and run `start_chat.sh CKPT`. After a change of the engine code, `cargo build --release` in `~/forgeinfer/src` and copy `target/release/forge` over `~/forgeinfer/bin/forge-chat` (the running engine uses that copy, never the build output).

Downloaded models on the same page (since 2026-10-12, 4 AM): an entry of `chat_models.json` may carry an `"engine"` block and a `"page"` block next to `"ckpt"` and `"note"`. A `.gguf` file is served with `--native` (its quantized blocks as they are, never `--q8`). `"engine"`: `tokenizer` (a Hugging Face `tokenizer.json`, needed for GGUF), `chat` (`chatml`, `smollm2`, `qwen3`, `qwen35`, `qwen35-think`), `ctx`, `alias`, sampling (`temp`, `top_k`, `top_p`, `min_p`, `repeat_penalty`), `mtp` (the `mtp.safetensors` draft head of a Qwen3.5 model: 4B 117 -> 161 and 9B 69 -> 100 tokens/s). `"page"`: `title`, `intro`, `temperature`, `max_tokens`, `max_tokens_limit`, `system`, `repetition_penalty`, `thinking` (shows the think switch; `thinking_default`), `examples`; without a `page` block the page is the Forge-1 page as before. The list in the live file: Qwen3.5-9B and 4B (4-bit, with their draft heads) and Qwen3-4B-Instruct-2507 (6-bit); add more the same way (the proxy re-reads the file). A model that does not come up (no room next to another process on the 8 GB card) is retried without its draft head, and when that fails too the model that was running comes back. The page prints under every answer: tokens and tokens/s of the decode, first token, prompt tokens (reused ones), how often the draft head was right. The switch of 2026-10-12 left backups (`chat_proxy.py.before_others`, `chat_models.json.before_others`, `bin/forge-chat.before_others`) and `~/forgeinfer/rollback_others.sh [CKPT]`. The engine now takes the working memory of the first request at start (a model that does not fit fails at start, with `GPU warm-up: ... OUT_OF_MEMORY`) and resets its session after a failed request (before, every later request of a hybrid model panicked with "recurrent state cannot be rolled back"). Tests of the page: `python -m unittest tools.tests.test_chat_proxy` (the page script runs against a fake page with node: `tools/tests/chat_page_harness.js`). Answers are rendered as markdown (bold, italic, inline code, fenced code blocks, bullet and numbered lists, headings, quotes, rules, https links), built node by node so that no model output is ever parsed as HTML; the renderer is throttled to every 40 ms while an answer streams. Live since 2026-10-12 7:18 AM; backup `~/forgeinfer/chat_proxy.py.before_md`.

The proxy by itself (`tools/chat_proxy.py`, standard library only):

```
python3 chat_proxy.py [--port 8191] [--engine http://127.0.0.1:8190] [--key-file PATH] [--run-dir DIR] [--models-file FILE] [--engine-bin BIN] [--tokenizer FILE] [--current-ckpt CKPT]
```

`GET /` is the page, `/v1/...` is forwarded to the engine with the API key added, `--run-dir` enables `/status` and `/status.json`, and `--models-file` (JSON `{"default": TITLE, "models": {TITLE: {"ckpt": PATH, "note": TEXT}}}`, example `tools/chat_models.example.json`) adds a model selector: `/api/models` lists the checkpoints and `POST /api/select {"name": ...}` restarts the engine on the chosen checkpoint (a few seconds).

Add a model to the page (run on the PC):

```
python tools/chat_add_model.py --ckpt LOCAL.bin --file NAME.bin --title TEXT --note TEXT [--default] [--restart]
python tools/chat_add_model.py --remote-ckpt PATH_ON_HOMELAB --file NAME.bin --title TEXT --note TEXT
```

It copies the checkpoint to the NVMe of the homelab (`~/forgeinfer/models_fast/NAME.bin`: the engine is ready in about 0.6 s and a model switch on the page takes about 2.5 s) when that has 30 GB free, else to the hard drive (`/mnt/data/forgeinfer-data/models/NAME.bin`, 127 MB/s: a 365M model needs about 12 s to load; `--hdd` forces it) and adds or updates the entry in `~/forgeinfer/chat_models.json`; nothing is replaced. The proxy re-reads that file whenever it changes, so the new model is in the page's list within seconds and no restart is needed; `--default` makes it the default model (what the engine loads at start) and `--restart` restarts engine and proxy (about ten seconds without the page) only to make a new default the running model. Put weights-only files there (`tools/ckpt_params_only.py`): the engine reads nothing else.

OpenAI-compatible use without the page: `forge serve <model> [--native|--q8] [--tokenizer FILE] [--host H] [--port P] [--ctx N] [--alias NAME] [--chat chatml|...] [--api-key KEY]` (see `docs/ARCHITECTURE.md` section 6).

## (f) Cut a checkpoint to weights only

```
python tools/ckpt_params_only.py IN.bin OUT.bin [--check]
```

Keeps the header line and the parameters (4 bytes per parameter instead of 12). Everything that only needs the weights reads exactly that prefix (`forge train --init-from`, `export_hf`, the chat engine, the ladders and the probes); it is NOT for `--resume`, which needs the moments. `--check` reads the output back and verifies header, length and that the parameters equal the input's prefix byte for byte. By its header the 365M model is 4.4 GB with moments and 1.46 GB without. `tools/tpu_arrival.sh` calls it before uploading.
