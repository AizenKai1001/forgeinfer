forge eval-lambada: LAMBADA for the Forge-1 evaluation ladder (a new subcommand of the forge CLI)
================================================================================================

Why: LAMBADA needs two things the existing evaluators cannot give: the greedy-match accuracy of the last word (the arg-max of the logits at every token of the word) and the log-likelihood of the whole word
per example (word perplexity). eval-mc only prints argmax-over-choices accuracies. tools/f1_eval.sh calls `forge eval-lambada` and reports LAMBADA as "unavailable" if the binary does not have it.

Files (nothing under crates/ was touched; this directory is the proposal):
  eval_lambada.rs   the new module (about 130 lines; uses Trainer::logits_at for the arg-max / log-likelihood and Trainer::token_nll for a self-check)
  main_rs.patch     two lines for crates/forge-cli/src/main.rs: `mod eval_lambada;` and the dispatch arm `Some("eval-lambada") => ...`

Apply (repo root, lead's decision):
  cp tools/rust_patches/eval_lambada/eval_lambada.rs crates/forge-cli/src/eval_lambada.rs
  git apply tools/rust_patches/eval_lambada/main_rs.patch
  cargo check -p forge-cli          (builds without nvcc: forge-kernels then has empty GPU images, which is fine for a type check)
  then build the dev binary as usual (FORGE_NO_SM80=1 on the homelab, target dir target-dev3) - not done by the author.

Usage:
  forge eval-lambada CKPT --tokenizer tokenizer_f1.json --data /mnt/data/forgeinfer-data/eval/lambada/lambada_openai.jsonl [--limit N] [--seq 256]
  data line: {"ctx": "passage without its last word", "cont": " word"}   (5153 lines, shuffled; tools/f1_eval_data.py writes it)
  output (parsed by tools/f1_eval.sh):
    selfcheck: largest |nll(logits_at) - nll(token_nll)| over the target tokens of the first 8 examples = 0.00xxx nats (expected below 0.02)
    lambada_openai: 5153 examples  acc 12.34 %  perplexity 123.45   (mean 4.8161 nats per word, 1.55 target tokens per word, 95 s; chance ~0 %)

What was verified and what was not (2026-10-10):
  verified   the module type-checks against the real crates (cargo check -p forge-cli, no nvcc, Rust 1.96) and main_rs.patch applies cleanly (git apply --check) to the main.rs of that day.
  verified   on a CPU: test_stub/ builds the module against the real forge-core and forge-tok with a stub Trainer (an oracle model that peeks at the true next token). 200 real LAMBADA lines, the real
             tokenizer_f1.json: oracle model acc 100 %, word perplexity 4.28 (analytic: exp(1.595 tokens x 0.911 nats) = 4.28); always-wrong model acc 0 %, 17.40 nats per word (analytic 1.595 x 10.91);
             selfcheck 0.00000; also with --seq 64 (left truncation path). This tests the plumbing (JSON, tokenization, context/continuation alignment, loop indices, output format), not the GPU model.
  NOT verified  it has never run on the GPU model (no GPU was allowed). First run: `--limit 50` and read the selfcheck line (a value above 0.02 nats means logits_at and token_nll disagree: do not trust the numbers).
  Re-run the CPU test: cd tools/rust_patches/eval_lambada/test_stub && cargo run --release -- dummy.bin --tokenizer TOKENIZER.json --data lambada_openai.jsonl --limit 200 --seq 256   (FAKE_MODE=wrong for the other model)
  Expected sanity of a trained 150M model: acc 5-30 %, perplexity 40-1000; a random-init model: acc ~0 %, perplexity near the vocabulary size.

Cross-check of ALL the Rust evaluators against an independent implementation (recommended in the first GPU window):
  tools/f1_eval_hf_ref.py scores a Hugging Face model with the same prompts and formulas. A plain-structure checkpoint (for example runs/f125_muon_v1) can be exported with `forge export-hf CKPT --out DIR
  --tokenizer TOK` (works on the CPU), scored by f1_eval_hf_ref.py (CPU, `--tokenizer-file TOK`) and by forge eval-mc / eval-hellaswag / eval-lambada with the same --limit on the GPU: the two must agree to within
  about one example per hundred (fp16 vs fp32 rounding). The commands and the Python-side numbers are in the research note "ForgeInfer - Evaluation Ladder", section "Validation".

Design notes:
  - one forward pass per target token (batch 1): about 1.5 passes per example with the 32k tokenizer; LAMBADA costs about 5-10 minutes on a T4 (estimate, not measured).
  - the context is cut from the left if context + word exceed seq + 1 tokens (the harness does the same at the model's maximum length); with the default seq 256 no LAMBADA example is cut (longest: 227 tokens).
  - the word's tokens are the tail of encode(ctx + cont), as in eval-mc; the log-likelihood is accumulated in f64 from the soft-capped logits of logits_at.
