# Executable specifications and golden test data

ForgeInfer is written against **executable specifications**: small, transparent Python/numpy programs that implement each piece in the simplest possible way and are checked against recorded
outputs of reference programs ("oracles": Hugging Face transformers, PyTorch CPU, the `tokenizers` and `gguf` libraries, llama.cpp). The Rust and CUDA code is a port of these files and is
tested against the same goldens. The oracles are never part of the product.

| Spec (tools/) | What it pins down | Verified against | Result (2026-10-09) |
|---|---|---|---|
| `ref_forward_numpy.py` | forward pass of the Llama and Qwen3 blocks (RMSNorm, GQA, rotate-half RoPE, QK-norm, SwiGLU, tied head), own safetensors reader | HF logits (6 prompts x 2 models), per-layer hidden states, 8 greedy tokens | logits within 6e-5, greedy tokens identical |
| `ref_tokenizer.py` | byte-level BPE with the GPT-2+Digits (SmolLM2) and Qwen2+NFC (Qwen3) pre-tokenizers, added tokens | 475 cases per tokenizer from `tokenizers` | 475/475 exact |
| `ref_gguf.py` | GGUF v3 container; dequantization of F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q4_K, Q6_K; llama.cpp's Q/K row permutation | 64 real blocks per type; full forward from the GGUF files | bit-exact; F16 file reproduces HF |
| `ref_backward_numpy.py` | hand-derived backward pass of the whole block incl. GQA, RoPE, QK-norm, SwiGLU, tied embeddings, cross-entropy | central finite differences (float64) | relative error <= 1.5e-6 |
| `ref_flash_numpy.py` | FlashAttention-2 forward/backward, causal, GQA, tiles that do not divide T | naive attention (float64) | <= 3e-15 |
| `ref_qwen35_numpy.py` | Qwen3.5 hybrid decoder: zero-centered RMSNorm, Gated DeltaNet (conv1d + SiLU, delta rule), gated full attention with partial RoPE | HF logits for 7 prompts (one of 121 tokens), hidden states | logits within 2.6e-5 |
| `ref_sampling_numpy.py` | sampler chain, exact speculative decoding | Monte-Carlo (200k rounds) | deviation < 0.007 |
| `ref_bpe_train.py` | byte-level BPE training (pair counts, inverted index, lazy heap) | Hugging Face `BpeTrainer` | identical tokens/byte (0.3821) |

Goldens live in `tests/golden/` and are produced on the homelab by `tools/gen_*.py` (they need the oracle environment under `~/forgeinfer/oracle` and `~/forgeinfer/venv`):
`tok_*.json`, `fwd_*.json/.f32`, `quant_*.in/.f32`, `chat_*.json`.

Run a spec on the homelab: `tools/hl.sh "~/forgeinfer/venv/bin/python tools/ref_forward_numpy.py smollm2"`.
