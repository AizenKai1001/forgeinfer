# ForgeInfer and Forge Studio

**Run language models on your own computer: a desktop app in Rust, and an inference engine written from scratch in Rust and CUDA.**

[![ci](https://github.com/AizenKai1001/forgeinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/AizenKai1001/forgeinfer/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

This repository holds two things that work together:

| | what it is |
|---|---|
| **Forge Studio** (`studio/`) | A desktop app for Windows, Linux and macOS in the style of LM Studio: chat with streaming and Markdown, a model library with a GGUF inspector, Hugging Face search and resumable downloads, a local OpenAI-compatible server, a hardware report, logs and settings. It starts and watches the engine for you. |
| **ForgeInfer** (`crates/`) | The engine behind it: an OpenAI-compatible inference server for GGUF models on NVIDIA GPUs, plus the tokenizer, chat templates, quantizer, trainer and evaluation tools around it. The core crates use **no third-party Rust crates and no llama.cpp code**: the CUDA driver API is loaded at run time and the GPU kernels are our own CUDA C++. |

Forge Studio is not tied to the engine. It also drives llama.cpp's `llama-server`, and it can chat with any server that is already running (Ollama, LM Studio, vLLM, another computer): that is how it works on a Mac today.

## Forge Studio

![Chat](docs/images/studio-chat.png)

- **Chat**: streaming answers, Markdown with code blocks, tables and lists, a folded "Thought for N s" block for thinking models, tokens per second and time to first token under every answer, regenerate, copy, chat history on disk (one JSON file per chat).
- **My models**: finds `.gguf` files in your folders, LM Studio's folder and Ollama's store (used in place, nothing is copied). Every card says what is inside (architecture, parameters, quantization, context) and whether it fits your video memory.
- **Model inspector**: take any GGUF apart: layer map (the hybrid pattern of Qwen3.5 shows up as 3 recurrent layers per attention layer), where the bytes are (feed-forward, attention, embeddings), the quantization mix, every metadata key and tensor.
- **Discover**: search Hugging Face for GGUF repositories, see every file with its quantization, size and a "fits in video memory" badge, download with progress, speed and resume.
- **Local server**: every loaded model is an OpenAI-compatible server (default port 1234, like LM Studio) with copy-paste snippets for curl, Python and other tools. Optional API key and network access.
- **Hardware**, **Logs**, **Settings**: what the computer offers, everything the engine prints, engine choice, context length, KV-cache compression, extra engine arguments.

![Model inspector](docs/images/studio-inspector.png)

See [docs/STUDIO.md](docs/STUDIO.md) for the full guide.

## Quick start

You need [Rust](https://rustup.rs). Forge Studio itself needs nothing else:

```bash
git clone https://github.com/AizenKai1001/forgeinfer
cd forgeinfer/studio
cargo run --release
```

To run models with the ForgeInfer engine you also need an NVIDIA GPU (GTX 16xx / RTX 20xx or newer) with a current driver, and the CUDA toolkit to *build* the engine (`nvcc`; it is not needed to run it):

```bash
cd forgeinfer
cargo build --release -p forge-cli          # target/release/forge   (Windows: tools/build_windows.ps1, see docs/windows_build.md)
target/release/forge serve path/to/model.gguf --port 1234
```

`forge serve` takes the tokenizer and the chat template from the GGUF file itself. Forge Studio finds the `forge` program next to itself, in `target/release/`, or in the `PATH`; you can also set the path in Settings. On a Mac or on a computer without an NVIDIA GPU, pick the "llama.cpp server" engine in Settings (you provide `llama-server`) or "Connect to a running server".

More detail: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## The engine

```bash
forge serve model.gguf --ctx 8192 --kv int8      # OpenAI-compatible: /v1/chat/completions (streaming), /v1/completions, /v1/models, /tokenize ...
forge info model.gguf                            # metadata and tensors of a GGUF file
forge tensor model.gguf blk.10.ffn_up.weight     # statistics and a block of numbers of one weight matrix
forge tokenize model.gguf "any text"             # token ids from the file's own tokenizer
```

- **Models**: `llama`, `qwen3` and `qwen35` (Qwen3.5: hybrid Gated DeltaNet and attention, 0.8B to 27B) architectures with byte-level BPE tokenizers: Qwen3, Qwen3.5, SmolLM2, Llama 3 family. Quantizations F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K to Q6_K. See [docs/MODELS.md](docs/MODELS.md) for what was tested and what is not supported.
- **Speed**: decode 1.2 to 1.7 times llama.cpp on the same files on an RTX 2060 SUPER (one measurement series, details and the prefill caveat in [docs/PERFORMANCE.md](docs/PERFORMANCE.md)); a 4-bit Qwen3.5-27B runs at about 9.7 tokens per second on that 8 GB card by keeping two thirds of the feed-forward layers in system memory and multiplying them with AVX2 kernels inside the CUDA graph.
- **Memory**: KV-cache compression (`--kv int8|int4`), quantized weights read as they are (no conversion to fp16), a CPU/GPU split planner for models bigger than video memory.
- **Speculative decoding** with the model's own multi-token-prediction head (Qwen3.5) and prefix reuse across chat turns, including state snapshots for the recurrent layers.

On a desktop RTX 4060 (8 GB, Windows, `forge.exe` built from this repository, thinking off, short generations, 4-bit files): Qwen3.5-0.8B 363 tokens/s, 4B 80 tokens/s, 9B 45 tokens/s.

## Also in this repository

The engine grew out of a from-scratch research stack: the same repository contains the byte-level BPE tokenizer and its trainer, a hand-written CUDA trainer (forward and backward, fp16, AdamW and Muon) checked against PyTorch, a JAX port for TPUs, the quantizer, the evaluation ladder and the scripts used to train small models (Forge-1, 150M to 365M parameters) on a home GPU, Kaggle TPUs and Modal. Start at [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/RUNBOOK.md](docs/RUNBOOK.md). These parts are research code: they are documented and tested, but they are tuned to the machines they were developed on.

## Repository map

| path | contents |
|---|---|
| `studio/` | Forge Studio (egui). Its own Cargo workspace; the only part that uses third-party crates. |
| `crates/forge-*` | the engine and tooling crates (std only): `forge-core`, `forge-gguf`, `forge-tok`, `forge-chat`, `forge-model`, `forge-quant`, `forge-cuda`, `forge-kernels`, `forge-gpu`, `forge-engine`, `forge-serve`, `forge-train`, `forge-cli` |
| `tpu/` | the JAX trainer and its tests |
| `tools/`, `cloud/` | helper scripts: data preparation, evaluation, benchmarks, the Windows build, Kaggle and Modal jobs (written for the author's machines; read before running) |
| `tests/golden/` | reference results recorded from Hugging Face, PyTorch and NumPy implementations |
| `docs/` | guides and design notes |

## Status and limits

- The engine needs an NVIDIA GPU. A CPU backend (for macOS and for computers without a GPU) and more architectures (Qwen2, Gemma, Mistral, mixture-of-experts) are on the [roadmap](docs/ROADMAP.md); until then Studio uses llama.cpp or an external server for those.
- SentencePiece tokenizers (Llama 2, Mistral 7B v0.1, Gemma) and IQ-quantizations are not supported by the engine.
- The Llama 3 chat format and tokenizer are implemented and unit-tested but have not been checked against real Llama 3 weights yet.
- Prefill (reading a long prompt) is slower than llama.cpp for short prompts on the Qwen3.5 models at the time of the measurement; decode is faster.
- Forge Studio is new. It was tested on Windows 11 with an RTX 4060 and compiles on Linux; macOS is built by CI but has not been run by the author.

## License

[Apache License 2.0](LICENSE). Models are not part of this repository and have their own licenses. See [NOTICE](NOTICE) for acknowledgements (the GGUF format and quantization layouts come from the ggml project).
