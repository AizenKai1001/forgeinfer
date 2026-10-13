# Getting started

This page takes you from nothing to a chat with a local model. About fifteen minutes, most of it compiling.

## 1. What you need

| for | you need |
|---|---|
| Forge Studio (the app) | [Rust](https://rustup.rs) (stable). Nothing else: no CUDA, no Python. |
| the ForgeInfer engine | an NVIDIA GPU of the Turing generation or newer (GTX 16xx, RTX 20xx, 30xx, 40xx, 50xx) and a current driver. To **build** it: the CUDA toolkit (12.x or 13.x, `nvcc`), and on Windows the Visual Studio 2022 C++ build tools. Running it needs the driver only. |
| the engine's CPU/GPU split for models bigger than video memory | a CPU with AVX2 (Intel 4th generation or newer, AMD Zen) and enough system memory for the part of the model that does not fit on the card |
| a Mac, or a PC without an NVIDIA GPU | Forge Studio plus a llama.cpp `llama-server` binary, or any server that is already running (see [STUDIO.md](STUDIO.md#engines)) |

## 2. Build

```bash
git clone https://github.com/AizenKai1001/forgeinfer
cd forgeinfer
```

**Forge Studio**

```bash
cd studio
cargo run --release          # builds and starts the app
```

**The engine, Linux**

```bash
cargo build --release -p forge-cli       # target/release/forge
```

**The engine, Windows** (sets up Visual Studio's compiler environment, checks nvcc, builds with a private target folder):

```powershell
powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1                 # portable images + a native image for an RTX 40 card
powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1 -CudaArchs none # portable images only (smaller, builds faster)
```

The result is `target\release\forge.exe`. [windows_build.md](windows_build.md) explains each step and the common errors.

## 3. Get a model

Any GGUF file of a [supported model](MODELS.md) works. Good first choices for an 8 GB card:

| model | file | size | what to expect |
|---|---|---|---|
| Qwen3.5-0.8B | `unsloth/Qwen3.5-0.8B-GGUF` / `Qwen3.5-0.8B-Q4_K_M.gguf` | 0.5 GB | instant, simple tasks |
| Qwen3.5-4B | `unsloth/Qwen3.5-4B-GGUF` / `Qwen3.5-4B-Q4_K_M.gguf` | 2.7 GB | a good everyday model |
| Qwen3.5-9B | `unsloth/Qwen3.5-9B-GGUF` / `Qwen3.5-9B-Q4_K_M.gguf` | 5.7 GB | the largest that fits an 8 GB card with room for context |
| Qwen3.5-27B | `unsloth/Qwen3.5-27B-GGUF` / `Qwen3.5-27B-Q4_K_M.gguf` | 16.7 GB | needs 32 GB of system memory; the engine keeps part of it in RAM |

In Forge Studio open **Discover**, search, and press Download. Or download any other way and put the file in a folder Studio scans (see **My models**).

## 4. Chat

Open Forge Studio, go to **My models** (or use the model picker at the top of **Chat**), press **Load**. The first load of a big model reads the file from disk and takes a while; the status bar shows the progress. Then type.

Without the app:

```bash
target/release/forge serve models/Qwen3.5-4B-Q4_K_M.gguf --port 1234 --ctx 8192
curl http://127.0.0.1:1234/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "x", "messages": [{"role": "user", "content": "Hello!"}], "stream": false}'
```

Any OpenAI client works against `http://127.0.0.1:1234/v1`.

## 5. Will it fit?

A rough rule for 4-bit files: the file size plus about 15 % plus the KV cache must fit in video memory for full speed.

| video memory | comfortable | works with the CPU/GPU split |
|---|---|---|
| 6 GB | up to 4B | 9B |
| 8 GB | up to 9B | 14B to 27B (slower, needs system memory) |
| 12 GB | up to 14B | 27B to 32B |
| 24 GB | up to 32B | 70B (needs 64 GB of system memory) |

The KV cache (the memory of the conversation) grows with the context length. `--kv int8` halves it with a negligible loss; Studio has the switch in Settings.

## 6. When something goes wrong

- **Studio says it cannot find the `forge` program**: build it (step 2) or set its path in Settings.
- **The model does not load**: open **Logs**. The engine prints the reason (an architecture or tokenizer it does not support, not enough memory, a driver too old).
- **"hybrid (Gated DeltaNet) models run with quantized weights"**: pass `--native` (Studio does this for GGUF files; `forge serve` does it by default for `.gguf`).
- **Windows build fails in `nvcc`**: CUDA 12.x needs Visual Studio 2022 (MSVC 14.4x); a newer compiler is rejected. See [windows_build.md](windows_build.md).
- **Slow answers on a model bigger than video memory**: that is the memory bandwidth of your system memory. Faster RAM (XMP/EXPO enabled) and `--depth 3` on Qwen3.5 models help the most; see [PERFORMANCE.md](PERFORMANCE.md).
