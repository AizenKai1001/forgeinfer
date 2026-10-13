# Forge Studio

Forge Studio is the desktop app of this project: one window to find, download, load and chat with local models, and to serve them to other programs. It is written in Rust with [egui](https://github.com/emilk/egui) (the `glow` OpenGL backend), has no web view and no background service, and keeps everything in one folder.

```bash
cd studio
cargo run --release
```

Release binaries for Windows, Linux and macOS are attached to the GitHub releases (built by `.github/workflows/release.yml`).

## The screens

### Chat

- Pick a model in the bar at the top; **Load** starts the engine on it. The bar shows the load time when it is ready.
- Answers stream in. Markdown is rendered: headings, bold and italic, inline code, fenced code blocks (with a Copy button), lists (nested), tables, quotes, links (click to open). A reply that is still streaming is rendered as it grows; an unclosed code fence is a code block up to the end.
- Thinking models (Qwen3, Qwen3.5) show a folded **Thought for N s** block above the answer; it is open while the model is still thinking. The switch is in **Parameters**.
- Under each answer: tokens, tokens per second (the server's own figure when it reports one), time to first token; **Copy**, **Regenerate**, **Delete**.
- **Parameters** (the button at the top right): temperature, top P, top K, min P, repeat penalty, max tokens, thinking (*model default*, always, or answer directly: the small Qwen3.5 models answer directly by default, the bigger ones think), the system prompt (global, and one for this chat only).
- Chats are saved after every answer, one JSON file each in the data folder; the list on the left opens, deletes and starts chats. **Copy chat** puts the conversation on the clipboard as Markdown.
- Enter sends, Shift+Enter adds a line, **Stop** cancels (the engine stops generating as soon as the connection closes). Under the input box: how much of the loaded context the conversation uses.

### My models

Lists every `.gguf` file under the folders in **Model folders** (the app's own `models` folder, LM Studio's folder if it exists) and every model in Ollama's store (`~/.ollama/models`, read through its manifests). Nothing is copied or moved. Each card shows the architecture, parameter count, quantization, file size, trained context, where it came from, and a badge: *fits in video memory*, *runs split between GPU and CPU*, or *too large*. A model the ForgeInfer engine cannot run says why (architecture or tokenizer) when you hover the disabled **Load** button. Drop a `.gguf` file or a folder on the window to add its folder. A Qwen3.5 card shows **prediction head: faster answers** when the head file is found, or a **Get the prediction head** button that fetches it (see below). **Inspect** opens the inspector.

### The inspector

Reads the GGUF header (no weights are loaded):

- **Summary**: architecture, parameters, layers, trained context, vocabulary, quantization, bits per weight.
- **Layer map**: one bar per layer, as tall as its weights, coloured by kind. A Qwen3.5 file shows its hybrid design: three recurrent *Gated DeltaNet* layers for every full *attention* layer.
- **What the bytes are** and **Quantization mix**: stacked bars (feed-forward, attention, output head, embeddings ...; which tensor formats hold how many bytes). Mixed files such as `Q4_K_M` really are mixes.
- **Metadata** (every key) and **Tensors** (name, format, shape, size).

The command line has the same information: `forge info model.gguf`, and `forge tensor model.gguf NAME` prints statistics and a block of numbers of one matrix.

### Discover

Search Hugging Face (`filter=gguf`, sorted by downloads). Pick a repository to list its `.gguf` files with their quantization, a one-line note on quality, size, and a fit badge. **Download** saves to `<download folder>/<owner>/<repo>/<file>` (the layout LM Studio uses, so both programs find it), with progress, speed and time left. A cancelled or interrupted download leaves a `.part` file and **resumes** with an HTTP range request. Split files (`-00001-of-00003`) must be downloaded part by part; projector files for vision (`mmproj`) are labelled.

No account is needed. Gated repositories that require a login are not supported yet.

### Local server

Whatever is loaded is served on `http://127.0.0.1:<port>/v1` (default port **1234**, the one LM Studio uses, so tools that expect it work; if it is busy the next free port is used). The screen shows ready-to-paste `curl`, Python (`openai` package) and generic settings. **Let other computers connect** makes the server listen on every interface; set an API key first.

### Hardware, Logs, Settings

- **Hardware**: processor, threads, memory, GPUs (name, video memory total and in use, driver), a table of how typical model sizes fit this computer, and which engine programs were found.
- **Logs**: the command line that started the engine and everything it prints (load steps, errors).
- **Settings**: see below.

## Engines

Settings → Engine picks what runs the models:

| engine | runs | needs |
|---|---|---|
| **ForgeInfer engine (CUDA)** | GGUF models (`llama`, `qwen3`, `qwen35` architectures) on an NVIDIA GPU, with KV-cache compression, the CPU/GPU split and speculative decoding | the `forge` program: next to the app, in `target/release/`, in the `PATH`, `FORGE_BIN`, or the path in Settings |
| **llama.cpp server** | any GGUF llama.cpp supports, on any GPU or the CPU; works on macOS | `llama-server` from a [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases) in the `PATH` or in Settings |
| **Connect to a running server** | nothing is started: Studio talks to the address you enter | an OpenAI-compatible server (Ollama on `:11434`, LM Studio on `:1234`, vLLM, the engine on another computer) |

Settings → When a model loads: context length, KV-cache mode (16-bit, 8-bit, 4-bit), speculative depth (Qwen3.5 models draft tokens with their own prediction head; 2 to 3 is fastest), extra engine arguments, "load the last model when the app starts", and **Unload after N idle minutes** (frees the video memory for games; any request to the local server counts as activity).

## Speculative decoding (the prediction head)

Qwen3.5 models answer 1.5 to 2.2 times faster with their prediction head (measurements in [PERFORMANCE.md](PERFORMANCE.md#speculative-decoding-on-the-rtx-4060)). The GGUF files do not contain it. Studio looks for `<model name>-mtp.safetensors` (the name without the quantization: `Qwen3.5-9B-mtp.safetensors` for `Qwen3.5-9B-Q4_K_M.gguf`) beside the model, in an `mtp` folder beside it, in your model folders and in the download folder, and starts the engine with `--mtp FILE --depth 3` when it finds one (Settings > When a model loads has the switch). **Get the prediction head** on a model's card downloads only the `mtp.*` tensors of the original checkpoint on Hugging Face (`Qwen/Qwen3.5-9B` and so on, 0.2 to 0.9 GB) into `<download folder>/mtp/`. If the engine cannot use a head (a file of another model), Studio starts the model again without it. The status bar of the chat shows **speculative** while it is on.

## Where things are

| what | where |
|---|---|
| settings, chats | Windows `%APPDATA%\ForgeStudio`, macOS `~/Library/Application Support/ForgeStudio`, Linux `~/.config/forge-studio` (override: `FORGE_STUDIO_HOME`) |
| portable mode | create a folder named `data` next to the program and everything (settings, chats, downloads) is kept there instead |
| downloads | `<data folder>/models` unless you change it in Settings |

## Command line

The window itself needs no options. These exist for tests and screenshots:

```
forge-studio --screen library|chat|discover|server|hardware|logs|settings
             --demo                          a sample conversation (nothing is saved)
             --load FILE.gguf                load a model at start      --select NAME  load the library model whose name contains NAME
             --send "text"                   send a message as soon as a model is ready
             --search "text"                 run a Hugging Face search at start
             --inspect FILE.gguf             open the inspector on a file
             --shot out.png [--shot-delay S] [--shot-after-reply]    save a screenshot and quit
             --size 1280x800
```

## How it is built

`studio/` is its own Cargo workspace (the engine crates stay free of third-party crates). Modules:

| module | job |
|---|---|
| `app.rs` | all state; worker threads report back through one channel; the screens only draw state and queue `Action`s that are applied after the frame |
| `engine.rs` | spawns `forge serve` / `llama-server`, collects its output, polls `/v1/models` until it answers, kills it on exit; the child ends with the app (Windows job object, Linux `PR_SET_PDEATHSIG`) |
| `client.rs` | OpenAI-compatible client: model list and streamed chat completions (SSE), including `reasoning_content`, usage and timings; retries with only the standard fields if a strict server rejects the extra samplers |
| `library.rs` | scans folders and Ollama's store, reads GGUF headers with the `forge-gguf` crate, builds the inspector's data |
| `hf.rs` | Hugging Face search, file listing, resumable download |
| `markdown.rs` | Markdown parser and egui renderer (links are hit-tested on the text layout) |
| `hardware.rs` | CPU, memory, GPUs (`nvidia-smi`), fit estimates |
| `chats.rs`, `settings.rs` | files on disk |
| `ui/` | one file per screen |

Tests: `cd studio && cargo test --lib` (31 unit tests, including a synthetic GGUF file for the scanner and inspector). `cargo test --lib live_ -- --ignored` runs a network test that searches, lists, downloads and resumes a 1.2 MB model from Hugging Face.

## Platform notes

- **Windows**: tested on Windows 11. Release builds have no console window.
- **Linux**: built and unit-tested by CI and compile-checked on Ubuntu 24.04 (X11 and Wayland through `winit`, OpenGL through `glutin`, no system libraries needed to build); needs a working OpenGL driver at run time. The window has not been run on a Linux desktop by the author.
- **macOS**: built and unit-tested by the CI workflow (Apple silicon), the window itself has not been run by the author. The default engine there is llama.cpp's `llama-server` (the ForgeInfer engine needs CUDA); or connect to a running server.
