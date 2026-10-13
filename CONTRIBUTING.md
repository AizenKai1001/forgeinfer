# Contributing

Thank you for looking. Bug reports, measurements on other hardware and pull requests are welcome.

## Ground rules

- **The engine crates stay std-only.** Everything under `crates/` uses the Rust standard library and nothing else (the CUDA driver API is loaded with `dlopen` / `LoadLibrary`, the kernels are CUDA C++). A pull request that adds a dependency there will be asked to remove it. Forge Studio (`studio/`) is a separate Cargo workspace and may use well-maintained crates.
- **Say what you measured.** Performance claims need the hardware, the model file, the command and the numbers (see `docs/PERFORMANCE.md` for the style). Correctness claims need a test: the repository has goldens recorded from reference implementations in `tests/golden/`.
- **Keep the tests green.** CPU-only tests run everywhere: `cargo test -p forge-core -p forge-tok -p forge-gguf -p forge-quant -p forge-model -p forge-chat` and, in `studio/`, `cargo test --lib`. Tests that need model files skip themselves when the files are absent (`FORGE_MODELS` points at a folder of models). GPU tests need an NVIDIA GPU.
- **Run untested code with a limit.** A looping test once grew to 30 GB on a shared machine. Use `ulimit -v` or a container when you try new code on a computer that does other work.

## Building

- Engine: `cargo build --release -p forge-cli` needs `nvcc` (CUDA toolkit 12 or 13) and, on Windows, the Visual Studio 2022 C++ tools: `tools/build_windows.ps1` sets the environment up (see `docs/windows_build.md`).
- Studio: `cd studio && cargo run --release`.

## Style

Rust 2021, `cargo fmt` defaults are fine, comments say why rather than what. No `unsafe` outside FFI and the SIMD kernels, and every `unsafe` block carries a `// SAFETY:` line.

## Good first contributions

- Try Forge Studio on macOS and Linux and report what breaks.
- Run `tools/bench_vs_llama.sh` on your GPU and share the table.
- New chat templates (`crates/forge-chat`) and pre-tokenizers (`crates/forge-tok/src/pretok.rs`) with goldens from the reference implementation.
- A CPU backend: see `docs/ROADMAP.md`.
