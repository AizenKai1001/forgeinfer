# Forge Studio

The desktop app of the ForgeInfer project: chat, model library with a GGUF inspector, Hugging Face downloads, a local OpenAI-compatible server. egui, no web view.

```bash
cargo run --release
```

The guide is [../docs/STUDIO.md](../docs/STUDIO.md); the engine it drives is described in [../docs/ENGINE.md](../docs/ENGINE.md). This folder is its own Cargo workspace: the engine crates in `../crates` use no third-party crates, this app does (egui/eframe, ureq, serde, png).
