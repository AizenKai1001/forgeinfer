# Security

## Reporting a problem

Please report security problems privately through GitHub's "Report a vulnerability" button on the repository's Security tab, not in a public issue. You will get an answer within a few days.

## What to know when you run this

- `forge serve` listens on `127.0.0.1` by default. `--host 0.0.0.0` (or the "let other computers connect" switch in Forge Studio) exposes it to your network: set `--api-key` first. The server has no TLS and no rate limiting; put it behind a reverse proxy if you expose it beyond a trusted network.
- The tokenizer and the GGUF reader parse files from the internet. They are written to reject truncated or oversized input (sizes are checked before allocation) and a fuzz-style test covers the pre-tokenizers, but treat model files from untrusted sources with the same care as any other download.
- Forge Studio talks to the network only when you ask: model searches and downloads from `huggingface.co`, and a server whose address you enter. Chats, settings and downloads stay on your computer.
- The engine starts as a child process of Forge Studio and ends with it (a job object on Windows, a parent-death signal on Linux).
