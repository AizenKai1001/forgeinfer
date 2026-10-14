# Embeddings

`forge serve` answers OpenAI's `POST /v1/embeddings` for **nomic-embed-text-v1.5** (the embedding model LM Studio ships; architecture `nomic-bert`, GGUF file of any quantization the engine reads). The encoder is written from scratch like the rest of the engine: a WordPiece tokenizer in `forge-tok`, a CPU encoder in `forge-embed` and a GPU encoder in `forge-embed-gpu` that reuses the tensor-core GEMMs of the engine.

## Use it

```bash
# the embedding model on its own (on the GPU when there is one that works, else the CPU)
forge serve nomic-embed-text-v1.5.Q4_K_M.gguf --port 8081

# next to a chat model: one server, two models (the embedder stays on the CPU so that the chat engine keeps the card to itself)
forge serve chat-model.gguf --embedding-model nomic-embed-text-v1.5.Q4_K_M.gguf
forge serve chat-model.gguf --embedding-model nomic-embed-text-v1.5.Q4_K_M.gguf --embedding-device gpu   # or on the GPU too (+0.4 GB)

curl http://127.0.0.1:8081/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model": "nomic-embed-text-v1.5.Q4_K_M", "input": ["search_query: what is TCP?", "search_document: TCP is a transport protocol."]}'
```

`input` is a string or an array of up to 2,048 strings. The reply is OpenAI's: `data[i].embedding` (a list of 768 floats, or with `"encoding_format": "base64"` the little-endian float32 bytes as base64, what the OpenAI SDKs ask for), `usage.prompt_tokens`, and `model`. `dimensions` keeps the first N values and normalises again (the model was trained for 64 to 768: Matryoshka representation learning). Vectors have length one, so the dot product is the cosine. A text longer than the model's context (2,048 tokens) is cut. `/v1/models` lists the embedding model with `meta.type: "embedding"`; `--api-key` protects both routes.

**The task prefix is part of the text.** nomic-embed-text was trained with a prefix per task and the server does not add one: `search_document: ` for the texts you index, `search_query: ` for the questions, `clustering: ` and `classification: ` for those tasks. Without the prefixes retrieval gets noticeably worse.

`forge embed MODEL.gguf TEXT...` prints the cosines between the texts (and `--json` the vectors, `--ids` the token ids, `--dims N`, `--cpu` / `--gpu`, `--device N`, `--repeat N` to time it).

## What was checked

- **Token ids** equal Hugging Face's `BertTokenizer` (built from the model's own vocabulary) on 23 texts: English, accents, CJK, Korean, emoji, punctuation, control characters, a 180-token paragraph, white space only, the empty string. The GGUF file stores the WordPiece vocabulary the way llama.cpp's converter writes it (word starts as `▁word`, continuations bare); the loader turns it back.
- **Vectors** of the CPU encoder equal a numpy implementation written separately from the architecture description (`tools/ref_nomic_bert.py`, run on the weights of the same file): cosine 1.0000000 on all 23 texts.
- **The GPU encoder** (fp16 weights and activations, fp32 accumulation and residual stream) gives cosine 0.999999 or better to the same references on every text, alone or in a pack, on an RTX 2060 SUPER; `forge-embed-gpu` has tests for that (`FORGE_EMBED_GGUF=... cargo test --release -p forge-embed-gpu`).
- **llama.cpp** (b11517) running the same weights in f32 (the Q4_K_M file converted with `llama-quantize --allow-requantize ... F32`) gives vectors equal to these to five digits (cosine 1.00000) on 22 of the 23 texts, and the same nearest neighbour for every text. The 23rd mixes Hangul and kana: llama.cpp's WordPiece makes 18 tokens of it where Hugging Face's makes 25 (cosine 0.958); this server follows Hugging Face, the tokenizer the model was trained with. On the Q4_K_M file itself llama.cpp's own vectors are 0.9935 (median) from these: its CPU backend rounds every activation vector to 8 bits for the integer dot products, and that rounding is chaotic through twelve layers (two runs of the rounded pipeline whose inputs differ by 1e-7 end 0.997 apart), a 0.5 percent noise that the f32 and fp16 paths here do not have. Both are fine for retrieval; `tools/embed_vs_llama.py` repeats the comparison against a running `llama-server`.
- The route is tested over HTTP against the golden vectors (`tools/embed_route_check.py`): every text alone, a batch of 23, base64, `dimensions`, the error replies, on the CPU and on the GPU.

## Speed

One pass over a pack of texts (`input` as a list): the tokens of all texts are laid one after the other in one matrix, with no padding, so that the tensor-core GEMMs see thousands of rows even when every text is a few dozen tokens long. Measured with `cargo run --release -p forge-embed-gpu --example embed_gpu_bench -- MODEL.gguf`, nomic-embed-text-v1.5 Q4_K_M, on an RTX 2060 SUPER:

| tokens per text | texts in the request | time | texts/s | tokens/s |
|---:|---:|---:|---:|---:|
| 26 | 1 | 3.0 ms | 332 | 8,600 |
| 26 | 32 | 13.5 ms | 2,373 | 63,600 |
| 26 | 128 | 56.9 ms | 2,251 | 70,000 |
| 120 | 1 | 5.1 ms | 197 | 23,700 |
| 120 | 32 | 71 ms | 450 | 54,600 |
| 514 | 1 | 21 ms | 47 | 24,200 |
| 514 | 32 | 617 ms | 52 | 26,800 |
| 1,023 | 1 | 65 ms | 15 | 15,800 |
| 1,023 | 32 | 2.07 s | 15 | 15,900 |

On an RTX 4060 (the same kernels: the Ampere-tuned GEMMs are not used yet) the same table reads 8,000 / 80,000 tokens/s for 26-token texts (1 / 32 per request), 66,000 for 120 tokens, 30,000 for 514 and 17,900 for 1,023; 23 texts (633 tokens) take 13 ms through HTTP. The vectors are the same on both cards, digit for digit.

The CPU encoder (8 threads, AVX2) does about 790 tokens/s: 32 texts of 121 tokens take 4.9 s. Short texts are limited by the GEMMs, long ones by the attention kernel (one block per query row and head on the CUDA cores: it grows with the square of the text length and is the next thing to rewrite for the tensor cores). Through HTTP a request adds a millisecond or two: 23 texts (633 tokens) took 19 ms on the GPU and 960 ms on the CPU.

The weights take 275 MB of video memory (fp16) and the working buffers 120 MB; the server loads the GPU encoder in about a second.

## What it does not do (yet)

- Other embedding architectures: BERT-family files (`bert`: all-MiniLM, bge, e5, gte), Jina, XLM-R (bge-m3), and decoder-based embedders (Qwen3-Embedding, Gemma). Nomic-bert is the one LM Studio ships by default and the one `forge-embed` is validated against; the encoder structure is shared, so the others are a matter of the layer variants (learned positions, GELU, biases).
- Rerankers, sparse and multi-vector outputs, long-context scaling beyond the 2,048 tokens of the file (the original model reaches 8,192 with a rope scaling that the GGUF does not carry).
- A float16 or int8 output format (`encoding_format` is `float` or `base64`).
