# Embeddings

`forge serve` answers OpenAI's `POST /v1/embeddings` for **nomic-embed-text-v1.5** (the embedding model LM Studio ships; architecture `nomic-bert`) and for the BERT family that has a WordPiece vocabulary: **bge**, **all-MiniLM**, **e5** and **gte**, architecture `bert` (checked: `bge-small-en-v1.5`, `bge-base-en-v1.5`, `all-MiniLM-L6-v2`, `e5-small-v2`, `gte-small`; `bge-large`, `all-MiniLM-L12`, `e5-base` and `gte-base` are the same code with other matrix sizes). The GGUF file may have any quantization the engine reads. The encoder is written from scratch like the rest of the engine: a WordPiece tokenizer in `forge-tok`, a CPU encoder in `forge-embed` and a GPU encoder in `forge-embed-gpu` that reuses the tensor-core GEMMs of the engine.

| model | file | size | vector | context | pooling | task prefixes |
|---|---|---:|---:|---:|---|---|
| nomic-embed-text-v1.5 | `nomic-embed-text-v1.5.Q4_K_M.gguf` | 84 MB | 768 | 2,048 | mean | `search_document: ` / `search_query: ` |
| bge-small-en-v1.5 | `bge-small-en-v1.5-q8_0.gguf` | 37 MB | 384 | 512 | `[CLS]` | none (an instruction in front of the question is optional) |
| all-MiniLM-L6-v2 | `all-MiniLM-L6-v2.Q8_0.gguf` | 25 MB | 384 | 512 | mean | none |
| e5-small-v2 | `e5-small-v2.Q8_0.gguf` | 37 MB | 384 | 512 | mean | `passage: ` / `query: ` |
| gte-small | `gte-small.Q8_0.gguf` | 37 MB | 384 | 512 | mean | none |

Pooling comes from the file (`bert.pooling_type`). A BERT-shaped model with another vocabulary (bge-m3 and the multilingual e5 models use SentencePiece) is refused with a message that says so.

## Use it

```bash
# the embedding model on its own (on the GPU when there is one that works, else the CPU)
forge serve nomic-embed-text-v1.5.Q4_K_M.gguf --port 8081

# next to a chat model: one server, two models (the embedder stays on the CPU so that the chat engine keeps the card to itself)
forge serve chat-model.gguf --embedding-model nomic-embed-text-v1.5.Q4_K_M.gguf
# loaded by the first request, on the GPU when it has 700 MB free then, else on the CPU (Forge Studio starts its engine like this): a chat that never asks for an embedding pays nothing
forge serve chat-model.gguf --embedding-model nomic-embed-text-v1.5.Q4_K_M.gguf --embedding-device auto
# on the GPU for sure (+0.4 GB, and an automatic context leaves room for it)
forge serve chat-model.gguf --embedding-model nomic-embed-text-v1.5.Q4_K_M.gguf --embedding-device gpu

curl http://127.0.0.1:8081/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model": "nomic-embed-text-v1.5.Q4_K_M", "input": ["search_query: what is TCP?", "search_document: TCP is a transport protocol."]}'
```

`input` is a string or an array of up to 2,048 strings. The reply is OpenAI's: `data[i].embedding` (a list of 768 floats for nomic-embed-text, 384 for the small BERT models, or with `"encoding_format": "base64"` the little-endian float32 bytes as base64, what the OpenAI SDKs ask for), `usage.prompt_tokens`, and `model`. `dimensions` keeps the first N values and normalises again (nomic-embed-text was trained for 64 to 768: Matryoshka representation learning; the BERT models were not, so cutting their vectors costs more). Vectors have length one, so the dot product is the cosine. A text longer than the model's context (2,048 tokens for nomic-embed-text, 512 for the BERT models) is cut. `/v1/models` lists the embedding model with `meta.type: "embedding"`; `--api-key` protects both routes.

**The task prefix is part of the text.** nomic-embed-text was trained with a prefix per task and the server does not add one: `search_document: ` for the texts you index, `search_query: ` for the questions, `clustering: ` and `classification: ` for those tasks. Without the prefixes retrieval gets noticeably worse. e5 was trained with `passage: ` and `query: `; bge, MiniLM and gte were trained without any (bge-*-en-v1.5 may take the sentence `Represent this sentence for searching relevant passages: ` in front of a question, which made no difference in the test below).

**The scale of the cosine is the model's own.** Related texts score 0.63 to 0.9 with nomic-embed-text and 0.60 to 0.74 with bge-small, but only 0.23 to 0.48 with all-MiniLM, whose unrelated pairs are at 0.1; e5 and gte squeeze every pair into 0.7 to 0.9. A minimum score that suits one model is wrong for another, so Forge Studio picks one per model (measured on this project's own documentation cut into 1,800-character passages: 14 questions with a known answer, 6 about bread and football; the best passage per question, related against unrelated):

| model | related (min / median) | unrelated (median / max) | Studio's minimum |
|---|---|---|---:|
| nomic-embed-text-v1.5 with prefixes | 0.626 / 0.702 | 0.530 / 0.635 | 0.60 |
| bge-small-en-v1.5 | 0.602 / 0.653 | 0.461 / 0.565 | 0.58 |
| all-MiniLM-L6-v2 | 0.227 / 0.375 | 0.137 / 0.160 | 0.20 |
| e5-small-v2 with prefixes | 0.813 / 0.839 | 0.799 / 0.811 | 0.82 |
| gte-small | 0.782 / 0.834 | 0.739 / 0.766 | 0.78 |

For e5 and gte the two ranges touch: the order of the passages says more than the minimum (`tools/embed_calibrate.py`, if you want to measure another model).

`forge embed MODEL.gguf TEXT...` prints the cosines between the texts (and `--json` the vectors, `--ids` the token ids, `--dims N`, `--cpu` / `--gpu`, `--device N`, `--repeat N` to time it).

## What was checked

- **Token ids** equal Hugging Face's `BertTokenizer` (built from the model's own vocabulary) on 23 texts: English, accents, CJK, Korean, emoji, punctuation, control characters, a 180-token paragraph, white space only, the empty string. The GGUF file stores the WordPiece vocabulary the way llama.cpp's converter writes it (word starts as `▁word`, continuations bare); the loader turns it back.
- **Vectors** of the CPU encoder equal a numpy implementation written separately from the architecture description (`tools/ref_nomic_bert.py`, run on the weights of the same file): cosine 1.0000000 on all 23 texts.
- **The GPU encoder** (fp16 weights and activations, fp32 accumulation and residual stream) gives cosine 0.999999 or better to the same references on every text, alone or in a pack, on an RTX 2060 SUPER; `forge-embed-gpu` has tests for that (`FORGE_EMBED_GGUF=... cargo test --release -p forge-embed-gpu`).
- **llama.cpp** (b11517) running the same weights in f32 (the Q4_K_M file converted with `llama-quantize --allow-requantize ... F32`) gives vectors equal to these to five digits (cosine 1.00000) on 22 of the 23 texts, and the same nearest neighbour for every text. The 23rd mixes Hangul and kana: llama.cpp's WordPiece makes 18 tokens of it where Hugging Face's makes 25 (cosine 0.958); this server follows Hugging Face, the tokenizer the model was trained with. On the Q4_K_M file itself llama.cpp's own vectors are 0.9935 (median) from these: its CPU backend rounds every activation vector to 8 bits for the integer dot products, and that rounding is chaotic through twelve layers (two runs of the rounded pipeline whose inputs differ by 1e-7 end 0.997 apart), a 0.5 percent noise that the f32 and fp16 paths here do not have. Both are fine for retrieval; `tools/embed_vs_llama.py` repeats the comparison against a running `llama-server`.
- **The BERT models** are checked against Hugging Face `transformers` itself (`tools/ref_hf_bert.py`: `AutoModel` and `AutoTokenizer` on the official checkpoint, the pooling of its `1_Pooling/config.json`, then normalised), on the 23 texts for bge-small-en-v1.5 and on 10 of them for bge-base-en-v1.5, all-MiniLM-L6-v2, e5-small-v2 (with its `query: ` prefix) and gte-small. Token ids equal the reference on every text. The CPU encoder on the f32 files (bge-small, bge-base, MiniLM) gives cosine 0.9999999 or better on every text, for the `[CLS]` pooling of bge and the mean pooling of MiniLM, and the GPU encoder 0.999999 or better; the Q8_0 files (all four models) 0.999 or better on both (their own rounding, the same on every implementation). The tests are the ones of nomic-embed-text with another golden file: `FORGE_EMBED_GGUF=bge-small-en-v1.5-f32.gguf FORGE_EMBED_GOLDEN=bge_small_en_v15_hf.json cargo test --release -p forge-embed -p forge-embed-gpu`. BERT's activation is the exact `gelu` (x times the normal distribution function, through erf), not the tanh approximation that llama.cpp uses, so the vectors follow Hugging Face's.
- The official `openai` Python package (1.66, which asks for base64 by default and decodes it itself) reads the answers: a batch, `dimensions` and `/v1/models` included.
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

The small BERT models are two to four times faster (same card, `embed_gpu_bench`, f32 files):

| tokens per text | texts in the request | bge-small-en-v1.5 texts/s | tokens/s | all-MiniLM-L6-v2 texts/s | tokens/s |
|---:|---:|---:|---:|---:|---:|
| 26 | 1 | 636 | 16,500 | 1,190 | 31,000 |
| 26 | 32 | 6,232 | 167,000 | 11,721 | 314,000 |
| 120 | 32 | 1,103 | 134,000 | 2,121 | 258,000 |
| 512 | 1 | 98 | 50,000 | 191 | 98,000 |
| 512 | 32 | 110 | 56,500 | 215 | 110,000 |

On the CPU (8 threads, AVX2) bge-small does 2,040 tokens/s and MiniLM-L6 4,080 (nomic-embed-text 790): 32 texts of 121 tokens take 1.9 s and 0.95 s.

## What it does not do (yet)

- Other embedding architectures: Jina (ALiBi positions), XLM-R models with a SentencePiece vocabulary (bge-m3, multilingual-e5), MPNet (all-mpnet-base-v2), models with a dense layer after the pooling (LaBSE), and decoder-based embedders (Qwen3-Embedding, Gemma). The `bert` architecture was added in v0.1.19 (learned positions, biases, GELU); the models in the table at the top have been checked, a cased BERT vocabulary has not.
- Rerankers, sparse and multi-vector outputs, long-context scaling beyond the 2,048 tokens of the file (the original model reaches 8,192 with a rope scaling that the GGUF does not carry).
- A float16 or int8 output format (`encoding_format` is `float` or `base64`).
